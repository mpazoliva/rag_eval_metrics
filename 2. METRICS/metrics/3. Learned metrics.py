import math
import os
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from scipy.special import softmax
from transformers import BartForConditionalGeneration, BartTokenizer

from bleurt import score
from evaluation_module.metrics.base import SupervisedMetric


class BleuRT(SupervisedMetric):

    def __init__(self):
        checkpoint_path = "/home/ubuntu/models/BLEURT-20"
        self.scorer = score.BleurtScorer(checkpoint=checkpoint_path)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes BlueRT score between two strings. 
        """
        bleurt_score = self.scorer.score(references=[correct_answer], candidates=[answer])[0]
        return bleurt_score


class BartScore(SupervisedMetric):

    def __init__(self, model_name: Optional[str] = "facebook/bart-large-cnn"):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes BartScore score between two strings. 
        """
        inputs = self.tokenizer(answer, return_tensors="pt", truncation=True, padding=True)
        targets = self.tokenizer(correct_answer, return_tensors="pt", truncation=True, padding=True,
                                 text_target=correct_answer)

        input_ids = inputs.input_ids.to(self.model.device)
        target_ids = targets.input_ids.to(self.model.device)

        outputs = self.model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss.item()

        bart_score = math.exp(-loss)

        return bart_score


class BEM(SupervisedMetric):

    def __init__(self):
        VOCAB_PATH = 'gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt'

        self.vocab_table = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(filename=VOCAB_PATH, key_dtype=tf.string,
                                          key_index=tf.lookup.TextFileIndex.WHOLE_LINE, value_dtype=tf.int64,
                                          value_index=tf.lookup.TextFileIndex.LINE_NUMBER), num_oov_buckets=1)
        self.cls_id, self.sep_id = self.vocab_table.lookup(tf.convert_to_tensor(['[CLS]', '[SEP]']))
        self.tokenizer = text.BertTokenizer(vocab_lookup_table=self.vocab_table, token_out_type=tf.int64,
                                            preserve_unused_token=True, lower_case=True)

        bem_model_url = 'https://tfhub.dev/google/answer_equivalence/bem/1'
        self.bem_model = hub.load(bem_model_url)

    # Function to tokenize and convert inputs to IDs using TensorFlow Text
    def bertify_example(self, question, reference, candidate):
        question = self.tokenizer.tokenize(tf.convert_to_tensor([question])).merge_dims(1, 2)
        reference = self.tokenizer.tokenize(tf.convert_to_tensor([reference])).merge_dims(1, 2)
        candidate = self.tokenizer.tokenize(tf.convert_to_tensor([candidate])).merge_dims(1, 2)

        input_ids, segment_ids = text.combine_segments((candidate, reference, question), self.cls_id, self.sep_id)

        return {'input_ids': input_ids.numpy(), 'segment_ids': segment_ids.numpy()}

    def pad(self, a, length=512):
        return np.append(a, np.zeros(length - a.shape[-1], np.int32))

    def bertify_examples(self, examples):
        input_ids = []
        segment_ids = []
        for example in examples:
            example_inputs = self.bertify_example(example['question'], example['reference'], example['candidate'])
            input_ids.append(self.pad(example_inputs['input_ids']))
            segment_ids.append(self.pad(example_inputs['segment_ids']))

        return {'input_ids': np.stack(input_ids), 'segment_ids': np.stack(segment_ids)}

    def compute_metric(self, answer, correct_answer, question, context):
        examples = [{'question': question, 'reference': correct_answer, 'candidate': answer}]

        inputs = self.bertify_examples(examples)

        raw_outputs = self.bem_model(inputs)

        bem_score = float(softmax(np.squeeze(raw_outputs))[1])

        return bem_score

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes BEM score between two strings.
        """
        question = kwargs.get("question", "")
        context = kwargs.get("context", "")
        return self.compute_metric(answer, correct_answer, question, context)
