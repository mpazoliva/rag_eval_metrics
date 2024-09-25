import os
import string
from abc import abstractmethod
from typing import Dict, Iterable, List, Optional, Tuple, Union

import nltk
import numpy as np
from bert_score import score
from iris_ml.algorithms.document_fingerprint import select_concepts
from iris_ml.algorithms.wisdm import WISDM
from iris_ml.interface.keyword_extraction_interface import KeywordExtractor
from iris_ml.interface.word_embeddings_interface import WordAnalyzer
from iris_ml.tools.models_loaders.hierarchy_models_loaders import AbstractsModels
from nltk.corpus import stopwords
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer, util

from evaluation_module.metrics.base import SupervisedMetric
from evaluation_module.utils import evaluation_cached_method

DEFAULT_SENTENCE_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_POS_TAG_WEIGHTS = {"NN": 2, "NNS": 2, "NNP": 2}

abstracts_model = None


class MoversDsitance(SupervisedMetric):
    """General implementation of the movers distance similarity measure between two sets of strings. The algorithm 
    conssits of two steps. First, compute the cost matrix which represents the distance between each string from the first
    set and each string from the second set (e.g. using cosine or Eucledean distance). Second, use the linear sum 
    assignment algorithm from scipy to find the optimal one-to-one mapping between the elements of the two sets.
    """

    @abstractmethod
    def get_embeddings(self, texts) -> Iterable[np.array]:
        ...

    @abstractmethod
    def parse_text(self, text: str) -> Union[List[str], Tuple[List[str], List[float]]]:
        ...

    def distance_measure(self, embedding1: np.array, embedding2: np.array) -> float:
        """Distance measure, could be overriden if cosine is not appropriate"""
        return cosine(embedding1, embedding2)

    def calculate_cost_matrix(self, set1: List[str], set2: List[str], **kwargs) -> np.array:
        """
        Calculate the cost matrix of the sets of strings, taking into account embeddings and optional weights

        Args:
            texts1: first set of strings
            texts2: second set of strings
            weights1 (Optional[numpy.array]): the relative weights of the elements of the first set
            weights2 (Optional[numpy.array]): the relative weights of the elements of the second set

        Returns:
            An numpy array of the distnaces/cost between elements of the two sets of strings
        """
        embeddings1 = self.get_embeddings(set1)
        embeddings2 = self.get_embeddings(set2)

        cost_matrix = np.ones((len(set1), len(set2)))

        for i, embedding1 in enumerate(embeddings1):
            for j, embedding2 in enumerate(embeddings2):
                if embedding1 is not None and embedding2 is not None:
                    cost_matrix[i, j] = self.distance_measure(embedding1, embedding2)

        return cost_matrix

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes the movers distance between and answer and a correct answer
        """
        answer_texts = self.parse_text(answer)
        correct_answer_texts = self.parse_text(correct_answer)

        if type(answer_texts) is tuple:
            weights1 = answer_texts[1]
            answer_texts = answer_texts[0]
        else:
            weights1 = None
        if type(correct_answer_texts) is tuple:
            weights2 = correct_answer_texts[1]
            correct_answer_texts = correct_answer_texts[0]
        else:
            weights2 = None

        weights = None

        if weights1 and weights2:
            weights = np.array([[w1 + w2 for w2 in weights2] for w1 in weights1])
        elif weights1:
            weights = np.array([[w1 for _ in correct_answer_texts] for w1 in weights1])
        elif weights2:
            weights = np.array([[w2 for w2 in weights2] for _ in answer_texts])

        cost_matrix = self.calculate_cost_matrix(answer_texts, correct_answer_texts)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        movers_distances = cost_matrix[row_ind, col_ind]

        if weights is not None:

            weights = weights[row_ind, col_ind]
            weights = weights / np.sum(weights)

            movers_distance = np.sum(movers_distances * weights)

        else:
            movers_distance = np.mean(movers_distances)

        return movers_distance


class WordMoversDistance(MoversDsitance):
    """
    Computes the Word Movers Distance score between two texts.
    """
    __name__ = "WMD"

    def __init__(self, word_analyzer: Optional[WordAnalyzer] = None, language: str = "english",
                 pos_tag_weights: Optional[Dict[str, str]] = None):

        global abstracts_model

        if word_analyzer is None:
            if not abstracts_model:
                if "ASBTRACT_MODELS" in os.environ:
                    abstracts_model = AbstractsModels(os.getenv("ASBTRACT_MODELS"))
                else:
                    raise Exception("Models not specified and ASBTRACT_MODELS is not set.")
            word_analyzer = abstracts_model.word_analyzer

        self.word_analyzer = word_analyzer
        self.language = language
        self.pos_tag_weights = pos_tag_weights if pos_tag_weights else DEFAULT_POS_TAG_WEIGHTS

    def get_embeddings(self, texts) -> Iterable[np.array]:
        embeddings = [self.word_analyzer.get_ngram_representation(text) for text in texts]
        embeddings = [embd if type(embd) is not str else None for embd in embeddings]

        return embeddings

    def parse_text(self, text: str) -> Union[List[str], Tuple[List[str], List[float]]]:
        # text = text.lower()
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words(self.language))
        tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
        # Count token frequencies
        token_freq = {}
        for token in tokens:
            if token in token_freq:
                token_freq[token] += 1
            else:
                token_freq[token] = 1

        # Normalize frequencies
        total_count = sum(token_freq.values())
        token_freq = {token: count / total_count for token, count in token_freq.items()}

        pos_tags = nltk.pos_tag(list(token_freq.keys()))
        pos_tag_weights = np.array([
            self.pos_tag_weights[pos_tag[1]] if pos_tag[1] in self.pos_tag_weights else 1 for pos_tag in pos_tags])
        weights = np.array(list(token_freq.values())) * pos_tag_weights

        return (list(token_freq.keys()), list(weights))


class WordMoversSimilarity(WordMoversDistance):
    __name__ = "WordMoversSimilarity"

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes the word movers similarity between and answer and a correct answer
        """
        sms_score = 1 - super().evaluate(answer, correct_answer)
        return sms_score


class SentenceMoversDistance(MoversDsitance):
    """
    Computes the Sentence Movers Distance score between two texts.
    """
    __name__ = "SMD"

    def __init__(self, sentence_embeddings_model: str = DEFAULT_SENTENCE_EMBEDDINGS_MODEL):
        self.model = SentenceTransformer(sentence_embeddings_model)

    def get_embeddings(self, texts) -> Iterable[np.array]:
        return self.model.encode(texts)

    def parse_text(self, text: str) -> Union[List[str], Tuple[List[str], List[float]]]:

        return nltk.sent_tokenize(text)


class SentenceMoversSimilarity(SentenceMoversDistance):
    __name__ = "SentenceMoversSimilarity"

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes the sentence movers similarity between and answer and a correct answer
        """
        wmd_score = 1 - super().evaluate(answer, correct_answer)
        return wmd_score


class Wisdm(SupervisedMetric):

    def __init__(self, word_analyzer: Optional[WordAnalyzer] = None,
                 keyword_extractor: Optional[KeywordExtractor] = None):
        global abstracts_model
        if not word_analyzer or not keyword_extractor:
            if not abstracts_model:
                if "ABSTRACT_MODELS" in os.environ:
                    abstracts_model = AbstractsModels(os.getenv("ABSTRACT_MODELS"))
                else:
                    raise Exception("Models not specified and ABSTRACT_MODELS is not set.")
            if not word_analyzer:
                word_analyzer = abstracts_model.word_analyzer
            if not keyword_extractor:
                keyword_extractor = abstracts_model.extractor
        self.wisdm = WISDM(word_analyzer, keyword_extractor)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes the wisdm score between two strings.
        """
        concepts1 = select_concepts(self.wisdm.extract_concepts_from_text(answer, percentage_to_display=1))
        concepts2 = select_concepts(self.wisdm.extract_concepts_from_text(correct_answer, percentage_to_display=1))
        score = self.wisdm.compute_similarity_btwn_concept_lists(concepts1, concepts2)
        wisdm_score = float(score)
        return wisdm_score


class BertScore(SupervisedMetric):

    def __init__(self, model_name=DEFAULT_SENTENCE_EMBEDDINGS_MODEL):
        self.model = SentenceTransformer(model_name)

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes the sentence score between two strings: 
        Get sentences' embeddings with fixed model.
        Computes cosine similarity among embeddings.
        """
        embeddings_answer = self.model.encode(answer, convert_to_tensor=True)
        embeddings_correct_answer = self.model.encode(correct_answer, convert_to_tensor=True)
        score = util.pytorch_cos_sim(embeddings_answer, embeddings_correct_answer)
        bertsentence_score = score.item()
        return bertsentence_score
