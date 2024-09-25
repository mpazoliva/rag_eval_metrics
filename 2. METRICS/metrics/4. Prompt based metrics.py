import os
from typing import Optional

import numpy as np
import ragas
import ragas.metrics
from datasets import Dataset
from iris_cache.tools import cached_method
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tonic_validate import Benchmark, LLMResponse, ValidateScorer
from tonic_validate.metrics import AnswerConsistencyMetric, AnswerSimilarityMetric

from evaluation_module.metrics.base import RAGMetric


class RagasMetric(RAGMetric):
    __ragas_metric__ = None

    def __init__(self, openai_key: Optional[str] = ""):
        if openai_key:
            os.environ['OPENAI_API_KEY'] = openai_key

    @property
    def ragas_metric(self):
        return self.__ragas_metric__

    def evaluate(self, answer: str, correct_answer: str, question: str, context: str, **kwargs) -> float:
        return self._evaluate(answer=answer, correct_answer=correct_answer, question=question, context=context,
                              metric_name=self.name, **kwargs)

    @cached_method()
    def _evaluate(self, answer: str, correct_answer: str, question: str, context: str, **kwargs) -> float:

        data = {"question": [question], "answer": [answer], "contexts": [[context]], "ground_truth": [correct_answer]}
        dataset = Dataset.from_dict(data)

        results = ragas.evaluate(dataset, metrics=[self.ragas_metric])

        return results[self.ragas_metric.name]


class RagasFaithfullness(RagasMetric):
    """
    Computes Ragas Faithfullnes score between two strings: answer and context. 
    Uses ragas library.
    """
    __name__ = "RagasFaithFullness"
    __ragas_metric__ = ragas.metrics.faithfulness


class RagasRelevancy(RagasMetric):
    """
    Computes Ragas Relevancy score between two strings: answer and question. 
    Uses ragas library.
    """
    __name__ = "RagasRelevancy"
    __ragas_metric__ = ragas.metrics.answer_relevancy


class RagasSimilarity(RagasMetric):
    """
    Computes Ragas Similarity score between two strings: generated answer and ground truth. 
    Uses ragas library.
    """
    __name__ = "RagasSimilarity"
    __ragas_metric__ = ragas.metrics.answer_similarity


class RagasCorrectness(RagasMetric):
    """
    Computes Ragas Correctness score between two strings: generated answer and ground truth. 
    Uses ragas library.
    """
    __name__ = "RagasCorrectness"
    __ragas_metric__ = ragas.metrics.answer_correctness


class TonicMetric(RAGMetric):
    __tonic_metric__ = None

    def __init__(self, metric_name: str, tonic_metric_class):
        self.__name__ = metric_name
        self.__tonic_metric__ = tonic_metric_class

    def evaluate(self, answer: str, correct_answer: str, question: str, context: str) -> float:
        benchmark = Benchmark(questions=[question], answers=[correct_answer])
        llm_response = LLMResponse(llm_answer=answer, llm_context_list=[context], benchmark_item=benchmark.items[0])

        scorer = ValidateScorer([self.__tonic_metric__()], model_evaluator="gpt-4-turbo")
        run = scorer.score_responses([llm_response])

        scores = run.run_data[0].scores
        score = list(scores.values())[0]
        return score


class TonicSimilarity(TonicMetric):
    """
    Computes Tonic Similarity score between two strings: generated answer and ground truth. 
    Uses tonic library.
    """

    def __init__(self):
        super().__init__("TonicSimilarity", AnswerSimilarityMetric)


class TonicConsistency(TonicMetric):
    """
    Computes Tonic Consistency score between: generated answer and context. 
    Uses tonic library.
    """

    def __init__(self):
        super().__init__("TonicConsistency", AnswerConsistencyMetric)
