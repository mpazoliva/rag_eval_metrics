from typing import Optional

import nltk
from rouge import Rouge
from sacrebleu.metrics import BLEU, CHRF, TER

nltk.download('punkt')
from nltk.translate.meteor_score import meteor_score

from evaluation_module.metrics.base import SupervisedMetric


class Rouge1(SupervisedMetric):
    __name__ = "Rouge1"

    def evaluate(self, answer: str, correct_answer: Optional[str] = "", **kwargs) -> float:
        """
        Computes the Rouge 1 score (unigrams matching ) between two strings. 
        Uses Rouge library.
        """
        rouge = Rouge()
        score = rouge.get_scores(answer, correct_answer, avg=True)
        rouge_score = score['rouge-1']['f']

        return rouge_score


class Rouge2(SupervisedMetric):
    __name__ = "Rouge2"

    def evaluate(self, answer: str, correct_answer: Optional[str] = "", **kwargs) -> float:
        """
        Computes the rouge 2 score (bigrams matching) between two strings. 
        Uses Rouge library.
        """
        rouge = Rouge()
        score = rouge.get_scores(answer, correct_answer, avg=True)
        rouge_score = score['rouge-2']['f']

        return rouge_score


class RougeL(SupervisedMetric):
    __name__ = "RougeL"

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes the rouge L score (longest string matching) between two strings. 
        Uses Rouge library.
        """
        rouge = Rouge()
        score = rouge.get_scores(answer, correct_answer, avg=True)
        rouge_score = score['rouge-l']['f']

        return rouge_score


class BleuBaseMetric(SupervisedMetric):

    def __init__(self, bleu_metric):
        self.bleu_metric = bleu_metric

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes some Blue version score between two strings. 
        """
        score = self.bleu_metric.sentence_score(answer, [correct_answer]).score
        return score


class Bleu(BleuBaseMetric):

    def __init__(self):
        super().__init__(BLEU(effective_order=True))


class Chrf(BleuBaseMetric):

    def __init__(self):
        super().__init__(CHRF())


class ChrfPlus(BleuBaseMetric):

    def __init__(self):
        super().__init__(CHRF(word_order=2))


class Ter(BleuBaseMetric):

    def __init__(self):
        super().__init__(TER())

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes TER score between two strings and inverts it.
        """
        score = self.bleu_metric.sentence_score(answer, [correct_answer]).score
        return 1 / (score + 1)


class Meteor(SupervisedMetric):
    __name__ = "Meteor"

    def evaluate(self, answer: str, correct_answer: str, **kwargs) -> float:
        """
        Computes the meteor score between two strings. 
        Uses NLTK library.
        """
        answer_tokens = nltk.word_tokenize(answer)
        correcto_answer_tokens = nltk.word_tokenize(correct_answer)
        score = meteor_score([answer_tokens], correcto_answer_tokens)
        return score
