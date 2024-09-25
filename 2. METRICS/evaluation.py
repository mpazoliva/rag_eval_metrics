import csv
import json
import logging
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from chat_tool.model_backend.conversation.conversation import Conversation
from iris_io.clients.generative_models.chatbot_model import ChatBotModel
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from evaluation_module.data_types import GenerationEvaluationResult, RAGExample
from evaluation_module.llm import HuggingFaceLLM
from evaluation_module.metrics import *
from evaluation_module.metrics.base import LLMMetric, Metric

logger = logging.getLogger(__name__)


@dataclass
class EvaluationDataset:
    name: str
    question_field: str
    context_field: str
    correct_answer_field: str
    data: List[Dict]


@dataclass
class EvaluationPipelineConfig:
    metrics: List[str]
    variables: List[str]


class EvaluationPipeline(object):

    def __init__(self, config: EvaluationPipelineConfig):
        self.config = config
        for var in self.config.variables:
            os.environ[var] = self.config.variables[var]

        self._metrics = []

    def init(self, model: HuggingFaceLLM, prompt_tpl: str):
        for metric_name in self.config.metrics:
            print(f"Initializing {metric_name}...", end="")
            sys.stdout.flush()

            metric_class = eval(metric_name)
            if issubclass(metric_class, LLMMetric):
                if metric_class is ContextSensitivity:
                    metric = metric_class(model=model, prompt_tpl=prompt_tpl)
                else:
                    metric = metric_class(model=model)
            else:
                metric = metric_class()
            self._metrics.append(metric)
            print("Done")

    def run(self, **kwargs):
        scores = {}
        for metric in self._metrics:
            scores[metric.name] = metric.evaluate(**kwargs)
        return scores


class GenerationEvaluation(object):

    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    def read_data(self, data):
        """
        Read RAG examples.
        """
        examples = []
        for row in data:
            example = RAGExample(question=row.get('question'), answer=row['answer'],
                                 correct_answer=row.get('correct_answer', None), context=row.get('context', None))
            examples.append(example)
        return examples

    def read_input(self, file_path: str):
        """
        Receives a .csv or .json file and returns the necessary ExampleData.
        """
        logger.info("Reading your input file as example data.")
        examples = []

        if file_path.endswith(".csv"):
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                examples.extend(self.read_data(csv.DictReader(csvfile)))
        elif file_path.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as json_file:
                examples.extend(self.read_data(json.load(json_file)))
        else:
            raise ValueError("Unsupported file format. Only .csv and .json files are supported.")

        logger.info("Input data read sucessfully.")
        return examples

    def evaluate_data(self, data: List[RAGExample]) -> GenerationEvaluationResult:
        """
        Evalutates data in which 'answer' has already been populated. 
        """

        # Dictionaries to store each metric as key with the values of examples.
        scores_raw = {}
        scores_mean = {}
        scores_std = {}
        scores_min = {}
        scores_max = {}

        # List to store all values computed for an example with each metric.
        invalid_index = []

        for m in self.metrics:
            logger.info("Computing values for metric: " + m.name)
            metric_scores = []
            questions = []
            contexts = []
            ground_truths = []
            answers = []
            for example in data:

                question = example.question
                questions.append(question)
                answer = example.answer
                answers.append(answer)
                correct_answer = example.correct_answer
                ground_truths.append(correct_answer)
                context = example.context
                contexts.append(context)

                try:
                    score = m.evaluate(answer=answer, correct_answer=correct_answer, question=question, context=context)
                    metric_scores.append(score)
                except:
                    score = None
                    metric_scores.append(score)

            # Save index of invalid computation
            for index, score in enumerate(metric_scores):
                if not isinstance(score, float) or score is None or np.isnan(score):
                    invalid_index.append(index)
                else:
                    pass
            # Clean nan values and error messages
            metric_scores_statistics = [
                x for x in metric_scores if isinstance(x, float) and not np.isnan(x) and x is not None]
            # Calculate mean
            mean_value = statistics.mean(metric_scores_statistics)
            # Calculate standard deviation
            std_value = statistics.stdev(metric_scores_statistics)
            # Calculate minimum
            min_value = min(metric_scores_statistics)
            # Calculate maximum
            max_value = max(metric_scores_statistics)

            # answers
            scores_raw[m.name] = metric_scores
            scores_mean[m.name] = mean_value
            scores_std[m.name] = std_value
            scores_min[m.name] = min_value
            scores_max[m.name] = max_value

            logger.info(f"Computation of {m.name} is done.")

        #Compute correlations
        logger.info("Computing correlation among all your metrics.")
        scores_corr = {}
        filtered_scores_raw = {
            key: [value for i, value in enumerate(values) if i not in invalid_index]
            for key, values in scores_raw.items()}

        for metric1, metric2 in combinations(filtered_scores_raw.keys(), 2):
            correlation_coefficient, _ = spearmanr(filtered_scores_raw[metric1], filtered_scores_raw[metric2])
            scores_corr[(metric1, metric2)] = correlation_coefficient

        def discretize_scores(scores, bins=3, labels=None):
            if labels is None:
                labels = list(range(1, bins + 1))
            categorized_scores = np.digitize(scores, bins=np.linspace(min(scores), max(scores), num=bins), right=True)
            return [labels[idx - 1] for idx in categorized_scores]

        discretized_scores = {metric: discretize_scores(scores) for metric, scores in filtered_scores_raw.items()}

        scores_kappa = {}
        for (metric1, metric2) in combinations(discretized_scores.keys(), 2):
            kappa = cohen_kappa_score(discretized_scores[metric1], discretized_scores[metric2])
            scores_kappa[(metric1, metric2)] = kappa

        # Creating an instance of GenerationEvaluationResult and returning it
        evaluation_result = GenerationEvaluationResult(questions=questions, contexts=contexts,
                                                       ground_truths=ground_truths, answers=answers,
                                                       scores_raw=scores_raw, scores_mean=scores_mean,
                                                       scores_std=scores_std, scores_min=scores_min,
                                                       scores_max=scores_max, scores_corr=scores_corr,
                                                       scores_kappa=scores_kappa)
        logger.info("The evaluation of your data was sucessfull.")
        return evaluation_result

    @staticmethod
    def save_results(result: List[GenerationEvaluationResult], output_path: str):
        logger.info("Saving your results in an external file.")
        if output_path.endswith(".json"):
            serialized_result = {
                "answer": result.answers,
                "scores_raw": {
                    str(metric): scores for metric, scores in result.scores_raw.items()},
                "scores_mean": {
                    str(metric): mean for metric, mean in result.scores_mean.items()},
                "scores_std": {
                    str(metric): std for metric, std in result.scores_std.items()},
                "scores_min": {
                    str(metric): minimum for metric, minimum in result.scores_min.items()},
                "scores_max": {
                    str(metric): maximum for metric, maximum in result.scores_max.items()},
                "scores_corr": {
                    str(metric_tuple): coefficient for metric_tuple, coefficient in result.scores_corr.items()}}

            with open(output_path, 'w') as file:
                json.dump(serialized_result, file, indent=4)

        elif output_path.endswith(".csv"):
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write headers
                writer.writerow(["Metric", "Answer", "Score Raw", "Score Mean", "Score Std", "Score Min", "Score Max"])
                # Iterate over each metric
                for metric in result.scores_raw.keys():
                    scores_raw = result.scores_raw.get(metric, [])
                    score_mean = result.scores_mean.get(metric, "")
                    score_std = result.scores_std.get(metric, "")
                    score_min = result.scores_min.get(metric, "")
                    score_max = result.scores_max.get(metric, "")
                    # Write data for each answer and its corresponding metrics
                    for i, answer in enumerate(result.answers):
                        writer.writerow([metric, answer, scores_raw[i], score_mean, score_std, score_min, score_max])
        else:
            raise ValueError("Unsupported output format. Only .json and .csv files are supported.")

        logger.info("Results were sucessfully saved.")


DEFAULT_RETRIEVAL_TOP_K_VALUES = list(range(1, 11)) + list(range(20, 101, 10))
