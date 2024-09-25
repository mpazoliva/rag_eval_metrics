import os
import re
import string
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import nltk
import openai
import pandas as pd
import requests
from groq import Groq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


@dataclass
class QAPair:
    question: str
    ground_truth: str
    correct_similar: str
    correct_different: str
    incorrect_similar: str
    incorrect_related: str
    incorrect_unrelated: Optional[str] = None


class LLMClient:

    def __init__(self, developer: str, model_name: str):
        self.developer = developer
        self.model_name = model_name
        self.stop_words = set(stopwords.words('english') + list(string.punctuation))

    def call_llm(self, prompt: str) -> str:
        """Calls the specified LLM and returns the response or an error."""
        try:
            if self.developer == "meta":
                chat_completion = client.chat.completions.create(messages=[{
                    "role": "user",
                    "content": prompt}], model=self.model_name)
                return chat_completion.choices[0].message.content
            elif self.developer == "openai":
                response = openai.Completion.create(engine=self.model_name, prompt=prompt, max_tokens=150,
                                                    temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0)
                return response.choices[0].text.strip()
        except Exception as e:
            return str(e)


class PubMedRetriever:
    # CONFIGURATION FILE OR CONSTANTS
    PUBMED_SEARCHURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, pubmed_apikey: str):
        self.pubmed_apikey = pubmed_apikey

    def get_abstracts(self, query: str, num_abstracts: int = 10) -> List[str]:
        """Fetches abstracts from PubMed based on a query.
            Returns a list with abstracts as strings."""
        abstracts = []
        retrieved_ids = set()
        attempts = 0

        while len(abstracts) < num_abstracts and attempts < 5:
            fetch_count = num_abstracts - len(abstracts) + 5

            search_url = self.PUBMED_SEARCHURL
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": fetch_count,
                "apikey": self.pubmed_apikey,
                "retmode": "json"}

            search_response = requests.get(search_url, params=search_params)
            if search_response.status_code != 200:
                return [f"Failed to retrieve data: {search_response.text}"]

            try:
                ids = search_response.json()['esearchresult']['idlist']
                ids = [id for id in ids if id not in retrieved_ids]
                retrieved_ids.update(ids)
            except Exception as e:
                return [f"Error parsing JSON response: {str(e)}"]

            fetch_url = self.FETCH_URL
            fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "apikey": self.pubmed_apikey}

            fetch_response = requests.get(fetch_url, params=fetch_params)
            if fetch_response.status_code != 200:
                return [f"Failed to retrieve abstracts: {fetch_response.text}"]

            xml_data = fetch_response.text
            root = ET.fromstring(xml_data)

            for article in root.findall('.//PubmedArticle'):
                abstract_text = article.find('.//AbstractText')
                if abstract_text is not None:
                    abstracts.append(abstract_text.text)
                    if len(abstracts) >= num_abstracts:
                        break

            attempts += 1

        return abstracts


class DataElement(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def definition(self) -> str:
        pass

    def generate(self, *inputs: str, max_attempts: int = 10) -> str:
        if max_attempts <= 0:
            print(ValueError("Maximum attempts reached"))
            return None

        generation = self.generate_(*inputs)

        if self.validate(generation, *inputs):
            return generation
        else:
            return self.generate(*inputs, max_attempts=max_attempts - 1)

    @abstractmethod
    def validate(self, *inputs: str) -> bool:
        pass

    @abstractmethod
    def generate_(self, prompt_template: str, *inputs: str) -> str:
        pass


class QuestionFactChecking(DataElement):

    def __init__(self, llm_client: LLMClient, generation_prompt, validation_prompt):
        self.llm_client = llm_client
        self.generation_prompt = generation_prompt
        self.validation_prompt = validation_prompt

    @property
    def name(self) -> str:
        return "Question Fact Checking"

    def definition(self) -> str:
        return "The answer is a composition based on only one substring of the context, that can be used directly as it is written in the context."

    def generate_(self, abstract: str) -> str:
        prompt = self.generation_prompt.format(abstract=abstract)
        generation = self.llm_client.call_llm(prompt)
        question_match = re.search(r'(\bWhat\b.*?[\?\!])', generation, re.IGNORECASE)
        if question_match:
            return question_match.group(1).strip()
        else:
            return generation.strip()

    def validate(self, question: str, abstract: str) -> bool:
        prompt = self.validation_prompt.format(question=question, abstract=abstract)
        validation = self.llm_client.call_llm(prompt)
        return validation == "Fact Checking"


class QuestionReasoning(DataElement):

    def __init__(self, llm_client: LLMClient, generation_prompt, validation_prompt):
        self.llm_client = llm_client
        self.generation_prompt = generation_prompt
        self.validation_prompt = validation_prompt

    @property
    def name(self) -> str:
        return "Question Reasoning"

    def definition(self) -> str:
        return "The answer is a composition based on a more elaborated understanding made from more than one substring of the context. Substrings cannot be used exactly as they are written; it requires a deeper understanding of the content."

    def generate_(self, abstract: str) -> str:
        prompt = self.generation_prompt.format(abstract=abstract)
        generation = self.llm_client.call_llm(prompt)
        question_match = re.search(r'(\bWhat\b.*?[\?\!])', generation, re.IGNORECASE)
        if question_match:
            return question_match.group(1).strip()
        else:
            return generation.strip()

    def validate(self, question: str, abstract: str) -> bool:
        prompt = self.validation_prompt.format(question=question, abstract=abstract)
        validation = self.llm_client.call_llm(prompt)
        return validation == "Reasoning"


class AnswerGroundTruth(DataElement):

    def __init__(self, llm_client: LLMClient, generation_prompt):
        self.llm_client = llm_client
        self.generation_prompt = generation_prompt

    @property
    def name(self) -> str:
        return "Answer Ground Truth"

    def definition(self) -> str:
        return "Right answer generated by model based on the abstract."

    def generate_(self, question: str, abstract: str) -> str:
        prompt = self.generation_prompt.format(question=question, abstract=abstract)
        return self.llm_client.call_llm(prompt)

    def validate(self, *inputs) -> bool:
        return True


class AnswerCorrectSimilar(DataElement):

    def __init__(self, llm_client: LLMClient, generation_prompt):
        self.llm_client = llm_client
        self.generation_prompt = generation_prompt

    @property
    def name(self) -> str:
        return "Answer Correct Similar"

    def definition(self) -> str:
        return "Correct and similar to ground truth. Shares +3 words with ground truth."

    def generate_(self, ground_truth: str, *inputs) -> str:
        prompt = self.generation_prompt.format(ground_truth=ground_truth)
        return self.llm_client.call_llm(prompt)

    def validate(self, answer: str, ground_truth: str, *inputs) -> bool:
        stop_words = set(stopwords.words('english'))
        tokens1 = [word.lower() for word in word_tokenize(ground_truth) if word.lower() not in stop_words]
        tokens2 = [word.lower() for word in word_tokenize(answer) if word.lower() not in stop_words]
        common_tokens = set(tokens1).intersection(tokens2)
        return len(common_tokens) >= 3


class AnswerCorrectDifferent(DataElement):

    def __init__(self, llm_client: LLMClient, generation_prompt):
        self.llm_client = llm_client
        self.generation_prompt = generation_prompt

    @property
    def name(self) -> str:
        return "Answer Correct Different"

    def definition(self) -> str:
        return "Correct, but different to ground truth. Does not share any word with ground truth, it can use synonyms."

    def generate_(self, ground_truth: str, *inputs) -> str:
        prompt = self.generation_prompt.format(ground_truth=ground_truth)
        return self.llm_client.call_llm(prompt)

    def validate(self, answer: str, ground_truth: str, *inputs) -> bool:
        stop_words = set(stopwords.words('english'))
        punctuations = set(string.punctuation)
        tokens1 = [
            word.lower() for word in word_tokenize(ground_truth)
            if word.lower() not in stop_words and word not in punctuations]
        tokens2 = [
            word.lower() for word in word_tokenize(answer)
            if word.lower() not in stop_words and word not in punctuations]
        common_tokens = set(tokens1).intersection(tokens2)
        return len(common_tokens) < 3


class AnswerIncorrectSimilar(DataElement):

    def __init__(self, llm_client: LLMClient, generation_prompt, validation_prompt):
        self.llm_client = llm_client
        self.generation_prompt = generation_prompt
        self.validation_prompt = validation_prompt

    @property
    def name(self) -> str:
        return "Answer Incorrect Similar"

    def definition(self) -> str:
        return "Wrong but similar to ground truth. Shares almost all words with ground truth, only differs by 1-3 key terms. Information is incorrect: opposite to truth or complete nonsense."

    def generate_(self, ground_truth: str, *inputs) -> str:
        prompt = self.generation_prompt.format(ground_truth=ground_truth)
        return self.llm_client.call_llm(prompt)

    def validate(self, answer: str, question: str, ground_truth: str) -> bool:
        answer_type = "Incorrect Similar"
        prompt = self.validation_prompt.format(question=question, ground_truth=ground_truth, answer=answer)
        validation = self.llm_client.call_llm(prompt)
        return validation == answer_type


class AnswerIncorrectRelated(DataElement):

    def __init__(self, llm_client: LLMClient, generation_prompt, validation_prompt):
        self.llm_client = llm_client
        self.generation_prompt = generation_prompt
        self.validation_prompt = validation_prompt

    @property
    def name(self) -> str:
        return "Answer Incorrect Related"

    def definition(self) -> str:
        return "Wrong but related to the topic. Information is correct, but it is not the answer to the question."

    def generate_(self, ground_truth: str, *inputs) -> str:
        prompt = self.generation_prompt.format(ground_truth=ground_truth)
        return self.llm_client.call_llm(prompt)

    def validate(self, answer: str, question: str, ground_truth: str) -> bool:
        answer_type = "Incorrect Related"
        prompt = self.validation_prompt.format(question=question, ground_truth=ground_truth, answer=answer)
        validation = self.llm_client.call_llm(prompt)
        return validation == answer_type


class AnswerIncorrectUnrelated(DataElement):

    def __init__(self, llm_client: LLMClient, generation_prompt, validation_prompt):
        self.llm_client = llm_client
        self.generation_prompt = generation_prompt
        self.validation_prompt = validation_prompt

    @property
    def name(self) -> str:
        return "Answer Incorrect Unrelated"

    def definition(self) -> str:
        return "Wrong and completely unrelated to the topic. Information is correct, but it is not related to the topic of the question."

    def generate_(self, ground_truth: str, *inputs) -> str:
        prompt = self.generation_prompt.format(ground_truth=ground_truth)
        return self.llm_client.call_llm(prompt)

    def validate(self, answer: str, question: str, ground_truth: str) -> bool:
        answer_type = "Incorrect Unrelated"
        prompt = self.generation_prompt.format(question=question, ground_truth=ground_truth, answer=answer)
        validation = self.llm_client.call_llm(prompt)
        return validation == answer_type


class GenerateCompletePair:

    def __init__(self, llm_client: LLMClient, prompts: Dict[str, str]):
        self.llm_client = llm_client

        self.generation_prompt_factchecking = prompts["prompt_factchecking"]
        self.generation_prompt_reasonig = prompts["prompt_reasoning"]
        self.validation_prompt_questions = prompts["prompt_validate_question"]
        self.generation_prompt_groundtruth = prompts["prompt_ground_truth"]
        self.generation_prompt_correct_similar = prompts["prompt_correct_similar"]
        self.generation_prompt_correct_different = prompts["prompt_correct_different"]
        self.generation_prompt_incorrect_similar = prompts["prompt_incorrect_similar"]
        self.generation_prompt_incorrect_related = prompts["prompt_incorrect_related"]
        self.validation_prompt_incorrect = prompts["prompt_validate_incorrect"]

        self.questions = {
            "fact_checking":
                QuestionFactChecking(llm_client, self.generation_prompt_factchecking, self.validation_prompt_questions),
            "reasoning":
                QuestionReasoning(llm_client, self.generation_prompt_reasonig, self.validation_prompt_questions)}
        self.ground_truth = AnswerGroundTruth(llm_client, self.generation_prompt_groundtruth)
        self.modifications = {
            "correct_similar":
                AnswerCorrectSimilar(llm_client, self.generation_prompt_correct_similar),
            "correct_different":
                AnswerCorrectDifferent(llm_client, self.generation_prompt_correct_different),
            "incorrect_similar":
                AnswerIncorrectSimilar(llm_client, self.generation_prompt_incorrect_similar,
                                       self.validation_prompt_incorrect),
            "incorrect_related":
                AnswerIncorrectRelated(llm_client, self.generation_prompt_incorrect_related,
                                       self.validation_prompt_incorrect)}

    def generate_complete_pair(self, abstract: str, question_type: str, max_attempts: int = 30) -> Optional[QAPair]:

        if question_type not in self.questions:
            raise ValueError(f"Unsupported question type: {question_type}")

        question = self.questions[question_type].generate(abstract, max_attempts=max_attempts)

        if question is None:
            print(f"Failed to generate a valid question for type: {question_type}")
            return None

        ground_truth = self.ground_truth.generate(question, abstract, max_attempts=max_attempts)

        answers = {}
        for key, element in self.modifications.items():
            answer = element.generate(ground_truth, question, max_attempts=max_attempts)
            if answer is None:
                print(f"Failed to generate a valid answer for modification: {key}")
                return None
            answers[key] = answer

        return QAPair(question=question, ground_truth=ground_truth, correct_similar=answers["correct_similar"],
                      correct_different=answers["correct_different"], incorrect_similar=answers["incorrect_similar"],
                      incorrect_related=answers["incorrect_related"])


class SensitivityDatasetGenerator:

    def __init__(self, llm_client: LLMClient, pubmed_apikey: str, prompts: Dict[str, str], save_path: str,
                 save_frequency: int = 10):
        self.llm_client = llm_client
        self.pubmed_retriever = PubMedRetriever(pubmed_apikey)
        self.generate_pair = GenerateCompletePair(llm_client, prompts)
        self.save_path = save_path
        self.save_frequency = save_frequency
        os.makedirs(save_path, exist_ok=True)

    def save_progress(self, data: List[Dict], abstracts_dict: Dict[int, str], filename: str):
        df = pd.DataFrame(
            data, columns=[
                'query', 'abstract_id', 'question_type', 'question', 'ground_truth', 'correct_similar',
                'correct_different', 'incorrect_similar', 'incorrect_related', 'incorrect_unrelated'])
        df.to_csv(os.path.join(self.save_path, f"{filename}_data.csv"), index=False)
        pd.to_pickle(abstracts_dict, os.path.join(self.save_path, f"{filename}_abstracts.pkl"))

    def generate_sensitivity_dataset(self, queries: List[str], abs_per_query: int = 10,
                                     question_per_abs: int = 6) -> Tuple[pd.DataFrame, Dict[int, str]]:
        data = []
        abstract_id = 0
        ground_truths_by_query = {query: [] for query in queries}
        abstracts_dict = {}

        max_retries = 10

        for query in queries:
            pairs_for_query = 0
            generated_pairs = 0

            while pairs_for_query < abs_per_query * question_per_abs:
                print(f"Working with query: {query}")
                abstracts = self.pubmed_retriever.get_abstracts(query, num_abstracts=abs_per_query)
                for abstract in abstracts:
                    success = False
                    retry_count = 0
                    while not success and retry_count < max_retries:
                        print(f"Working with abstract: {abstract_id}")
                        abstracts_dict[abstract_id] = abstract
                        success = True
                        local_pairs_generated = 0

                        for _ in range(question_per_abs // 2):
                            qa_fc_pair = self.generate_pair.generate_complete_pair(abstract, "fact_checking")
                            qa_r_pair = self.generate_pair.generate_complete_pair(abstract, "reasoning")

                            if qa_fc_pair:
                                print("A fact checking set was created.")
                                ground_truths_by_query[query].append(qa_fc_pair.ground_truth)
                                data.append({
                                    'query': query,
                                    'abstract_id': abstract_id,
                                    'question_type': "Fact Checking",
                                    'question': qa_fc_pair.question,
                                    'ground_truth': qa_fc_pair.ground_truth,
                                    'correct_similar': qa_fc_pair.correct_similar,
                                    'correct_different': qa_fc_pair.correct_different,
                                    'incorrect_similar': qa_fc_pair.incorrect_similar,
                                    'incorrect_related': qa_fc_pair.incorrect_related,
                                    'incorrect_unrelated': ''})
                                local_pairs_generated += 1
                                pairs_for_query += 1
                            else:
                                success = False
                                break

                            if qa_r_pair:
                                print("A reasoning set was created.")
                                ground_truths_by_query[query].append(qa_r_pair.ground_truth)
                                data.append({
                                    'query': query,
                                    'abstract_id': abstract_id,
                                    'question_type': "Reasoning",
                                    'question': qa_r_pair.question,
                                    'ground_truth': qa_r_pair.ground_truth,
                                    'correct_similar': qa_r_pair.correct_similar,
                                    'correct_different': qa_r_pair.correct_different,
                                    'incorrect_similar': qa_r_pair.incorrect_similar,
                                    'incorrect_related': qa_r_pair.incorrect_related,
                                    'incorrect_unrelated': ''})
                                local_pairs_generated += 1
                                pairs_for_query += 1
                            else:
                                success = False
                                break

                            if pairs_for_query >= abs_per_query * question_per_abs:
                                break

                        if success:
                            generated_pairs += local_pairs_generated
                            abstract_id += 1
                        else:
                            retry_count += 1
                            print(f"Retry attempt {retry_count} for abstract {abstract_id}")

                        if pairs_for_query >= abs_per_query * question_per_abs or success:
                            break

                    if pairs_for_query >= abs_per_query * question_per_abs:
                        break

                    if generated_pairs % self.save_frequency == 0 and generated_pairs > 0:
                        self.save_progress(data, abstracts_dict, f"intermediate_{generated_pairs}")

        print("All the answers' sets were successfully created.")
        print("Adding Incorrect Unrelated.")
        for entry in data:
            current_query = entry['query']
            other_queries = [query for query in queries if query != current_query]
            unrelated_ground_truth = None

            for other_query in other_queries:
                if ground_truths_by_query[other_query]:
                    unrelated_ground_truth = ground_truths_by_query[other_query].pop(0)
                    ground_truths_by_query[other_query].append(unrelated_ground_truth)
                    break

            if unrelated_ground_truth is None:
                unrelated_ground_truth = "No unrelated ground truth found."

            entry['incorrect_unrelated'] = unrelated_ground_truth

        self.save_progress(data, abstracts_dict, "final")

        df = pd.DataFrame(
            data, columns=[
                'query', 'abstract_id', 'question_type', 'question', 'ground_truth', 'correct_similar',
                'correct_different', 'incorrect_similar', 'incorrect_related', 'incorrect_unrelated'])

        return df, abstracts_dict
