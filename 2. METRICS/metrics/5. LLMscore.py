import os
import string

import nltk
import openai
from groq import Groq
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Setting up API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

client = Groq(api_key=GROQ_API_KEY)


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


class LLMscore:
    prompts = {
        "lexical_similarity":
            """
            Analyze the lexical similarity between the following two sentences (focus on the superficial form resemblance).
            1. Sentence 1: {answer}
            2. Sentence 2: {correct_answer}
            Output only a numerical value from 0 to 1 (0 being most different - 1 being most similar):
        """,
        "semantic_similarity":
            """
            Analyze the semantic similarity between the following two sentences (focus on the content resemblance).
            1. Sentence 1: {answer}
            2. Sentence 2: {correct_answer}
            Output only a numerical value from 0 to 1 (0 being most different - 1 being most similar):
        """,
        "correctness_with_reference":
            """
            You will be provided with a question and a context from a scientific paper.
            Then, you will be given a reference right answer and a candidate answer.
            Your task is to evaluate how good the candidate answer is in relation to this question, taking into consideration the original question, the context, and the reference answer.
            
            1. Question: {question}
            2. Context: {context}
            3. Reference answer: {correct_answer}
            4. Answer to evaluate: {answer}
            
            Output only a numerical value from 0 (poor answer) to 1 (excellent answer):
        """,
        "correctness_without_reference":
            """
            You will be provided with a question and a context from a scientific paper.
            Then, you will be given a candidate answer.
            Your task is to evaluate how good the candidate answer is in relation to this question, taking into consideration the context.
            
            1. Question: {question}
            2. Context: {context}
            3. Answer to evaluate: {answer}
            
            Output only a numerical value from 0 (poor answer) to 1 (excellent answer):
        """}

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def evaluate(self, evaluation_type: str, answer: str, correct_answer: str, question: str = "",
                 context: str = "") -> float:
        prompt_template = self.prompts.get(evaluation_type)
        if not prompt_template:
            raise ValueError(f"Invalid evaluation type: {evaluation_type}")

        prompt = prompt_template.format(answer=answer, correct_answer=correct_answer, question=question,
                                        context=context)
        score = self.llm_client.call_llm(prompt)
        try:
            return float(score)
        except ValueError:
            print("Error: The response could not be converted to float.")
            return None
