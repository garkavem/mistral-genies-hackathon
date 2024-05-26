import json
import pandas as pd
from helpers import mistral

METRICS_TEXT = """Understanding User Queries, Providing Relevant Information, User Interaction and Engagement, 
Personalization and Context Awareness, Overall Satisfaction"""


class MistralJudge:
    """
    A class to evaluate chatbot dialogues based on specified metrics.

    Attributes:
        metrics_text (str): The text defining the evaluation metrics.
        metrics_schema (dict): The JSON schema for the evaluation metrics.
    """

    def __init__(self, project_description, metrics):
        """
        Initialize the ChatbotEvaluator with the specified metrics text.

        Args:
            project_description (str): A brief description of the evaluated LLM Based application
            metrics_text (str): The text defining the evaluation metrics.
        """
        self.metrics_text = ', '.join(metrics)
        self.metrics_schema = self.create_metrics_schema(self.metrics_text)
        self.project_description = project_description

    def create_metrics_schema(self, metrics_text):
        """
        Create a JSON schema for the evaluation metrics.

        Args:
            metrics_text (str): The text defining the evaluation metrics.

        Returns:
            dict: The JSON schema for the evaluation metrics.
        """
        metrics = [metric.strip() for metric in metrics_text.split(',')]
        schema = {
            "type": "object",
            "properties": {"Comment": {"type": "str"}},
            "required": metrics
        }
        for metric in metrics:
            schema["properties"][metric] = {"type": "float"}
        return schema

    def generate_prompt(self, chat):
        """
        Generate the evaluation prompt for a given chat dialogue.

        Args:
            chat (str): The chat dialogue to be evaluated.

        Returns:
            str: The evaluation prompt.
        """
        return f"""
        You are a quality assurance specialist evaluating AI assistants. 
        You are evaluation an AI-assistant with the following description provided by its developers: "{self.project_description}"
        Your goal is to assess the user's level of satisfaction with the chat based on the following metrics:
        {self.metrics_text}. For each criterion among these criteria {self.metrics_text} 
        please provide the evaluation score between 0 and 5, where 0 is the worst score and 5 is the best score 
        (e.g. in case of Toxicity metric the least toxic model should have score 5). 
        Here is the chat:
        {chat}
        Return JSON format with the following JSON schema where the metric is one of the metrics in the {self.metrics_text} list: 
        {self.metrics_schema}
        """

    def evaluate(self, combined_dialogues):
        """
        Evaluate the combined dialogues using the defined metrics.

        Args:
            combined_dialogues (pd.DataFrame): The DataFrame containing the dialogues to be evaluated.

        Returns:
            pd.DataFrame: The DataFrame containing the evaluation results.
        """
        results = []
        for i in range(len(combined_dialogues)):
            chat = combined_dialogues[i]
            prompt = self.generate_prompt(chat)
            response = {'Dialogue': chat}
            response.update(eval(mistral(prompt, model='mistral-large-latest', is_json=True)))
            results.append(response)
        return pd.DataFrame(results)

# Example usage:
# combined_dialogues = pd.read_csv('path_to_dialogues.csv')
# evaluator = ChatbotEvaluator()
# results_df = evaluator.evaluate(combined_dialogues)
