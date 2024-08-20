from transformers import pipeline

class UserPromptUnderstanding:
    def __init__(self):
        # Load a pre-trained Transformer model for NLP tasks
        self.nlp_model = pipeline("text-classification", model="bert-base-uncased")

    def process_prompt(self, prompt):
        # Process the user prompt and extract key information
        result = self.nlp_model(prompt)
        return result
