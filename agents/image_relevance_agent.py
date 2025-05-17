import os
import time
from typing import List, Dict
import requests
from dotenv import load_dotenv
load_dotenv()
class ImageRelevanceAgent:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initializes the ImageRelevanceAgent with access to Groq-hosted LLaMA-3 model.
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set. Please load your .env file or set the variable.")

        self.model = model_name
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def classify_images(self, chunks: List[Dict]) -> List[Dict]:
        """
        Annotates each chunk's images with a classification: 'augment_metadata' or 'irrelevant'.
        :param chunks: List of document chunks containing optional 'images' fields.
        :return: Modified chunks with classified image actions.
        """
        for chunk in chunks:
            text = chunk.get("text", "")
            images = chunk.get("images", [])
            for image in images:
                description = image.get("description", "")
                prompt = self._build_prompt(text, description)
                try:
                    label = self._query_groq_model(prompt)
                    # Normalize label to expected outputs
                    label = label.lower()
                    if label not in ["augment_metadata", "irrelevant"]:
                        print(f"[Warning] Unexpected label '{label}' received. Defaulting to 'irrelevant'.")
                        label = "irrelevant"
                    image["action"] = label
                except Exception as e:
                    print(f"[ImageRelevanceAgent] Error classifying image: {e}")
                    image["action"] = "unknown"
                time.sleep(1.2)  # Prevent rate limit issues
        return chunks

    def _build_prompt(self, context_text: str, image_description: str) -> str:
        return (
            "You are an AI assistant classifying image relevance in documents.\n"
            f"Context: {context_text}\n"
            f"Image Description: {image_description}\n"
            "Is the image relevant to the context and should it be used to augment the final answer?\n"
            "Respond with one of exactly: augment_metadata or irrelevant."
        )

    def _query_groq_model(self, prompt: str) -> str:
        """
        Calls the Groq LLaMA model API to classify the image relevance.
        :param prompt: Prompt to pass to the model.
        :return: One of: 'augment_metadata', 'irrelevant'
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 10
        }

        response = requests.post(self.endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Groq API error: {response.status_code} - {response.text}")

        completion = response.json()
        try:
            content = completion["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Groq API response format: {completion}") from e

        return content
