# agents/hallucination_verifier_agent.py

import os
import time
import requests
from typing import List, Dict

class HallucinationVerifierAgent:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initializes the hallucination verifier agent using Groq-hosted LLaMA 3 model.
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.model = model_name
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def score_chunks(self, chunks: List[Dict]) -> List[float]:
        """
        Scores each chunk for hallucination likelihood using Groq model.

        :param chunks: List of context chunks (each should contain 'text')
        :return: List of float hallucination scores (0-1), one per chunk
        """
        scores = []
        for chunk in chunks:
            text = chunk.get("text", "")
            prompt = self._build_prompt(text)
            try:
                score = self._query_groq_model(prompt)
            except Exception as e:
                print(f"[HallucinationVerifierAgent] Error scoring chunk: {e}")
                score = 0.5  # Neutral confidence if error occurs
            scores.append(score)
            time.sleep(1.2)  # Avoid rate-limiting
        return scores

    def score_hallucination(self, query: str, chunks: List[Dict]) -> List[float]:
        """
        Scores hallucination risk for a query and context chunks.
        
        :param query: User query string
        :param chunks: List of context chunks
        :return: List of float scores indicating hallucination risk
        """
        # Call score_chunks with the provided chunks
        return self.score_chunks(chunks)

    def _build_prompt(self, chunk_text: str) -> str:
        return (
            "Evaluate the following document snippet for factual confidence.\n"
            f"Text:\n\"{chunk_text}\"\n\n"
            "On a scale from 0 to 1, how confident are you that this content is accurate and not hallucinated? "
            "Only return the numeric score. For example: 0.85"
        )

    def _query_groq_model(self, prompt: str) -> float:
        """
        Sends prompt to Groq LLaMA model and returns hallucination confidence score.

        :param prompt: The prompt string to classify hallucination risk.
        :return: Float score from 0.0 to 1.0
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        response = requests.post(self.endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Groq API Error {response.status_code}: {response.text}")

        content = response.json()["choices"][0]["message"]["content"].strip()
        try:
            score = float(content)
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
        except ValueError:
            raise ValueError(f"Unexpected response from model: '{content}'")