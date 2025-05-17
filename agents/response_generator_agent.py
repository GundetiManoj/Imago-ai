from typing import List, Dict, Optional
import os
import requests

class GroqClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided for GroqClient.")
        self.endpoint = "https://api.groq.com/openai/v1/completions"
        
    def generate(self, model, prompt, max_tokens=256, temperature=0.7):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(self.endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Error calling Groq API: {response.text}")
            
        result = response.json()
        return type('Response', (), {
            'generated_text': result.get('choices', [{}])[0].get('text', '')
        })


class ResponseGeneratorAgent:
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        hallucination_threshold: float = 0.7,
    ):
        """
        Initialize the response generator agent with Groq LLaMA client.

        :param model_name: Groq LLaMA model name
        :param api_key: API key for Groq authentication
        :param hallucination_threshold: Confidence threshold above which to abstain
        """
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("API key must be provided for GroqClient.")

        self.client = GroqClient(api_key=api_key)
        self.model_name = model_name
        self.hallucination_threshold = hallucination_threshold

    def generate_response(
        self,
        context_chunks: List[Dict],
        question: str,
        hallucination_scores: Optional[List[float]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate a final response using Groq LLaMA from context chunks and hallucination scores.

        :param context_chunks: List of dicts containing 'text' and optionally 'images' metadata
        :param question: The user question string
        :param hallucination_scores: Optional list of hallucination confidence scores per chunk
        :param max_tokens: Max tokens for generation
        :param temperature: Sampling temperature for generation

        :return: Dict with keys 'response_text', 'abstain', and 'reason'
        """
        # Abstain if hallucination risk too high
        if hallucination_scores:
            max_score = max(hallucination_scores)
            if max_score > self.hallucination_threshold:
                return {
                    "response_text": (
                        "Cannot generate confident response due to uncertainty in provided context. "
                        "Please review the source documents."
                    ),
                    "abstain": True,
                    "reason": f"Hallucination risk detected with max score {max_score:.2f}",
                }

        # Aggregate textual content and image metadata descriptions
        aggregated_text = ""
        for chunk in context_chunks:
            aggregated_text += chunk.get("text", "") + " "
            images = chunk.get("images", [])
            for img in images:
                if img.get("action") == "augment_metadata":
                    desc = img.get("description", "No description")
                    aggregated_text += f"[Image metadata: {desc}] "

        aggregated_text = aggregated_text.strip()

        if not aggregated_text:
            return {
                "response_text": "No sufficient information found to answer the query.",
                "abstain": True,
                "reason": "Aggregated context is empty.",
            }

        # Prepare prompt for Groq LLaMA
        prompt = f"Context:\n{aggregated_text}\n\nQuestion:\n{question}\n\nAnswer:"

        # Call Groq LLaMA model for generation
        try:
            generation_response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            generated_text = generation_response.generated_text.strip()
        except Exception as e:
            return {
                "response_text": "Error generating response from language model.",
                "abstain": True,
                "reason": f"Exception during LLM generation: {str(e)}",
            }

        if not generated_text:
            return {
                "response_text": "Language model returned an empty response.",
                "abstain": True,
                "reason": "Empty response from language model.",
            }

        return {
            "response_text": generated_text,
            "abstain": False,
            "reason": None,
        }