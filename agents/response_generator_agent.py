# agents/response_generator_agent.py

from typing import List, Dict, Optional
import os
import requests
import re
from transformers import pipeline

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
        
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=10)
        if response.status_code != 200:
            raise Exception(f"Error calling Groq API: {response.text}")
            
        result = response.json()
        return type('Response', (), {
            'generated_text': result.get('choices', [{}])[0].get('text', '')
        })


class ResponseGeneratorAgent:
    def __init__(
        self,
        model_name: str = "groq-llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        hallucination_threshold: float = 0.85,
        use_local_fallback: bool = True
    ):
        """
        Initialize the response generator agent with Groq LLaMA client.

        :param model_name: Groq LLaMA model name
        :param api_key: API key for Groq authentication
        :param hallucination_threshold: Confidence threshold above which to abstain
        :param use_local_fallback: Whether to use local model as fallback
        """
        # Set up API client if possible
        self.use_api = False
        try:
            api_key = api_key or os.getenv("GROQ_API_KEY")
            if api_key:
                self.client = GroqClient(api_key=api_key)
                self.use_api = True
            else:
                print("[WARNING] No API key provided - will use local fallback model only")
        except Exception as e:
            print(f"[WARNING] Could not initialize API client: {e}")

        self.model_name = model_name
        self.hallucination_threshold = hallucination_threshold
        
        # Initialize local fallback model if requested
        self.use_local_fallback = use_local_fallback
        self.local_model = None
        if use_local_fallback:
            try:
                print("[INFO] Loading local fallback model...")
                # Use a small local model for fallback
                self.local_model = pipeline(
                    "text-generation",
                    model="databricks/dolly-v2-3b",  # You can use a smaller model too
                    device_map="auto",  # Use GPU if available
                    torch_dtype="auto"
                )
                print("[INFO] Local fallback model loaded successfully")
            except Exception as e:
                print(f"[WARNING] Could not load local fallback model: {e}")
                print("[INFO] Trying to load a smaller model...")
                try:
                    # Try an even smaller model
                    self.local_model = pipeline(
                        "text-generation", 
                        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        device_map="auto"
                    )
                    print("[INFO] Smaller local fallback model loaded")
                except Exception as e2:
                    print(f"[WARNING] Could not load smaller fallback model: {e2}")

    def generate_response(
        self,
        context_chunks: List[Dict],
        question: str,
        hallucination_scores: Optional[List[float]] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate a final response using available models and fallbacks.

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

        # Prepare prompt for generation
        prompt = f"Context:\n{aggregated_text}\n\nQuestion:\n{question}\n\nAnswer:"

        # First try using the API if available
        if self.use_api:
            try:
                print("[INFO] Generating response with Groq API...")
                generation_response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                generated_text = generation_response.generated_text.strip()
                
                if generated_text:
                    return {
                        "response_text": generated_text,
                        "abstain": False,
                        "reason": None,
                    }
            except Exception as e:
                print(f"[WARNING] API generation failed: {e}")
                # Continue to fallback if API fails

        # If API failed or wasn't available, try local model
        if self.local_model:
            try:
                print("[INFO] Using local fallback model for generation...")
                # Truncate context if too long for local model
                if len(prompt) > 2048:
                    # Keep question intact but truncate context
                    question_part = f"Question:\n{question}\n\nAnswer:"
                    context_limit = 2048 - len(question_part) - 20
                    truncated_context = aggregated_text[:context_limit] + "..."
                    prompt = f"Context:\n{truncated_context}\n\n{question_part}"
                
                # Generate with local model
                local_response = self.local_model(
                    prompt,
                    max_length=len(prompt.split()) + 100,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.local_model.tokenizer.eos_token_id
                )
                
                # Extract just the generated part (after the prompt)
                generated_text = local_response[0]['generated_text']
                if len(generated_text) > len(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                else:
                    generated_text = "Based on the provided context, " + self._extract_answer_manually(context_chunks, question)
                
                return {
                    "response_text": generated_text,
                    "abstain": False,
                    "reason": None,
                }
            except Exception as e:
                print(f"[WARNING] Local generation failed: {e}")
        
        # If everything failed, generate a simple extractive answer
        try:
            print("[INFO] Falling back to rule-based response extraction...")
            extracted_answer = self._extract_answer_manually(context_chunks, question)
            
            return {
                "response_text": f"Based on the provided context: {extracted_answer}",
                "abstain": False,
                "reason": None,
            }
        except Exception as e:
            # Last resort - if absolutely everything fails
            return {
                "response_text": "Unable to generate a response due to technical issues.",
                "abstain": True,
                "reason": f"All generation methods failed: {str(e)}",
            }

    def _extract_answer_manually(self, chunks: List[Dict], question: str) -> str:
        """
        Extract a simple answer from chunks when all generation methods fail.
        
        :param chunks: Context chunks
        :param question: User question
        :return: Simple extracted answer
        """
        # Extract question keywords
        question_lower = question.lower()
        question_words = set(re.findall(r'\b\w{4,}\b', question_lower))
        
        best_chunk = None
        best_score = -1
        
        for chunk in chunks:
            text = chunk.get("text", "")
            text_lower = text.lower()
            
            # Count keyword matches
            score = sum(1 for word in question_words if word in text_lower)
            
            # Specific question types
            if "what is" in question_lower and "is a" in text_lower:
                score += 2
            if "how to" in question_lower and any(w in text_lower for w in ["step", "instruction", "guide"]):
                score += 2
            if "purpose" in question_lower and any(w in text_lower for w in ["purpose", "designed", "used for", "helps"]):
                score += 3
                
            if score > best_score:
                best_score = score
                best_chunk = text
        
        if best_chunk:
            # For "what is" questions, try to extract a definition
            if "what is" in question_lower:
                # Look for sentences containing keywords
                sentences = re.split(r'[.!?]', best_chunk)
                for sent in sentences:
                    if any(word in sent.lower() for word in question_words):
                        return sent.strip()
            
            # For "purpose" questions
            if "purpose" in question_lower:
                purpose_matches = re.search(r'(used for|designed to|purpose is|helps to|allows you to)([^.!?]+)', best_chunk, re.IGNORECASE)
                if purpose_matches:
                    return purpose_matches.group(0).strip()
            
            # Default to returning a relevant portion (first 200 chars)
            return best_chunk[:200] + "..."
        
        return "No specific answer found in the provided context."