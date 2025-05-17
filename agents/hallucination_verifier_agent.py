# agents/hallucination_verifier_agent.py
import os
import time
import re
import statistics
from typing import List, Dict, Tuple
import requests

class HallucinationVerifierAgent:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", threshold: float = 0.7):
        """
        Initializes the hallucination verifier agent.
        
        :param model_name: The model to use for verification
        :param threshold: Threshold above which to consider hallucination risk high
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            print("[WARNING] GROQ_API_KEY not set - using rule-based verification instead")
        
        self.model = model_name
        self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.threshold = threshold
        self.cache = {}
        
        # List of words that indicate factual content
        self.factual_indicators = [
            'figure', 'table', 'diagram', 'step', 'instructions', 
            'shown', 'listed', 'described', 'depicted', 'illustrated',
            'reference', 'manual', 'guide', 'document', 'specification',
            'measurement', 'dimension', 'size', 'weight', 'quantity'
        ]

    def score_hallucination(self, query: str, chunks: List[Dict]) -> List[float]:
        """
        Analyzes chunks for hallucination risk based on query and context.
        
        :param query: User query
        :param chunks: List of content chunks
        :return: List of hallucination risk scores (0-1)
        """
        # If no chunks, return high risk
        if not chunks:
            return [1.0]
            
        # First try rule-based evaluation to avoid API calls
        rule_based_scores = self._rule_based_evaluation(query, chunks)
        
        # If scores are all low (safe), return them directly
        if max(rule_based_scores) < 0.5:
            print("[INFO] Using rule-based hallucination scores (all low risk)")
            return rule_based_scores
        
        # If we have API access and there are potential high-risk chunks,
        # evaluate those chunks with the model
        if self.api_key:
            print("[INFO] Evaluating potential hallucination risks with model")
            scores = []
            for i, chunk in enumerate(chunks):
                # Only use model for higher risk chunks
                if rule_based_scores[i] > 0.3:
                    score = self._evaluate_chunk_with_model(query, chunk)
                else:
                    score = rule_based_scores[i]
                scores.append(score)
            return scores
        else:
            # No API, return rule-based scores
            return rule_based_scores

    def _rule_based_evaluation(self, query: str, chunks: List[Dict]) -> List[float]:
        """
        Evaluate hallucination risk using rule-based heuristics.
        
        :return: List of scores between 0 and 1
        """
        scores = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            
            # Initialize base score
            score = 0.5 
            
            # Check for factual indicators that reduce hallucination risk
            factual_matches = sum(1 for word in self.factual_indicators if word in text.lower())
            score -= factual_matches * 0.05  # Reduce score for each factual indicator
            
            #  Check if query terms are in the chunk
            query_terms = set(re.findall(r'\b\w{4,}\b', query.lower()))
            text_terms = set(re.findall(r'\b\w{4,}\b', text.lower()))
            query_term_matches = len(query_terms.intersection(text_terms))
            
            # More query term matches = lower hallucination risk
            if query_terms:
                match_ratio = query_term_matches / len(query_terms)
                score -= match_ratio * 0.2
            
            # Length and detail checks
            if len(text) < 100:
                score += 0.1  # Short responses have higher hallucination risk
            if len(text) > 500:
                score -= 0.1  # Detailed responses have lower hallucination risk
                
            #  Numbers, dates, and specific details suggest factual content
            if re.search(r'\b\d+(\.\d+)?(cm|mm|m|kg|g|lb|inch|ft|°C|°F)\b', text):
                score -= 0.15  # Measurements suggest factual content
                
            if re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text) or re.search(r'\b\d{1,2}-\d{1,2}-\d{2,4}\b', text):
                score -= 0.1  # Dates suggest factual content
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(score, 1.0))
            scores.append(score)
            
        return scores
        
    def _evaluate_chunk_with_model(self, query: str, chunk: Dict) -> float:
        """
        Use the Groq model to evaluate hallucination risk.
        
        :param query: User query
        :param chunk: Content chunk
        :return: Hallucination risk score between 0 and 1
        """
        text = chunk.get("text", "")
        
        # If empty text, return high risk
        if not text:
            return 1.0
            
        # Check cache
        cache_key = self._generate_cache_key(query, text)
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Create prompt for assessing factuality
        prompt = self._build_prompt(query, text)
        
        try:
            # Call model
            score = self._query_groq_model(prompt)
            
            # Store in cache
            self.cache[cache_key] = score
            return score
            
        except Exception as e:
            print(f"[WARNING] Error querying model: {e}")
            # Fall back to rule-based score
            rule_score = self._rule_based_evaluation(query, [chunk])[0]
            return rule_score
    
    def _generate_cache_key(self, query: str, text: str) -> str:
        """Generate cache key for query+text combination"""
        import hashlib
        combined = (query + text).encode('utf-8')
        return hashlib.md5(combined).hexdigest()
            
    def _build_prompt(self, query: str, chunk_text: str) -> str:
        """
        Build a more balanced prompt that doesn't bias toward hallucination.
        """
        return (
            "You are evaluating if a text excerpt contains sufficient information to answer a question.\n\n"
            f"Question: {query}\n\n"
            f"Text excerpt: {chunk_text}\n\n"
            "On a scale from 0 to 1, rate the risk of hallucination if this excerpt were used to answer the question.\n"
            "- 0 means the excerpt clearly contains all needed information\n"
            "- 0.5 means some information is present but some may need to be inferred\n"
            "- 1 means the excerpt is unrelated or missing critical information\n\n"
            "Respond with ONLY a number between 0 and 1."
        )

    def _query_groq_model(self, prompt: str) -> float:
        """
        Query Groq model with improved error handling and retries.
        
        :param prompt: The prompt to send
        :return: Float score from 0.0 to 1.0
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,  # Lower temperature for more consistent results
            "max_tokens": 10
        }
        
        # Try up to 3 times with backoff
        for attempt in range(3):
            try:
                response = requests.post(self.endpoint, headers=headers, json=payload, timeout=10)
                if response.status_code != 200:
                    raise RuntimeError(f"Groq API Error {response.status_code}: {response.text}")
                
                content = response.json()["choices"][0]["message"]["content"].strip()
                
                # Extract the first number found
                match = re.search(r'([0-9]*\.?[0-9]+)', content)
                if not match:
                    raise ValueError(f"No number found in response: '{content}'")
                    
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
                
            except Exception as e:
                print(f"[WARNING] API error (attempt {attempt+1}/3): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
        # If all attempts failed, return a moderate score
        return 0.5