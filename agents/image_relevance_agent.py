import os
import re
from typing import List, Dict, Optional
import hashlib
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm

class ImageRelevanceAgent:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize ImageRelevanceAgent with sentence-transformers model.
        
        :param model_name: Name of the sentence-transformers model to use
        """
        print(f"[INFO] Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache = {}
        
        # Keywords that generally indicate relevance when found in descriptions
        self.relevance_indicators = [
            'figure', 'diagram', 'illustration', 'chart', 'graph', 'table', 
            'image', 'picture', 'photo', 'screenshot', 'icon', 'logo',
            'step', 'instruction', 'assembly', 'part', 'component', 'product',
            'measurement', 'dimension', 'size', 'quantity', 'number',
            'example', 'sample', 'reference', 'comparison'
        ]
        
        # Common reference patterns (figure X, diagram Y, etc.)
        self.reference_pattern = re.compile(
            r'\b(fig(\.|ure)?|diagram|chart|table|ill(\.|ustration)?)\s*(\d+|[A-Z])\b', 
            re.IGNORECASE
        )

    def classify_images(self, chunks: List[Dict]) -> List[Dict]:
        """
        Classify image relevance in document chunks using sentence-transformers.
        
        :param chunks: List of document chunks containing optional 'images' fields
        :return: Modified chunks with classified image actions
        """
        print("[INFO] Classifying images with sentence-transformers...")
        
        # Gather all tasks
        tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            images = chunk.get("images", [])
            
            for img_idx, image in enumerate(images):
                # Skip if already classified
                if "action" in image and image["action"] in ["augment_metadata", "irrelevant"]:
                    continue
                
                description = image.get("description", "")
                image_path = image.get("path", "")
                
                if description or image_path:
                    tasks.append((chunk_idx, img_idx, text, description, image_path))
        
        # Process all tasks with progress bar
        for task in tqdm(tasks, desc="Classifying images"):
            chunk_idx, img_idx, text, description, image_path = task
            
            # Check cache first
            cache_key = self._generate_cache_key(text, description, image_path)
            if cache_key in self.cache:
                chunks[chunk_idx]["images"][img_idx]["action"] = self.cache[cache_key]
                continue
            
            # Determine relevance
            try:
                # Default to rule-based if no description
                if not description:
                    is_relevant = self._rule_based_classification(text, description, image_path)
                else:
                    # Use semantic similarity with sentence-transformers
                    is_relevant = self._semantic_similarity_classification(text, description)
                    
                    # If the result is borderline, augment with rule-based approach
                    if 0.3 <= is_relevant <= 0.7:
                        rule_result = self._rule_based_classification(text, description, image_path)
                        # Boost or reduce based on rules
                        is_relevant = is_relevant + 0.2 if rule_result else is_relevant - 0.1
                
                # Convert to binary decision with threshold
                action = "augment_metadata" if is_relevant > 0.5 else "irrelevant"
                chunks[chunk_idx]["images"][img_idx]["action"] = action
                self.cache[cache_key] = action
                
            except Exception as e:
                print(f"[WARNING] Error classifying image: {str(e)}")
                # Default to irrelevant in case of error
                chunks[chunk_idx]["images"][img_idx]["action"] = "irrelevant"
        
        return chunks
    
    def _generate_cache_key(self, text: str, description: str, image_path: str = "") -> str:
        """Generate a cache key from content"""
        components = []
        if text:
            components.append(text[:100])
        if description:
            components.append(description[:100])
        if image_path:
            components.append(os.path.basename(image_path))
        
        combined = "".join(components).encode('utf-8')
        return hashlib.md5(combined).hexdigest()
    
    def _semantic_similarity_classification(self, text: str, description: str) -> float:
        """
        Use sentence-transformers to compute semantic similarity between text and description.
        
        :param text: Document context
        :param description: Image description
        :return: Similarity score between 0 and 1
        """
        if not text or not description:
            return 0.0
            
        # Truncate if too long
        if len(text) > 512:
            # Take first and last parts
            text = text[:256] + " ... " + text[-256:]
        
        # Encode sentences
        text_embedding = self.model.encode(text, convert_to_tensor=True)
        desc_embedding = self.model.encode(description, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = util.pytorch_cos_sim(text_embedding, desc_embedding).item()
        
        return float(similarity)
    
    def _rule_based_classification(self, text: str, description: str, filename: str = "") -> bool:
        """
        Apply rule-based heuristics to determine image relevance.
        
        :param text: Document context
        :param description: Image description
        :param filename: Image filename
        :return: Boolean indicating relevance
        """
        # RULE 1: Check for references
        text_references = set(self.reference_pattern.findall(text))
        desc_references = set(self.reference_pattern.findall(description))
        if text_references and desc_references and text_references.intersection(desc_references):
            return True
        
        # RULE 2: Check for relevance indicators
        for indicator in self.relevance_indicators:
            if indicator in description.lower() and indicator in text.lower():
                return True
        
        # RULE 3: Check for descriptive filename
        if filename:
            filename_lower = filename.lower()
            for indicator in self.relevance_indicators:
                if indicator in filename_lower:
                    return True
                    
        # RULE 4: Check if entire description appears in text
        if description and len(description) > 15 and description.lower() in text.lower():
            return True
            
        # Default to False if no rules matched
        return False