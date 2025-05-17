# agents/document_segment.py

from typing import List, Dict, Optional
from textwrap import wrap
import re
import os
class DocumentSegmenter:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        """
        Initialize the segmenter.

        :param chunk_size: Number of characters per chunk
        :param overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def segment_document(
        self,
        document_text: str,
        image_metadata: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Segment the input document into overlapping text chunks.

        :param document_text: Raw text of the document
        :param image_metadata: List of dicts with image metadata (optional)

        :return: List of dicts, each with:
            - 'text': The chunked text content
            - 'images': Optional image metadata relevant to the chunk
        """
        if not document_text:
            return []

        # Break text into overlapping chunks using sliding window
        chunks = []
        start = 0
        
        # Extract page markers to track which page we're on
        page_markers = {}
        for match in re.finditer(r"--- Page (\d+) ---", document_text):
            page_markers[match.start()] = int(match.group(1))
        
        while start < len(document_text):
            end = min(start + self.chunk_size, len(document_text))
            chunk_text = document_text[start:end]
            
            # Determine current page based on chunk position
            current_page = 1  
            for pos, page in sorted(page_markers.items()):
                if pos <= start:
                    current_page = page
            
            chunk = {'text': chunk_text.strip(), 'start_pos': start, 'end_pos': end, 'page': current_page}

            # Add relevant image metadata if available
            if image_metadata:
                chunk_images = []
                for page_data in image_metadata:
                    if page_data["page"] == current_page:
                        for block in page_data["blocks"]:
                            # Check if the block appears in the chunk text
                            if block["text"] in chunk_text:
                                chunk_images.append({
                                    "description": block["text"],
                                    "bounding_box": block["bounding_box"],
                                    "page": block["page"]
                                })
                chunk['images'] = chunk_images

            chunks.append(chunk)
            start += self.chunk_size - self.overlap

        return chunks