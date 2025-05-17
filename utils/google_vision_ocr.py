# utils/google_vision_ocr.py

import io
from typing import Tuple, List, Dict
from google.cloud import vision
from google.cloud.vision_v1 import types
from pdf2image import convert_from_path
from PIL import Image
import os

# Initializes Google Cloud Vision client (relies on GOOGLE_APPLICATION_CREDENTIALS env variable)
client = vision.ImageAnnotatorClient()


def extract_text_and_image_metadata(pdf_path: str, dpi: int = 300) -> Tuple[str, List[Dict]]:
    """
    Convert PDF pages to images and extract:
        - full document text
        - image metadata (per page) from Google Cloud Vision API.

    :param pdf_path: Path to the input PDF file.
    :param dpi: Resolution for converting PDF to images.
    :return: Tuple of (full_text: str, image_metadata: List[Dict])
    """
    print(f"[OCR] Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=dpi)
    full_text = ""
    metadata_per_page = []

    for page_num, image in enumerate(images, start=1):
        print(f"[OCR] Processing page {page_num}...")

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        content = img_byte_arr.getvalue()
        image_obj = vision.Image(content=content)

        response = client.document_text_detection(image=image_obj)

        if response.error.message:
            raise RuntimeError(f"Vision API error on page {page_num}: {response.error.message}")

        page_text = response.full_text_annotation.text
        full_text += f"\n--- Page {page_num} ---\n" + page_text

        blocks = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                block_text = ""
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        block_text += word_text + " "
                bounding_box = [(v.x, v.y) for v in block.bounding_box.vertices]
                blocks.append({
                    "text": block_text.strip(),
                    "bounding_box": bounding_box,
                    "page": page_num
                })

        metadata_per_page.append({
            "page": page_num,
            "blocks": blocks
        })

    return full_text.strip(), metadata_per_page
