from agents.document_segmenter_agent import DocumentSegmenter
from agents.retriever_agent import RetrieverAgent
from agents.image_relevance_agent import ImageRelevanceAgent
from agents.hallucination_verifier_agent import HallucinationVerifierAgent
from agents.response_generator_agent import ResponseGeneratorAgent
from utils.google_vision_ocr import extract_text_and_image_metadata
import sys
from dotenv import load_dotenv
load_dotenv()

def run_pipeline(pdf_path: str, user_query: str):
    print("[INFO] Starting RAG pipeline...")
    
    # Step 1: Extract document text and images using Google Cloud Vision
    print("[INFO] Extracting text and images from PDF...")
    doc_text, image_metadata = extract_text_and_image_metadata(pdf_path)

    # Step 2: Segment the document into chunks
    print("[INFO] Segmenting document into chunks...")
    segmenter = DocumentSegmenter(chunk_size=800, overlap=100)
    chunks = segmenter.segment_document(doc_text, image_metadata)

    # Step 3: Classify and route image-containing chunks
    print("[INFO] Classifying image relevance...")
    image_router = ImageRelevanceAgent()
    chunks = image_router.classify_images(chunks)

    # Step 4: Retrieve relevant context for the query
    print("[INFO] Retrieving relevant context...")
    retriever = RetrieverAgent(corpus=chunks)
    retrieved_chunks = retriever.retrieve(query=user_query)

    # Step 5: Detect hallucination risk
    print("[INFO] Checking for hallucination risks...")
    verifier = HallucinationVerifierAgent(threshold=0.85)
    hallucination_scores = verifier.score_hallucination(user_query, retrieved_chunks)

    # Step 6: Generate response using LLM
    print("[INFO] Generating response...")
    response_agent = ResponseGeneratorAgent(hallucination_threshold=0.85)
    result = response_agent.generate_response(
        question=user_query,
        context_chunks=retrieved_chunks,
        hallucination_scores=hallucination_scores
    )

    print("\n==== FINAL RESPONSE ====")
    print("Response:", result['response_text'])
    if result['abstain']:
        print("ABSTAINED")
        print("Reason:", result['reason'])
    print("========================")


if __name__ == "__main__":
    # Hardcode your input PDF path and query here:
    input_file = r"datasets\21583473018.pdf"
    user_query = "What is the purpose of this document?"

    run_pipeline(input_file, user_query)