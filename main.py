import logging

from src.embedder import WatermarkEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Parametry wejściowe
    PDF_INPUT = "data/test.pdf"
    ALFA_AI = 0.5  # To docelowo będzie wyliczane przez Twój model

    # Uruchomienie orkiestratora
    embedder = WatermarkEmbedder(pdf_path=PDF_INPUT, alpha=ALFA_AI)
    embedder.embed_watermark()


if __name__ == "__main__":
    main()