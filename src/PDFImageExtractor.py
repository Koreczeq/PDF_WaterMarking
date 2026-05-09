import fitz  # PyMuPDF
import io
import os
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PDFImageExtractor:
    """A class to extract images from a PDF file."""
    def __init__(self, pdf_path: str, output_dir: str = 'data/extracted_images'):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def extract(self) -> int:
        """Extract images from the PDF and return the count."""
        try:
            doc = fitz.open(self.pdf_path)
            img_count = 0

            for page_index in range(len(doc)):
                for img_index, img in enumerate(doc[page_index].get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    image_bytes = base_image["image"]
                    image_name = f"p{page_index+1}_i{img_index}.{base_image['ext']}"
                    
                    with open(os.path.join(self.output_dir, image_name), "wb") as f:
                        f.write(image_bytes)
                    img_count += 1
            
            logging.info(f"Successfully extracted {img_count} images.")
            return img_count
        except Exception as e:
            logging.error(f"Error occurred while extracting images: {e}")
            return 0