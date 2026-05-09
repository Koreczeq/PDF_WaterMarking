import os
import logging
import cv2
import fitz  # PyMuPDF

from src.PDFImageExtractor import PDFImageExtractor
from src.processor import WatermarkProcessor
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WatermarkEmbedder:
    """A class to embed watermarks into images extracted from PDFs."""
    def __init__(self, pdf_path: str, alpha: float = 0.8):
        self.pdf_path = pdf_path
        self.alpha = alpha
        self.temp_dir = 'data/temp'
        self.output_dir = 'data/processed'
        
        self.extractor = PDFImageExtractor(pdf_path, output_dir=self.temp_dir)
        self.processor = WatermarkProcessor(alpha=alpha)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def embed_watermark(self):
        """Run the full process of extracting images and embedding watermarks."""
        logging.info(f"Starting watermark embedding for PDF: {self.pdf_path}")

        # Step 1: Extract images from PDF
        self.extractor.extract()    

        doc = fitz.open(self.pdf_path)
        
        images_found = False

        for page in doc:
            img_list = page.get_images(full=True)
            for img_info in img_list:
                images_found = True
                xref = img_info[0]
                
                # Extract the raw image so it can be processed
                base_image = doc.extract_image(xref)
                image_ext = base_image["ext"]
                
                # Save it temporarily so OpenCV can load it
                temp_img_path = os.path.join(self.temp_dir, f"temp_ref.{image_ext}")
                with open(temp_img_path, "wb") as f:
                    f.write(base_image["image"])

                # 2. DCT processing
                processed_img = self.processor.process_image(temp_img_path)

                if processed_img is not None:
                    # Encode the image back into binary format (e.g. PNG)
                    _, img_encoded = cv2.imencode(f".{image_ext}", processed_img)
                    img_bytes = img_encoded.tobytes()

                    # 3. Reintegration: replace the PDF image using xref
                    page.replace_image(xref, stream=img_bytes)
                    logging.info(f"Updated image (xref: {xref}) on page {page.number + 1}")

        if images_found:
            output_path = os.path.join(self.output_dir, f"watermarked_{os.path.basename(self.pdf_path)}")
            doc.save(output_path)
            doc.close()
            logging.info(f"Process completed. File saved to: {output_path}")
        else:
            logging.warning("Images not found in the PDF document.")

