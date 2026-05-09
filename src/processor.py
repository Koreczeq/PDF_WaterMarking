import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WatermarkProcessor:
    """A class transforms images to DCT domain and applies watermarking."""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def process_image(self, image_path: str) -> np.ndarray:
        """Process the image and return the watermarked image."""

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.error(f"Could not read image: {image_path}")
            return None

        h, w = img.shape
        img = cv2.resize(img, (w // 8 * 8, h // 8 * 8))  # Ensure dimensions are multiples of 8

        watermarked_img = np.zeros_like(img, dtype=np.float32)
        for i in range(0, img.shape[0], 8):
            for j in range(0, img.shape[1], 8):
                block = img[i:i+8, j:j+8]
                dct_block = cv2.dct(block.astype(np.float32))
                dct_block[4, 4] += self.alpha * np.random.rand()  # Simple watermarking
                watermarked_block = cv2.idct(dct_block)
                watermarked_img[i:i+8, j:j+8] = watermarked_block  

        return np.clip(watermarked_img, 0, 255).astype(np.uint8)
