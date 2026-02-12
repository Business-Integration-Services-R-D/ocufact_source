import io
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from typing import List
import fitz  # PyMuPDF
from ..schemas import OCRWord, PageResult


class OCRService:
    def __init__(self,
                 det_model_dir: str = None,
                 rec_model_dir: str = None,
                 use_angle_cls: bool = False,
                 lang: str = 'en',
                 show_log: bool = False,
                 **kwargs):
        self.use_angle_cls = use_angle_cls
        self.ocr = PaddleOCR(use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

    def image_bytes_to_words(self, image_bytes: bytes) -> List[OCRWord]:
        """OCR an image given as bytes. Return list of OCRWord with coords & score."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(image)

        results = self.ocr.predict(img_np)

        words: List[OCRWord] = []
        if not results or len(results) == 0:
            return words

        # New PaddleOCR predict structure: result[0]["rec_texts"] and result[0]["rec_polys"]
        page_result = results[0]
        rec_texts = page_result.get("rec_texts", [])
        rec_polys = page_result.get("rec_polys", [])
        
        # Ensure both lists have same length
        min_length = min(len(rec_texts), len(rec_polys))
        
        for i in range(min_length):
            text = rec_texts[i]
            poly = rec_polys[i]
            
            # Convert polygon to bounding box coordinates
            if poly is not None and len(poly) >= 4:
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                minx, maxx = int(min(xs)), int(max(xs))
                miny, maxy = int(min(ys)), int(max(ys))
                
                # Use default confidence since predict doesn't return confidence scores
                confidence = 0.95  # Default high confidence
                
                words.append(OCRWord(
                    text=str(text), 
                    coords=[minx, miny, maxx, maxy], 
                    confidence=float(confidence)
                ))

        return words

    def image_to_page_words(self, image_bytes: bytes) -> PageResult:
        words = self.image_bytes_to_words(image_bytes)
        return PageResult(page=1, words=words)

    def pdf_to_pages(self, pdf_bytes: bytes) -> List[PageResult]:
        pages: List[PageResult] = []
        try:
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                words = self.image_bytes_to_words(img_data)
                # Scale coordinates back (used 2x zoom)
                for word in words:
                    word.coords = [coord // 2 for coord in word.coords]

                pages.append(PageResult(page=page_num + 1, words=words))
            pdf_document.close()
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")

        return pages

