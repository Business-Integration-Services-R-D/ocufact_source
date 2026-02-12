import cv2
import numpy as np
from PIL import Image
import io
from typing import List, BinaryIO
import fitz  # PyMuPDF
from ..schemas import MaskField
from ..utils import draw_filled_rectangle, normalize_coords


class MaskService:
    def __init__(self):
        pass
    
    def mask_image(self, image_bytes: bytes, mask_fields: List[MaskField]) -> bytes:
        """Apply masks to image and return masked image bytes."""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data")
        
        height, width = image.shape[:2]
        masked_image = image.copy()
        
        # Apply each mask
        for field in mask_fields:
            # Normalize coordinates
            coords = normalize_coords(field.coords, width, height)
            
            # Apply mask with transparency
            masked_image = draw_filled_rectangle(
                masked_image, 
                coords, 
                field.color, 
                alpha=0.7
            )
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.png', masked_image)
        return buffer.tobytes()
    
    def mask_pdf(self, pdf_bytes: bytes, mask_fields_per_page: List[List[MaskField]]) -> bytes:
        """Apply masks to PDF pages and return masked PDF bytes."""
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Create new PDF for output
            output_pdf = fitz.open()
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # Get mask fields for this page (if any)
                if page_num < len(mask_fields_per_page):
                    mask_fields = mask_fields_per_page[page_num]
                else:
                    mask_fields = []
                
                if mask_fields:
                    # Convert page to image for masking
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Apply masks
                    nparr = np.frombuffer(img_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is not None:
                        height, width = image.shape[:2]
                        masked_image = image.copy()
                        
                        for field in mask_fields:
                            # Scale coordinates for 2x zoom
                            scaled_coords = [coord * 2 for coord in field.coords]
                            coords = normalize_coords(scaled_coords, width, height)
                            
                            masked_image = draw_filled_rectangle(
                                masked_image, 
                                coords, 
                                field.color, 
                                alpha=0.7
                            )
                        
                        # Convert masked image back to PDF page
                        _, buffer = cv2.imencode('.png', masked_image)
                        img_bytes = buffer.tobytes()
                        
                        # Create new page with masked image
                        img_doc = fitz.open(stream=img_bytes, filetype="png")
                        img_page = img_doc[0]
                        
                        # Scale back to original size
                        original_rect = page.rect
                        new_page = output_pdf.new_page(width=original_rect.width, height=original_rect.height)
                        new_page.show_pdf_page(original_rect, img_doc, 0)
                        
                        img_doc.close()
                    else:
                        # If image processing fails, copy original page
                        output_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                else:
                    # No masks for this page, copy original
                    output_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
            
            # Get output PDF bytes
            output_bytes = output_pdf.tobytes()
            
            pdf_document.close()
            output_pdf.close()
            
            return output_bytes
            
        except Exception as e:
            raise ValueError(f"Error processing PDF: {str(e)}")
    
    def create_preview_mask(self, image_bytes: bytes, mask_fields: List[MaskField], preview_alpha: float = 0.3) -> bytes:
        """Create a preview of masks with lower opacity for preview purposes."""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data")
        
        height, width = image.shape[:2]
        preview_image = image.copy()
        
        # Apply each mask with preview alpha
        for field in mask_fields:
            coords = normalize_coords(field.coords, width, height)
            preview_image = draw_filled_rectangle(
                preview_image, 
                coords, 
                field.color, 
                alpha=preview_alpha
            )
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.png', preview_image)
        return buffer.tobytes()

