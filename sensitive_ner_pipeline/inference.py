import os
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import fitz  # PyMuPDF for PDF handling

MODEL_PATH = "models\dbmdz/bert-base-turkish-cased"

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# OCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

def extract_text_from_image(image_path):
    """Extract text from image using OCR"""
    ocr_result = ocr.ocr(image_path, cls=True)
    texts = []
    for page in ocr_result:
        for line in page:
            texts.append(line[1][0])
    return " ".join(texts)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF page by page"""
    doc = fitz.open(pdf_path)
    pages_text = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        # Try to extract text directly first
        text = page.get_text()
        
        # If no text found, use OCR
        if not text.strip():
            # Convert page to image and use OCR
            pix = page.get_pixmap()
            img_path = f"temp_page_{page_num}.png"
            pix.save(img_path)
            text = extract_text_from_image(img_path)
            os.remove(img_path)
        
        pages_text.append(text.strip())
    
    doc.close()
    return pages_text

def run_inference(file_path):
    """Run inference on PDF or image file"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        print(f"📄 Processing PDF: {file_path}")
        pages_text = extract_text_from_pdf(file_path)
        
        for page_num, text in enumerate(pages_text, 1):
            if text:
                print(f"\n📃 Page {page_num}:")
                print("-" * 50)
                
                # Run NER on this page
                results = ner_pipeline(text)
                
                if results:
                    print("🔍 Sensitive data found:")
                    for r in results:
                        print(f"  • {r['entity_group']}: '{r['word']}' (confidence: {r['score']:.2f})")
                else:
                    print("✅ No sensitive data detected")
                    
    elif file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        print(f"🖼️ Processing image: {file_path}")
        text = extract_text_from_image(file_path)
        
        if text:
            print(f"\n📃 Extracted text:")
            print("-" * 50)
            
            # Run NER
            results = ner_pipeline(text)
            
            if results:
                print("🔍 Sensitive data found:")
                for r in results:
                    print(f"  • {r['entity_group']}: '{r['word']}' (confidence: {r['score']:.2f})")
            else:
                print("✅ No sensitive data detected")
        else:
            print("⚠️ No text extracted from image")
    else:
        print(f"❌ Unsupported file format: {file_ext}")

# Example usage
if __name__ == "__main__":
    test_file = "/content/sample_invoice.png"  # change to your image/pdf
    run_inference(test_file)
