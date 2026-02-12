from paddleocr import PaddleOCR
import json

# Simple test with minimal output
ocr = PaddleOCR(show_log=False)

try:
    result = ocr.predict("Invoice-Page-1-EN.jpg")
    
    print("Result structure:")
    print(f"Type: {type(result)}")
    print(f"Length: {len(result)}")
    
    if result and len(result) > 0:
        print(f"First item keys: {result[0].keys()}")
        
        rec_texts = result[0]["rec_texts"]
        rec_polys = result[0]["rec_polys"]
        
        print(f"Number of texts: {len(rec_texts)}")
        print(f"Number of polys: {len(rec_polys)}")
        
        # Show first few items
        for i in range(min(3, len(rec_texts))):
            print(f"Text {i}: {rec_texts[i]}")
            print(f"Poly {i}: {rec_polys[i]}")
            print()
        
        # Save to file for inspection
        with open("predict_result.json", "w") as f:
            json.dump(result, f, indent=2)
        print("Result saved to predict_result.json")
        
except Exception as e:
    print(f"Error: {e}")

