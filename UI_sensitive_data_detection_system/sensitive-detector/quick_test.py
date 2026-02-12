from paddleocr import PaddleOCR
import json

# Simple test
ocr = PaddleOCR()

# Test with a simple image path
result = ocr.predict("Invoice-Page-1-EN.jpg")

print(result[0]["rec_texts"])
print(result[0]["rec_polys"])

print(len(result[0]["rec_texts"]))
print(len(result[0]["rec_polys"]))




# # Save result
# with open("ocr_result.json", "w") as f:
#     json.dump(result, f, indent=2)
# print("Result saved to ocr_result.json")
