#!/usr/bin/env python3
"""
Test script to examine PaddleOCR v5 result structure
"""

from paddleocr import PaddleOCR
import json
import os

def test_paddleocr():
    print("🔍 Testing PaddleOCR v5 API...")
    
    # Initialize OCR with new v5 parameters
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False, 
        use_textline_orientation=False,
        lang="en",
        show_log=True
    )
    
    # Test image path
    test_image = "../Invoice-Page-1-EN.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        print("Please ensure Invoice-Page-1-EN.jpg is in the parent directory")
        return
    
    print(f"📄 Processing image: {test_image}")
    
    # Run OCR with ocr method
    try:
        result = ocr.ocr(test_image, cls=False)
        
        print("\n📊 Raw Result Structure:")
        print("=" * 50)
        print(f"Type: {type(result)}")
        print(f"Length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        if result:
            print(f"First item type: {type(result[0]) if result else 'N/A'}")
            if result and len(result) > 0:
                print(f"First item: {result[0]}")
                
            print("\n📋 Sample items (first 3):")
            for i, item in enumerate(result[:3]):
                print(f"Item {i}: {item}")
                print(f"  Type: {type(item)}")
                if hasattr(item, '__len__') and len(item) >= 2:
                    print(f"  Box: {item[0] if len(item) > 0 else 'N/A'}")
                    print(f"  Text info: {item[1] if len(item) > 1 else 'N/A'}")
                print()
        
        # Try to parse using expected structure
        print("\n🔧 Parsing to our format:")
        words = []
        
        if result:
            for item in result:
                try:
                    if len(item) >= 2:
                        box, text_info = item[0], item[1]
                        
                        # Extract text and confidence
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text, confidence = text_info[0], text_info[1]
                        else:
                            text, confidence = str(text_info), 1.0
                        
                        # Convert box to minx, miny, maxx, maxy
                        if box and len(box) >= 4:
                            xs = [pt[0] for pt in box]
                            ys = [pt[1] for pt in box]
                            minx, maxx = int(min(xs)), int(max(xs))
                            miny, maxy = int(min(ys)), int(max(ys))
                            
                            word_dict = {
                                "text": str(text),
                                "coords": [minx, miny, maxx, maxy],
                                "confidence": float(confidence)
                            }
                            words.append(word_dict)
                            
                except Exception as e:
                    print(f"  ⚠️ Error parsing item {item}: {e}")
        
        print(f"✅ Parsed {len(words)} words")
        if words:
            print("Sample parsed words:")
            for word in words[:3]:
                print(f"  {word}")
        
        # Save full result for inspection
        with open("paddleocr_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full result saved to: paddleocr_result.json")
        
        return result, words
        
    except Exception as e:
        print(f"❌ Error running OCR: {e}")
        return None, []

if __name__ == "__main__":
    test_paddleocr()
