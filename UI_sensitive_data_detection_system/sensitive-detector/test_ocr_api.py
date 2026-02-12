#!/usr/bin/env python3
"""
Test script to trigger OCR API and see the error
"""

import requests
import os

def test_ocr_api():
    # Test image path
    test_image = "Invoice-Page-1-EN.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"📄 Testing OCR API with: {test_image}")
    
    # API endpoint
    url = "http://localhost:8000/api/v1/ocr/image"
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (test_image, f, 'image/jpeg')}
            
            print("🚀 Sending request...")
            response = requests.post(url, files=files, timeout=60)
            
            print(f"📊 Response status: {response.status_code}")
            print(f"📄 Response content: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success! Found {len(result.get('pages', []))} pages")
                if result.get('pages'):
                    words = result['pages'][0].get('words', [])
                    print(f"   First page has {len(words)} words")
                    if words:
                        print(f"   First word: {words[0]}")
            else:
                print(f"❌ Error: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_ocr_api()
