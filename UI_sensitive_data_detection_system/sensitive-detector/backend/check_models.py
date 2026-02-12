#!/usr/bin/env python3
"""
Simple script to check if models can be loaded correctly
"""

import os
import sys

def check_model_path(path, model_name):
    """Check if model path exists and contains required files"""
    if not os.path.exists(path):
        print(f"❌ {model_name} path does not exist: {path}")
        return False
    
    required_files = ['config.json', 'tokenizer.json']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ {model_name} missing files: {missing_files}")
        return False
    
    print(f"✅ {model_name} path is valid: {path}")
    return True

def check_regex_synthesizer():
    """Check if regex synthesizer can be imported"""
    try:
        # Add paths similar to the service
        sys.path.insert(0, "/app")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
        
        from smart_regex_synthesizer2 import SmartRegexSynthesizer, validate
        print("✅ Smart regex synthesizer imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Could not import smart_regex_synthesizer2: {e}")
        return False

def main():
    print("🔍 Checking model availability...")
    print("-" * 50)
    
    # Check binary classification model
    binary_path = os.getenv("BINARY_MODEL_PATH", "/app/binary_classification_model")
    if not os.path.exists(binary_path):
        binary_path = "../binary_classification_model"
    
    binary_ok = check_model_path(binary_path, "Binary Classification Model")
    
    # Check label detection model  
    label_path = os.getenv("LABEL_MODEL_PATH", "/app/label_detection_model")
    if not os.path.exists(label_path):
        label_path = "../label_detection_model"
        
    label_ok = check_model_path(label_path, "Label Detection Model")
    
    # Check regex synthesizer
    regex_ok = check_regex_synthesizer()
    
    print("-" * 50)
    
    if binary_ok and label_ok and regex_ok:
        print("✅ All components are ready!")
        return 0
    else:
        print("❌ Some components are missing. Please check the setup.")
        return 1

if __name__ == "__main__":
    exit(main())

