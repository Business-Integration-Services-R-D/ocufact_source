import os
import torch
from transformers import RobertaTokenizer, RobertaForTokenClassification
from typing import List, Tuple
from ..utils import batch_list


class ModelService:
    def __init__(self, 
                 binary_model_path: str = None,
                 label_model_path: str = None,
                 batch_size: int = 32):
        # Set default paths based on environment
        if binary_model_path is None:
            binary_model_path = os.getenv("BINARY_MODEL_PATH", "/app/binary_classification_model")
        if label_model_path is None:
            label_model_path = os.getenv("LABEL_MODEL_PATH", "/app/label_detection_model")
            
        # Fallback for local development
        if not os.path.exists(binary_model_path):
            binary_model_path = "../binary_classification_model"
        if not os.path.exists(label_model_path):
            label_model_path = "../label_detection_model"
            
        # Check if models exist
        if not os.path.exists(binary_model_path):
            raise FileNotFoundError(f"Binary classification model not found at: {binary_model_path}")
        if not os.path.exists(label_model_path):
            raise FileNotFoundError(f"Label detection model not found at: {label_model_path}")
            
        print(f"Loading binary model from: {binary_model_path}")
        print(f"Loading label model from: {label_model_path}")
        
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load binary classification model (token classification model)
        print("Loading binary classification tokenizer...")
        self.binary_tokenizer = RobertaTokenizer.from_pretrained(binary_model_path)
        print("Loading binary classification model (token classification)...")
        self.binary_model = RobertaForTokenClassification.from_pretrained(binary_model_path)
        print("Moving binary model to device...")
        self.binary_model.to(self.device)
        self.binary_model.eval()
        print("Binary classification model loaded successfully")
        
        # Load label detection model (also token classification)
        print("Loading label detection tokenizer...")
        self.label_tokenizer = RobertaTokenizer.from_pretrained(label_model_path)
        print("Loading label detection model (token classification)...")
        self.label_model = RobertaForTokenClassification.from_pretrained(label_model_path)
        print("Moving label model to device...")
        self.label_model.to(self.device)
        self.label_model.eval()
        print("Label detection model loaded successfully")
        
        # Load actual label mappings from the model config
        self.binary_id2label = self.binary_model.config.id2label
        self.label_id2label = self.label_model.config.id2label
        
        print(f"Binary model labels: {len(self.binary_id2label)} classes")
        print(f"Label model labels: {len(self.label_id2label)} classes")
    
    def predict_binary(self, texts: List[str]) -> List[Tuple[bool, float]]:
        """Predict if texts are sensitive (using token classification model)."""
        if not texts:
            return []
        
        results = []
        
        for batch in batch_list(texts, self.batch_size):
            # Tokenize batch
            inputs = self.binary_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.binary_model(**inputs)
                # For token classification, we get logits for each token
                # Shape: (batch_size, sequence_length, num_labels)
                logits = outputs.logits
                
                for i, text_logits in enumerate(logits):
                    # Get predictions for each token
                    predictions = torch.nn.functional.softmax(text_logits, dim=-1)
                    
                    # Check if any token is predicted as sensitive (label 1 or 2)
                    # Label 0: O (non-sensitive), Label 1: B-sensitive, Label 2: I-sensitive
                    sensitive_tokens = predictions[:, 1:].sum(dim=-1)  # Sum B- and I- probabilities
                    
                    # Consider text sensitive if any token has >0.5 probability of being sensitive
                    max_sensitive_score = sensitive_tokens.max().item()
                    is_sensitive = max_sensitive_score > 0.5
                    
                    results.append((is_sensitive, max_sensitive_score))
        
        return results
    
    def predict_labels(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict labels for texts (using token classification model)."""
        if not texts:
            return []
        
        results = []
        
        for batch in batch_list(texts, self.batch_size):
            # Tokenize batch
            inputs = self.label_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.label_model(**inputs)
                # For token classification, we get logits for each token
                logits = outputs.logits
                
                for i, text_logits in enumerate(logits):
                    # Get predictions for each token
                    predictions = torch.nn.functional.softmax(text_logits, dim=-1)
                    
                    # Find the most confident non-O prediction across all tokens
                    best_label = "O"
                    best_score = 0.0
                    
                    for token_preds in predictions:
                        # Skip O class (index 0)
                        non_o_preds = token_preds[1:]
                        if len(non_o_preds) > 0:
                            max_idx = torch.argmax(non_o_preds).item() + 1  # +1 because we skipped index 0
                            max_score = non_o_preds[max_idx - 1].item()
                            
                            if max_score > best_score:
                                best_score = max_score
                                label_name = self.label_id2label.get(max_idx, "UNKNOWN")
                                # Convert BIO tags to simple labels (B-NAME -> NAME)
                                if label_name.startswith(('B-', 'I-')):
                                    best_label = label_name[2:]
                                else:
                                    best_label = label_name
                    
                    results.append((best_label, best_score))
        
        return results
