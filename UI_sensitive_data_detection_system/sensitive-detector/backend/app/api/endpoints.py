import asyncio
import re
import traceback
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Form
from fastapi.responses import Response
from typing import List, Dict, Any
import json

logger = logging.getLogger(__name__)

from ..schemas import (
    OCRResponse, DetectionResponse, MaskRequest, RegexGenerateRequest, 
    RegexGenerateResponse, RegexMatchRequest, RegexMatchResponse,
    CoordinateMatchRequest, ErrorResponse, DetectionWord, DetectionPageResult,
    RegexMatchWord, RegexMatchPageResult, MaskField, SensitiveResponse,
    SensitiveWord, SensitivePageResult
)
from ..services.ocr_service import OCRService
from ..services.model_service import ModelService
from ..services.regex_service import RegexService
from ..services.mask_service import MaskService
from ..services.matching_service import MatchingService

router = APIRouter()

# Global service instances - loaded at startup
ocr_service = None
model_service = None
regex_service = None
mask_service = None
matching_service = None


def initialize_services():
    """Initialize all services at startup"""
    global ocr_service, model_service, regex_service, mask_service, matching_service
    
    logger.info("🚀 Initializing services at startup...")
    
    # Initialize OCR service
    try:
        logger.info("📄 Loading OCR service...")
        ocr_service = OCRService()
        logger.info("✅ OCR service loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load OCR service: {e}")
        raise
    
    # Initialize model service
    try:
        logger.info("🤖 Loading ML models...")
        model_service = ModelService()
        logger.info("✅ Model service loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model service: {e}")
        # Create a dummy model service that returns placeholder results
        class DummyModelService:
            def predict_binary(self, texts):
                return [(False, 0.5) for _ in texts]
            def predict_labels(self, texts):
                return [("O", 0.5) for _ in texts]
        model_service = DummyModelService()
        logger.warning("⚠️ Using dummy model service due to model loading failure")
    
    # Initialize other services
    try:
        logger.info("🔧 Loading other services...")
        regex_service = RegexService()
        mask_service = MaskService()
        matching_service = MatchingService()
        logger.info("✅ All services loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load other services: {e}")
        raise


def get_ocr_service() -> OCRService:
    if ocr_service is None:
        raise RuntimeError("OCR service not initialized. Call initialize_services() first.")
    return ocr_service


def get_model_service() -> ModelService:
    if model_service is None:
        raise RuntimeError("Model service not initialized. Call initialize_services() first.")
    return model_service


def get_regex_service() -> RegexService:
    if regex_service is None:
        raise RuntimeError("Regex service not initialized. Call initialize_services() first.")
    return regex_service


def get_mask_service() -> MaskService:
    if mask_service is None:
        raise RuntimeError("Mask service not initialized. Call initialize_services() first.")
    return mask_service


def get_matching_service() -> MatchingService:
    if matching_service is None:
        raise RuntimeError("Matching service not initialized. Call initialize_services() first.")
    return matching_service


@router.post("/ocr/image", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    ocr: OCRService = Depends(get_ocr_service)
):
    """Extract text from image using OCR."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"OCR image request - filename: {file.filename}, content_type: {file.content_type}")
        
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        logger.info(f"File read successfully - size: {len(contents)} bytes")
        
        # Run OCR in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        logger.info("Starting OCR processing...")
        page_result = await loop.run_in_executor(None, ocr.image_to_page_words, contents)
        logger.info(f"OCR completed - found {len(page_result.words)} words")
        
        return OCRResponse(pages=[page_result])
        
    except Exception as e:
        logger.error(f"OCR error: {e}")
        logger.error(f"OCR traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(
    file: UploadFile = File(...),
    ocr: OCRService = Depends(get_ocr_service)
):
    """Extract text from PDF using OCR."""
    try:
        if not file.content_type or file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        contents = await file.read()
        
        # Run OCR in thread pool
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(None, ocr.pdf_to_pages, contents)
        
        return OCRResponse(pages=pages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/binary/image", response_model=SensitiveResponse)
async def detect_binary_image(
    file: UploadFile = File(...),
    ocr: OCRService = Depends(get_ocr_service),
    model: ModelService = Depends(get_model_service)
):
    """Detect sensitive data in image using binary classification."""
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        
        # Run OCR
        loop = asyncio.get_event_loop()
        page_result = await loop.run_in_executor(None, ocr.image_to_page_words, contents)
        
        # Extract texts for classification
        texts = [word.text for word in page_result.words]
        
        # Run binary classification
        predictions = await loop.run_in_executor(None, model.predict_binary, texts)
        
        # Combine results - only include sensitive data
        sensitive_words = []
        for word, (is_sensitive, score) in zip(page_result.words, predictions):
            if is_sensitive:  # Only include sensitive data
                sensitive_words.append(SensitiveWord(
                    text=word.text,
                    label="UNKNOWN"  # Binary classification doesn't provide specific labels
                ))
        
        sensitive_page = SensitivePageResult(page=1, words=sensitive_words)
        return SensitiveResponse(pages=[sensitive_page])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/binary/pdf", response_model=SensitiveResponse)
async def detect_binary_pdf(
    file: UploadFile = File(...),
    ocr: OCRService = Depends(get_ocr_service),
    model: ModelService = Depends(get_model_service)
):
    """Detect sensitive data in PDF using binary classification."""
    try:
        if not file.content_type or file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        contents = await file.read()
        
        # Run OCR
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(None, ocr.pdf_to_pages, contents)
        
        sensitive_pages = []
        
        for page in pages:
            # Extract texts for classification
            texts = [word.text for word in page.words]
            
            if texts:
                # Run binary classification
                predictions = await loop.run_in_executor(None, model.predict_binary, texts)
                
                # Combine results - only include sensitive data
                sensitive_words = []
                for word, (is_sensitive, score) in zip(page.words, predictions):
                    if is_sensitive:  # Only include sensitive data
                        sensitive_words.append(SensitiveWord(
                            text=word.text,
                            label="UNKNOWN"  # Binary classification doesn't provide specific labels
                        ))
            else:
                sensitive_words = []
            
            sensitive_pages.append(SensitivePageResult(page=page.page, words=sensitive_words))
        
        return SensitiveResponse(pages=sensitive_pages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/labels/image", response_model=SensitiveResponse)
async def detect_labels_image(
    file: UploadFile = File(...),
    ocr: OCRService = Depends(get_ocr_service),
    model: ModelService = Depends(get_model_service)
):
    """Detect labeled sensitive data in image."""
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        
        # Run OCR
        loop = asyncio.get_event_loop()
        page_result = await loop.run_in_executor(None, ocr.image_to_page_words, contents)
        
        # Extract texts for classification
        texts = [word.text for word in page_result.words]
        
        # Run label classification
        predictions = await loop.run_in_executor(None, model.predict_labels, texts)
        
        # Combine results - only include sensitive data
        sensitive_words = []
        for word, (label, score) in zip(page_result.words, predictions):
            if label != "O":  # Only include sensitive data (not "O" which means non-sensitive)
                sensitive_words.append(SensitiveWord(
                    text=word.text,
                    label=label
                ))
        
        sensitive_page = SensitivePageResult(page=1, words=sensitive_words)
        return SensitiveResponse(pages=[sensitive_page])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/labels/pdf", response_model=SensitiveResponse)
async def detect_labels_pdf(
    file: UploadFile = File(...),
    ocr: OCRService = Depends(get_ocr_service),
    model: ModelService = Depends(get_model_service)
):
    """Detect labeled sensitive data in PDF."""
    try:
        if not file.content_type or file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        contents = await file.read()
        
        # Run OCR
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(None, ocr.pdf_to_pages, contents)
        
        sensitive_pages = []
        
        for page in pages:
            # Extract texts for classification
            texts = [word.text for word in page.words]
            
            if texts:
                # Run label classification
                predictions = await loop.run_in_executor(None, model.predict_labels, texts)
                
                # Combine results - only include sensitive data
                sensitive_words = []
                for word, (label, score) in zip(page.words, predictions):
                    if label != "O":  # Only include sensitive data (not "O" which means non-sensitive)
                        sensitive_words.append(SensitiveWord(
                            text=word.text,
                            label=label
                        ))
            else:
                sensitive_words = []
            
            sensitive_pages.append(SensitivePageResult(page=page.page, words=sensitive_words))
        
        return SensitiveResponse(pages=sensitive_pages)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match/coords/image", response_model=OCRResponse)
async def match_coords_image(
    file: UploadFile = File(...),
    coords: str = None,
    ocr: OCRService = Depends(get_ocr_service),
    matching: MatchingService = Depends(get_matching_service)
):
    """Match OCR words with given coordinates in image."""
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not coords:
            raise HTTPException(status_code=400, detail="Coordinates are required")
        
        # Parse coordinates from JSON string
        try:
            coords_list = json.loads(coords)
            if not isinstance(coords_list, list):
                raise ValueError("Coordinates must be a list")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid coordinates format: {e}")
        
        contents = await file.read()
        
        # Run OCR
        loop = asyncio.get_event_loop()
        page_result = await loop.run_in_executor(None, ocr.image_to_page_words, contents)
        
        # Find matching words
        matching_words = matching.find_matching_words(page_result, coords_list)
        
        # Create result with only matching words
        matched_page = page_result.model_copy()
        matched_page.words = matching_words
        
        return OCRResponse(pages=[matched_page])
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mask/image")
async def mask_image(
    file: UploadFile = File(...),
    fields: str = None,
    mask: MaskService = Depends(get_mask_service)
):
    """Apply masks to image."""
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not fields:
            raise HTTPException(status_code=400, detail="Mask fields are required")
        
        # Parse mask fields from JSON string
        try:
            fields_data = json.loads(fields)
            mask_fields = [MaskField(**field) for field in fields_data]
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid fields format: {e}")
        
        contents = await file.read()
        
        # Apply masks
        loop = asyncio.get_event_loop()
        masked_bytes = await loop.run_in_executor(None, mask.mask_image, contents, mask_fields)
        
        return Response(content=masked_bytes, media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mask/pdf")
async def mask_pdf(
    file: UploadFile = File(...),
    fields: str = None,
    mask: MaskService = Depends(get_mask_service)
):
    """Apply masks to PDF."""
    try:
        if not file.content_type or file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        if not fields:
            raise HTTPException(status_code=400, detail="Mask fields are required")
        
        # Parse mask fields from JSON string (per page)
        try:
            fields_data = json.loads(fields)
            mask_fields_per_page = []
            for page_fields in fields_data:
                page_masks = [MaskField(**field) for field in page_fields]
                mask_fields_per_page.append(page_masks)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid fields format: {e}")
        
        contents = await file.read()
        
        # Apply masks
        loop = asyncio.get_event_loop()
        masked_bytes = await loop.run_in_executor(None, mask.mask_pdf, contents, mask_fields_per_page)
        
        return Response(content=masked_bytes, media_type="application/pdf")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regex/generate", response_model=RegexGenerateResponse)
async def generate_regex(
    request: RegexGenerateRequest,
    regex_service: RegexService = Depends(get_regex_service)
):
    """Generate regex pattern from phrases."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, regex_service.generate_regex, request.phrases)
        
        return RegexGenerateResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regex/stored")
async def get_stored_regexes(
    regex_service: RegexService = Depends(get_regex_service)
):
    """Get all stored regex patterns."""
    try:
        return {"regexes": regex_service.get_stored_regexes()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regex/match/image")
async def regex_match_image(
    file: UploadFile = File(...),
    regex: str = Form(...),
    ocr: OCRService = Depends(get_ocr_service),
    regex_service: RegexService = Depends(get_regex_service)
):
    """Match regex pattern against OCR text in image."""
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not regex:
            raise HTTPException(status_code=400, detail="Regex pattern is required")
        
        # Validate regex
        if not regex_service.validate_regex(regex):
            raise HTTPException(status_code=400, detail="Invalid regex pattern")
        
        contents = await file.read()
        
        # Run OCR
        loop = asyncio.get_event_loop()
        page_result = await loop.run_in_executor(None, ocr.image_to_page_words, contents)
        
        # Extract texts and return matches with coords
        try:
            compiled = re.compile(regex)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {e}")

        matches_with_coords: List[Dict[str, Any]] = []
        for word in page_result.words:
            text = word.text or ""
            for m in compiled.finditer(text):
                try:
                    matched_text = m.group(0)
                except Exception:
                    start, end = m.span()
                    matched_text = text[start:end]
                matches_with_coords.append({
                    "text": matched_text,
                    "coords": word.coords,
                })

        return {"matches": matches_with_coords}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regex/match/pdf")
async def regex_match_pdf(
    file: UploadFile = File(...),
    regex: str = Form(...),
    ocr: OCRService = Depends(get_ocr_service),
    regex_service: RegexService = Depends(get_regex_service)
):
    """Match regex pattern against OCR text in PDF."""
    try:
        if not file.content_type or file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        if not regex:
            raise HTTPException(status_code=400, detail="Regex pattern is required")
        
        # Validate regex
        if not regex_service.validate_regex(regex):
            raise HTTPException(status_code=400, detail="Invalid regex pattern")
        
        contents = await file.read()
        
        # Run OCR
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(None, ocr.pdf_to_pages, contents)
        
        # Aggregate matches (text + coords) across all pages
        try:
            compiled = re.compile(regex)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid regex pattern: {e}")

        matches_with_coords: List[Dict[str, Any]] = []
        for page in pages:
            for word in page.words:
                text = word.text or ""
                for m in compiled.finditer(text):
                    try:
                        matched_text = m.group(0)
                    except Exception:
                        start, end = m.span()
                        matched_text = text[start:end]
                    matches_with_coords.append({
                        "text": matched_text,
                        "coords": word.coords,
                    })

        return {"matches": matches_with_coords}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/labels")
async def get_model_labels(
    model: ModelService = Depends(get_model_service)
):
    """Get the label mappings from the loaded models."""
    try:
        # Get labels from both models
        binary_labels = {}
        label_labels = {}
        
        if hasattr(model, 'binary_id2label'):
            binary_labels = model.binary_id2label
        
        if hasattr(model, 'label_id2label'):
            label_labels = model.label_id2label
        
        return {
            "binary_labels": binary_labels,
            "label_labels": label_labels,
            "binary_label_count": len(binary_labels),
            "label_count": len(label_labels)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

