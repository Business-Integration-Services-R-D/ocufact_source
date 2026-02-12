from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any


class OCRWord(BaseModel):
    text: str
    coords: List[int]  # [minx, miny, maxx, maxy]
    confidence: float


class PageResult(BaseModel):
    page: int
    words: List[OCRWord]


class OCRResponse(BaseModel):
    pages: List[PageResult]


class DetectionWord(BaseModel):
    text: str
    coords: List[int]  # [minx, miny, maxx, maxy]
    confidence: float
    is_sensitive: Optional[bool] = None
    label: Optional[str] = None
    score: Optional[float] = None


class DetectionPageResult(BaseModel):
    page: int
    words: List[DetectionWord]


class DetectionResponse(BaseModel):
    pages: List[DetectionPageResult]


class SensitiveWord(BaseModel):
    text: str
    label: Optional[str] = None


class SensitivePageResult(BaseModel):
    page: int
    words: List[SensitiveWord]


class SensitiveResponse(BaseModel):
    pages: List[SensitivePageResult]


class MaskField(BaseModel):
    coords: List[int]  # [minx, miny, maxx, maxy]
    color: str = "#FF0000"


class MaskRequest(BaseModel):
    fields: List[MaskField]


class RegexGenerateRequest(BaseModel):
    phrases: List[str]


class RegexGenerateResponse(BaseModel):
    regex: str
    confidence: float


class RegexMatchRequest(BaseModel):
    regex: str


class RegexMatchWord(BaseModel):
    text: str
    coords: List[int]  # [minx, miny, maxx, maxy]
    confidence: float
    match: bool


class RegexMatchPageResult(BaseModel):
    page: int
    words: List[RegexMatchWord]


class RegexMatchResponse(BaseModel):
    pages: List[RegexMatchPageResult]


class CoordinateMatchRequest(BaseModel):
    coords: List[List[int]]  # List of [minx, miny, maxx, maxy]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

