from typing import List, Tuple
from ..schemas import OCRWord, PageResult
from ..utils import boxes_overlap, bbox_overlap_ratio


class MatchingService:
    def __init__(self, overlap_threshold: float = 0.1):
        self.overlap_threshold = overlap_threshold
    
    def find_matching_words(self, page_result: PageResult, query_coords: List[List[int]]) -> List[OCRWord]:
        """Find OCR words that overlap with query coordinates."""
        matching_words = []
        
        for word in page_result.words:
            for query_coord in query_coords:
                if boxes_overlap(word.coords, query_coord, self.overlap_threshold):
                    matching_words.append(word)
                    break  # Avoid duplicates
        
        return matching_words
    
    def find_words_in_region(self, page_result: PageResult, region_coords: List[int]) -> List[OCRWord]:
        """Find all OCR words that overlap with a specific region."""
        matching_words = []
        
        for word in page_result.words:
            if boxes_overlap(word.coords, region_coords, self.overlap_threshold):
                matching_words.append(word)
        
        return matching_words
    
    def calculate_overlap_scores(self, page_result: PageResult, query_coords: List[List[int]]) -> List[Tuple[OCRWord, float]]:
        """Calculate overlap scores between OCR words and query coordinates."""
        word_scores = []
        
        for word in page_result.words:
            max_overlap = 0.0
            for query_coord in query_coords:
                overlap = bbox_overlap_ratio(word.coords, query_coord)
                max_overlap = max(max_overlap, overlap)
            
            if max_overlap > 0:
                word_scores.append((word, max_overlap))
        
        # Sort by overlap score (highest first)
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        return word_scores
    
    def filter_by_confidence(self, words: List[OCRWord], min_confidence: float = 0.5) -> List[OCRWord]:
        """Filter OCR words by minimum confidence threshold."""
        return [word for word in words if word.confidence >= min_confidence]
    
    def group_nearby_words(self, words: List[OCRWord], max_distance: int = 50) -> List[List[OCRWord]]:
        """Group words that are close to each other spatially."""
        if not words:
            return []
        
        groups = []
        remaining_words = words.copy()
        
        while remaining_words:
            current_group = [remaining_words.pop(0)]
            
            # Find words close to any word in current group
            i = 0
            while i < len(remaining_words):
                word = remaining_words[i]
                
                # Check if word is close to any word in current group
                is_close = False
                for group_word in current_group:
                    distance = self._calculate_distance(word.coords, group_word.coords)
                    if distance <= max_distance:
                        is_close = True
                        break
                
                if is_close:
                    current_group.append(remaining_words.pop(i))
                else:
                    i += 1
            
            groups.append(current_group)
        
        return groups
    
    def _calculate_distance(self, coords1: List[int], coords2: List[int]) -> float:
        """Calculate distance between centers of two bounding boxes."""
        x1_center = (coords1[0] + coords1[2]) / 2
        y1_center = (coords1[1] + coords1[3]) / 2
        x2_center = (coords2[0] + coords2[2]) / 2
        y2_center = (coords2[1] + coords2[3]) / 2
        
        return ((x2_center - x1_center) ** 2 + (y2_center - y1_center) ** 2) ** 0.5
    
    def merge_overlapping_boxes(self, coords_list: List[List[int]], overlap_threshold: float = 0.5) -> List[List[int]]:
        """Merge bounding boxes that overlap significantly."""
        if not coords_list:
            return []
        
        merged = []
        remaining = coords_list.copy()
        
        while remaining:
            current = remaining.pop(0)
            
            # Find all boxes that overlap with current
            overlapping = [current]
            i = 0
            while i < len(remaining):
                if bbox_overlap_ratio(current, remaining[i]) >= overlap_threshold:
                    overlapping.append(remaining.pop(i))
                else:
                    i += 1
            
            # Merge all overlapping boxes
            if overlapping:
                min_x = min(box[0] for box in overlapping)
                min_y = min(box[1] for box in overlapping)
                max_x = max(box[2] for box in overlapping)
                max_y = max(box[3] for box in overlapping)
                merged.append([min_x, min_y, max_x, max_y])
        
        return merged

