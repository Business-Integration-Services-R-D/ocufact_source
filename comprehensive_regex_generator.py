"""
Comprehensive Regex Generator

A sophisticated rule-based system that analyzes input phrases to generate regex patterns
by dissecting them into common structural components. The algorithm identifies:
- Longest common prefix and suffix across all inputs
- Common special characters and their positions
- Repeating structural patterns
- Variable vs fixed-length segments
- Optional components

The system builds regex from the ground up without using predefined templates,
focusing on understanding the actual structure of the input data.
"""

import re
from typing import List, Dict, Set, Tuple, Optional, Any, Union
from collections import defaultdict, Counter
from dataclasses import dataclass
import string


@dataclass
class SegmentPattern:
    """Represents a structural segment in the pattern"""
    content_type: str  # 'literal', 'digits', 'letters', 'mixed', 'special'
    min_length: int
    max_length: int
    case_pattern: str  # 'upper', 'lower', 'mixed', 'none'
    sample_values: Set[str]
    is_optional: bool = False
    position: int = 0


@dataclass
class StructuralAnchor:
    """Represents a fixed anchor point in the pattern"""
    character: str
    positions: List[int]  # positions across different samples
    is_consistent: bool  # appears in same relative position
    frequency: float  # how often it appears


class ComprehensiveRegexGenerator:
    """
    Main class for generating regex patterns through comprehensive structural analysis
    """
    
    def __init__(self):
        self.input_phrases: List[str] = []
        self.analysis_cache: Dict[str, Any] = {}
        self.structural_anchors: List[StructuralAnchor] = []
        self.segment_patterns: List[SegmentPattern] = []
        
    def add_phrases(self, phrases: List[str]) -> None:
        """Add input phrases for analysis"""
        self.input_phrases = [str(p).strip() for p in phrases if p and str(p).strip()]
        self.analysis_cache.clear()
        
    def find_longest_common_prefix(self) -> str:
        """Find the longest common prefix across all input phrases"""
        if not self.input_phrases:
            return ""
            
        if len(self.input_phrases) == 1:
            return ""
            
        prefix = ""
        min_length = min(len(phrase) for phrase in self.input_phrases)
        
        for i in range(min_length):
            char = self.input_phrases[0][i]
            if all(phrase[i] == char for phrase in self.input_phrases):
                prefix += char
            else:
                break
                
        return prefix
    
    def find_longest_common_suffix(self) -> str:
        """Find the longest common suffix across all input phrases"""
        if not self.input_phrases:
            return ""
            
        if len(self.input_phrases) == 1:
            return ""
            
        suffix = ""
        min_length = min(len(phrase) for phrase in self.input_phrases)
        
        for i in range(1, min_length + 1):
            char = self.input_phrases[0][-i]
            if all(phrase[-i] == char for phrase in self.input_phrases):
                suffix = char + suffix
            else:
                break
                
        return suffix
    
    def extract_special_characters(self) -> Dict[str, List[int]]:
        """Extract special characters and their positions across all phrases"""
        special_chars = {}
        punctuation_and_symbols = set(string.punctuation + string.whitespace)
        
        for phrase in self.input_phrases:
            for i, char in enumerate(phrase):
                if char in punctuation_and_symbols:
                    if char not in special_chars:
                        special_chars[char] = []
                    special_chars[char].append(i)
                    
        return special_chars
    
    def analyze_character_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Analyze character patterns in a text segment"""
        patterns = []
        i = 0
        
        while i < len(text):
            current_char = text[i]
            
            # Determine character type
            if current_char.isdigit():
                char_type = 'digit'
                pattern_func = lambda c: c.isdigit()
            elif current_char.isalpha():
                if current_char.isupper():
                    char_type = 'upper_letter'
                    pattern_func = lambda c: c.isupper()
                else:
                    char_type = 'lower_letter'
                    pattern_func = lambda c: c.islower()
            elif current_char.isspace():
                char_type = 'whitespace'
                pattern_func = lambda c: c.isspace()
            else:
                char_type = 'special'
                pattern_func = lambda c: not (c.isalnum() or c.isspace())
            
            # Count consecutive characters of same type
            j = i
            segment_content = ""
            while j < len(text) and pattern_func(text[j]):
                segment_content += text[j]
                j += 1
            
            patterns.append({
                'type': char_type,
                'content': segment_content,
                'length': j - i,
                'start_pos': i,
                'end_pos': j
            })
            
            i = j
            
        return patterns
    
    def find_structural_anchors(self) -> List[StructuralAnchor]:
        """Identify consistent structural anchor points across phrases"""
        special_chars = self.extract_special_characters()
        anchors = []
        
        for char, positions in special_chars.items():
            # Calculate frequency
            frequency = len(positions) / len(self.input_phrases)
            
            # Check if positions are relatively consistent
            if len(set(positions)) <= len(self.input_phrases):  # Allow some variation
                # Calculate relative positions
                relative_positions = []
                for i, phrase in enumerate(self.input_phrases):
                    phrase_positions = [pos for pos in positions if pos < len(phrase) and phrase[pos] == char]
                    if phrase_positions:
                        # Use relative position (percentage of phrase length)
                        rel_pos = phrase_positions[0] / len(phrase)
                        relative_positions.append(rel_pos)
                
                # Check consistency of relative positions
                if relative_positions:
                    pos_variance = max(relative_positions) - min(relative_positions)
                    is_consistent = pos_variance < 0.3  # Allow 30% variance
                    
                    anchors.append(StructuralAnchor(
                        character=char,
                        positions=positions,
                        is_consistent=is_consistent,
                        frequency=frequency
                    ))
        
        # Sort by frequency and consistency
        anchors.sort(key=lambda x: (x.frequency, x.is_consistent), reverse=True)
        return anchors
    
    def segment_by_anchors(self, phrase: str, anchors: List[StructuralAnchor]) -> List[str]:
        """Segment a phrase using identified structural anchors"""
        if not anchors:
            return [phrase]
        
        # Find anchor positions in this specific phrase
        anchor_positions = []
        for anchor in anchors:
            for i, char in enumerate(phrase):
                if char == anchor.character:
                    anchor_positions.append((i, anchor.character))
        
        # Sort by position
        anchor_positions.sort()
        
        # Create segments
        segments = []
        last_pos = 0
        
        for pos, anchor_char in anchor_positions:
            # Add segment before anchor
            if pos > last_pos:
                segments.append(phrase[last_pos:pos])
            
            # Add anchor as separate segment
            segments.append(anchor_char)
            last_pos = pos + 1
        
        # Add final segment
        if last_pos < len(phrase):
            segments.append(phrase[last_pos:])
        
        return [seg for seg in segments if seg]  # Remove empty segments
    
    def analyze_segment_patterns(self, segments_by_phrase: List[List[str]]) -> List[SegmentPattern]:
        """Analyze patterns in corresponding segments across phrases"""
        if not segments_by_phrase:
            return []
        
        max_segments = max(len(segments) for segments in segments_by_phrase)
        segment_patterns = []
        
        for seg_idx in range(max_segments):
            # Collect corresponding segments
            corresponding_segments = []
            for phrase_segments in segments_by_phrase:
                if seg_idx < len(phrase_segments):
                    corresponding_segments.append(phrase_segments[seg_idx])
            
            if not corresponding_segments:
                continue
            
            # Analyze this segment position
            segment_analysis = self.analyze_segment_group(corresponding_segments, seg_idx)
            if segment_analysis:
                segment_patterns.append(segment_analysis)
        
        return segment_patterns
    
    def analyze_segment_group(self, segments: List[str], position: int) -> Optional[SegmentPattern]:
        """Analyze a group of corresponding segments"""
        if not segments:
            return None
        
        # Determine content type
        content_types = set()
        lengths = []
        case_patterns = set()
        sample_values = set(segments)
        
        for segment in segments:
            lengths.append(len(segment))
            
            if segment.isdigit():
                content_types.add('digits')
                case_patterns.add('none')
            elif segment.isalpha():
                content_types.add('letters')
                if segment.isupper():
                    case_patterns.add('upper')
                elif segment.islower():
                    case_patterns.add('lower')
                else:
                    case_patterns.add('mixed')
            elif segment.isalnum():
                content_types.add('mixed')
                if segment.isupper():
                    case_patterns.add('upper')
                elif segment.islower():
                    case_patterns.add('lower')
                else:
                    case_patterns.add('mixed')
            elif len(segment) == 1 and not segment.isalnum():
                content_types.add('special')
                case_patterns.add('none')
            else:
                # Check if it's a literal (same across all samples)
                if len(set(segments)) == 1:
                    content_types.add('literal')
                    case_patterns.add('none')
                else:
                    content_types.add('mixed')
                    case_patterns.add('mixed')
        
        # Determine dominant patterns
        dominant_content_type = max(content_types, key=lambda x: sum(1 for s in segments if self.matches_content_type(s, x)))
        dominant_case_pattern = max(case_patterns, key=case_patterns.count) if case_patterns else 'none'
        
        # Check if optional (appears in less than all phrases)
        is_optional = len(segments) < len(self.input_phrases)
        
        return SegmentPattern(
            content_type=dominant_content_type,
            min_length=min(lengths) if lengths else 0,
            max_length=max(lengths) if lengths else 0,
            case_pattern=dominant_case_pattern,
            sample_values=sample_values,
            is_optional=is_optional,
            position=position
        )
    
    def matches_content_type(self, segment: str, content_type: str) -> bool:
        """Check if a segment matches a specific content type"""
        if content_type == 'digits':
            return segment.isdigit()
        elif content_type == 'letters':
            return segment.isalpha()
        elif content_type == 'mixed':
            return segment.isalnum() and not segment.isdigit() and not segment.isalpha()
        elif content_type == 'special':
            return len(segment) == 1 and not segment.isalnum()
        elif content_type == 'literal':
            return True  # Literals are determined by consistency across samples
        else:
            return False
    
    def detect_repeating_patterns(self) -> Dict[str, Any]:
        """Detect repeating structural patterns within and across phrases"""
        patterns = {}
        
        for phrase in self.input_phrases:
            # Look for repeating subsequences
            char_patterns = self.analyze_character_patterns(phrase)
            
            # Find repeating pattern groups
            for i in range(len(char_patterns) - 1):
                for j in range(i + 2, len(char_patterns) + 1):
                    subpattern = char_patterns[i:j]
                    pattern_key = tuple((p['type'], p['length']) for p in subpattern)
                    
                    if pattern_key not in patterns:
                        patterns[pattern_key] = {
                            'count': 0,
                            'examples': [],
                            'positions': []
                        }
                    
                    patterns[pattern_key]['count'] += 1
                    patterns[pattern_key]['examples'].append(''.join(p['content'] for p in subpattern))
                    patterns[pattern_key]['positions'].append(i)
        
        # Filter for actually repeating patterns
        repeating_patterns = {k: v for k, v in patterns.items() if v['count'] > len(self.input_phrases)}
        
        return repeating_patterns
    
    def generate_regex_component(self, segment: SegmentPattern) -> str:
        """Generate regex component for a specific segment pattern"""
        if segment.content_type == 'literal':
            # If all samples have the same literal value
            if len(segment.sample_values) == 1:
                return re.escape(next(iter(segment.sample_values)))
            else:
                # Multiple literal options - use alternation
                escaped_values = [re.escape(val) for val in sorted(segment.sample_values)]
                return f"(?:{'|'.join(escaped_values)})"
        
        elif segment.content_type == 'digits':
            base_pattern = r'\d'
            
        elif segment.content_type == 'letters':
            if segment.case_pattern == 'upper':
                base_pattern = r'[A-Z]'
            elif segment.case_pattern == 'lower':
                base_pattern = r'[a-z]'
            else:
                base_pattern = r'[A-Za-z]'
                
        elif segment.content_type == 'mixed':
            base_pattern = r'[A-Za-z0-9]'
            
        elif segment.content_type == 'special':
            # Handle special characters
            if len(segment.sample_values) == 1:
                return re.escape(next(iter(segment.sample_values)))
            else:
                escaped_chars = [re.escape(char) for char in segment.sample_values]
                return f"[{''.join(escaped_chars)}]"
        
        else:
            # Fallback for unknown types
            base_pattern = r'.'
        
        # Add quantifiers based on length patterns
        if segment.min_length == segment.max_length:
            if segment.min_length == 1:
                quantifier = ""
            else:
                quantifier = f"{{{segment.min_length}}}"
        else:
            if segment.min_length == 0:
                if segment.max_length == 1:
                    quantifier = "?"
                else:
                    quantifier = f"{{0,{segment.max_length}}}"
            elif segment.min_length == 1:
                if segment.max_length > 10:  # Avoid very large quantifiers
                    quantifier = "+"
                else:
                    quantifier = f"{{{segment.min_length},{segment.max_length}}}"
            else:
                quantifier = f"{{{segment.min_length},{segment.max_length}}}"
        
        component = base_pattern + quantifier
        
        # Make optional if needed
        if segment.is_optional:
            component = f"(?:{component})?"
        
        return component
    
    def optimize_regex(self, regex: str) -> str:
        """Optimize the generated regex pattern"""
        # Remove redundant groups
        regex = re.sub(r'\(([^)]*)\)\{1\}', r'\1', regex)
        
        # Simplify single character classes
        regex = re.sub(r'\[(.)\]', lambda m: re.escape(m.group(1)), regex)
        
        # Optimize quantifiers
        regex = re.sub(r'\{1\}', '', regex)
        regex = re.sub(r'\{0,1\}', '?', regex)
        regex = re.sub(r'\{1,\}', '+', regex)
        regex = re.sub(r'\{0,\}', '*', regex)
        
        # Merge adjacent similar patterns
        regex = re.sub(r'\\d\+\\d\+', r'\\d+', regex)
        regex = re.sub(r'\[A-Za-z\]\+\[A-Za-z\]\+', r'[A-Za-z]+', regex)
        
        return regex
    
    def generate_regex(self) -> str:
        """Main method to generate regex pattern from input phrases"""
        if not self.input_phrases:
            return ""
        
        # Step 1: Find common prefix and suffix
        common_prefix = self.find_longest_common_prefix()
        common_suffix = self.find_longest_common_suffix()
        
        # Step 2: Remove prefix and suffix from phrases for core analysis
        core_phrases = []
        for phrase in self.input_phrases:
            start_idx = len(common_prefix)
            end_idx = len(phrase) - len(common_suffix) if common_suffix else len(phrase)
            core_phrase = phrase[start_idx:end_idx]
            core_phrases.append(core_phrase)
        
        # Step 3: Find structural anchors in core phrases
        temp_phrases = self.input_phrases
        self.input_phrases = core_phrases
        structural_anchors = self.find_structural_anchors()
        self.input_phrases = temp_phrases
        
        # Step 4: Segment phrases by anchors
        segments_by_phrase = []
        for core_phrase in core_phrases:
            segments = self.segment_by_anchors(core_phrase, structural_anchors)
            segments_by_phrase.append(segments)
        
        # Step 5: Analyze segment patterns
        segment_patterns = self.analyze_segment_patterns(segments_by_phrase)
        
        # Step 6: Generate regex components
        regex_parts = []
        
        # Add prefix
        if common_prefix:
            regex_parts.append(re.escape(common_prefix))
        
        # Add core pattern
        for segment in segment_patterns:
            component = self.generate_regex_component(segment)
            regex_parts.append(component)
        
        # Add suffix
        if common_suffix:
            regex_parts.append(re.escape(common_suffix))
        
        # Combine and optimize
        regex = ''.join(regex_parts)
        regex = self.optimize_regex(regex)
        
        return regex
    
    def validate_regex(self, regex: str) -> Dict[str, Any]:
        """Validate the generated regex against input phrases"""
        if not regex:
            return {'valid': False, 'error': 'Empty regex pattern'}
        
        try:
            compiled_regex = re.compile(f'^{regex}$')
        except re.error as e:
            return {'valid': False, 'error': f'Regex compilation failed: {e}'}
        
        matches = 0
        failures = []
        
        for phrase in self.input_phrases:
            if compiled_regex.match(phrase):
                matches += 1
            else:
                failures.append(phrase)
        
        match_rate = matches / len(self.input_phrases) if self.input_phrases else 0
        
        return {
            'valid': True,
            'match_rate': match_rate,
            'total_matches': matches,
            'total_phrases': len(self.input_phrases),
            'failures': failures[:5]  # Show first 5 failures
        }
    
    def analyze_input_structure(self) -> Dict[str, Any]:
        """Comprehensive analysis of input structure"""
        if not self.input_phrases:
            return {}
        
        analysis = {
            'phrase_count': len(self.input_phrases),
            'common_prefix': self.find_longest_common_prefix(),
            'common_suffix': self.find_longest_common_suffix(),
            'structural_anchors': self.find_structural_anchors(),
            'repeating_patterns': self.detect_repeating_patterns(),
            'length_distribution': Counter(len(phrase) for phrase in self.input_phrases),
            'character_frequency': Counter(''.join(self.input_phrases)),
            'special_characters': self.extract_special_characters()
        }
        
        return analysis


def main():
    """Main function with example input arrays"""
    
    # Example 1: Email-like patterns
    email_patterns = [
        "abc@mail.com",
        "dfdfdfb@mm.com.tr",
        "dfdfdfb@mm.com.tr"
    ]
    
    # Example 2: Dash-separated patterns
    dash_patterns = [
        "123-abc-123",
        "323-dfgf-323",
        "999-QQQ-000"
    ]
    
    # Example 3: Phone number patterns
    phone_patterns = [
        "+90 532 123 45 67",
        "+90 541 987 65 43",
        "+90 555 111 22 33"
    ]
    
    # Example 4: Mixed format patterns
    mixed_patterns = [
        "ID-2023-ABC-001",
        "ID-2024-XYZ-002",
        "ID-2023-DEF-003"
    ]
    
    test_cases = [
        ("Email Patterns", email_patterns),
        ("Dash Patterns", dash_patterns),
        ("Phone Patterns", phone_patterns),
        ("Mixed Patterns", mixed_patterns)
    ]
    
    for test_name, patterns in test_cases:
        generator = ComprehensiveRegexGenerator()
        generator.add_phrases(patterns)
        
        # Generate regex
        regex_pattern = generator.generate_regex()
        
        # Validate
        validation = generator.validate_regex(regex_pattern)
        
        # Analyze structure
        analysis = generator.analyze_input_structure()
        
        # Store results in variables (no printing as per requirements)
        test_result = {
            'name': test_name,
            'input_patterns': patterns,
            'generated_regex': regex_pattern,
            'validation': validation,
            'analysis': analysis
        }
        
        # Variable assignment for inspection
        locals()[f'result_{test_name.lower().replace(" ", "_")}'] = test_result


if __name__ == "__main__":
    main()
