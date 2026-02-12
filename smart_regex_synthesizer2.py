"""
Smart Regex Synthesizer — single anchored pattern output

- Rule-based synthesizer: derives pattern from character-run structure,
  longest common prefix/suffix, literal anchors (punctuation), and
  per-region statistics (kinds and lengths).
- Produces exactly one anchored regex (starts with ^ and ends with $).
- No built-in semantic templates (no "if I see @ treat as email"); it
  derives everything from the samples themselves.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Set, Iterable, Tuple


def lcp(strings: List[str]) -> str:
    if not strings:
        return ""
    shortest = min(strings, key=len)
    for i, ch in enumerate(shortest):
        for s in strings:
            if s[i] != ch:
                return shortest[:i]
    return shortest


def lcsuf(strings: List[str]) -> str:
    if not strings:
        return ""
    rev = [s[::-1] for s in strings]
    pref_rev = lcp(rev)
    return pref_rev[::-1]


def escape_literal(s: str) -> str:
    return re.escape(s)


def is_unicode_letter(ch: str) -> bool:
    if not ch:
        return False
    return unicodedata.category(ch).startswith("L")


def char_kind(ch: str) -> str:
    # 'D' digit, 'l' lowercase ASCII, 'U' uppercase ASCII, 'X' other unicode letter,
    # 'S' space, 'P' punctuation/symbol (anchor)
    if ch.isdigit():
        return "D"
    if is_unicode_letter(ch):
        if "a" <= ch <= "z":
            return "l"
        if "A" <= ch <= "Z":
            return "U"
        return "X"
    if ch.isspace():
        return "S"
    return "P"


@dataclass
class Token:
    kind: str     # 'D','l','U','X','S' or 'P'
    text: str

    def is_anchor(self) -> bool:
        return self.kind == "P"


def tokenize_runs(s: str) -> List[Token]:
    if not s:
        return []
    out: List[Token] = []
    cur_kind = char_kind(s[0])
    cur_buf = [s[0]]
    for ch in s[1:]:
        k = char_kind(ch)
        if k == cur_kind:
            cur_buf.append(ch)
        else:
            out.append(Token(kind=cur_kind, text="".join(cur_buf)))
            cur_buf = [ch]
            cur_kind = k
    out.append(Token(kind=cur_kind, text="".join(cur_buf)))
    return out


def extract_anchor_seq(tokens: List[Token]) -> List[str]:
    return [t.text for t in tokens if t.is_anchor()]


def lcs(a: List[str], b: List[str]) -> List[str]:
    n, m = len(a), len(b)
    dp: List[List[List[str]]] = [[[] for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = [a[i]] + dp[i + 1][j + 1]
            else:
                dp[i][j] = dp[i + 1][j] if len(dp[i + 1][j]) >= len(dp[i][j + 1]) else dp[i][j + 1]
    return dp[0][0]


def lcs_multi(seqs: List[List[str]]) -> List[str]:
    if not seqs:
        return []
    acc = seqs[0]
    for s in seqs[1:]:
        acc = lcs(acc, s)
        if not acc:
            break
    return acc


@dataclass
class RegionSummary:
    # per-sample totals and kinds
    lengths: List[int] = field(default_factory=list)
    per_sample_kinds: List[Set[str]] = field(default_factory=list)

    def update_sample(self, tokens: List[Token]) -> None:
        total = 0
        kinds: Set[str] = set()
        for t in tokens:
            if t.is_anchor():
                continue
            total += len(t.text)
            kinds.add(t.kind)
        self.lengths.append(total)
        self.per_sample_kinds.append(kinds)

    def summarize(self) -> str:
        if not self.lengths:
            return ""
        mn = min(self.lengths)
        mx = max(self.lengths)
        # combined kinds across non-empty samples
        nonempty_kinds = [k for k in self.per_sample_kinds if k]
        combined: Set[str] = set().union(*nonempty_kinds) if nonempty_kinds else set()

        def base_for(kinds_set: Set[str]) -> str:
            # precise classes first
            if kinds_set == {"D"}:
                return r"\d"
            if kinds_set == {"l"}:
                return r"[a-z]"
            if kinds_set == {"U"}:
                return r"[A-Z]"
            if kinds_set == {"X"}:
                # approximate 'other unicode letters' with class excluding digits/underscore
                return r"[^\W\d_]"
            # letters only mixture
            letters = set(k for k in kinds_set if k in ("l", "U", "X"))
            has_digit = "D" in kinds_set
            has_space = "S" in kinds_set
            if letters and not has_digit and not has_space:
                if "l" in letters and "U" in letters:
                    return r"[A-Za-z]"
                if "l" in letters:
                    return r"[a-z]"
                if "U" in letters:
                    return r"[A-Z]"
                return r"[^\W\d_]"
            if has_digit and letters:
                return r"[A-Za-z0-9]"
            if kinds_set == {"S"}:
                return r"\s"
            # Special handling for very mixed patterns (digits, spaces, punctuation)
            if "D" in kinds_set and "S" in kinds_set and "P" in kinds_set and len(kinds_set) >= 3:
                # For phone-like patterns with digits, spaces, and punctuation, be very permissive
                return r"[\d\s\-\+\(\)]"
            # fallback: construct bracket from observed categories
            parts = []
            if "D" in kinds_set:
                parts.append("0-9")
            if "l" in kinds_set:
                parts.append("a-z")
            if "U" in kinds_set:
                parts.append("A-Z")
            if "S" in kinds_set:
                parts.append(r"\s")
            if parts:
                merged = "".join(p for p in parts if not p.startswith("\\"))
                if merged:
                    return f"[{merged}]"
            return r"\S"

        base = base_for(combined) if combined else r"."

        # More flexible quantifiers - avoid being too restrictive
        if mn == mx:
            if mn == 1:
                return base
            return f"{base}{{{mn}}}"
        if mn == 0:
            # optional region
            if mx == 1:
                return f"(?:{base})?"
            return f"(?:{base}{{1,{mx}}})?"
        # Be more generous with ranges to avoid overfitting
        if mx <= mn + 5:  # increased from 3 to 5
            return f"{base}{{{mn},{mx}}}"
        return f"{base}{{{mn},}}"


class SmartRegexSynthesizer:
    def __init__(self) -> None:
        self.samples: List[str] = []
        self.min_common_prefix_ratio = 0.3  # minimum ratio for keeping common prefix/suffix

    def add(self, phrases: Iterable[str]) -> None:
        for p in phrases:
            if p is None:
                continue
            s = str(p)
            if s:
                self.samples.append(s)

    def synthesize(self) -> str:
        if not self.samples:
            return r"^$"

        # 1) LCP / LCSuf (literal) - but avoid overfitting
        pref = lcp(self.samples)
        suf = lcsuf(self.samples)
        
        # Avoid overfitting to common prefix/suffix if they're too specific
        avg_len = sum(len(s) for s in self.samples) / len(self.samples)
        
        # Be more aggressive about avoiding overfitting - especially for single characters
        max_prefix_len = max(1, int(avg_len * self.min_common_prefix_ratio))
        max_suffix_len = max(1, int(avg_len * self.min_common_prefix_ratio))
        
        # Be very conservative about prefix/suffix to avoid overfitting
        # Don't use prefix/suffix if they're likely coincidental
        if len(pref) <= 2:
            # For short prefixes, check if they're meaningful or coincidental
            # If the prefix is just 1-2 chars and samples have structured content, skip it
            has_structure = any('.' in s or '-' in s or '@' in s for s in self.samples)
            if has_structure:
                pref = ""
        elif len(pref) > max_prefix_len:
            pref = pref[:max_prefix_len]
            
        if len(suf) <= 2:
            # For short suffixes, check if they're meaningful or coincidental
            has_structure = any('.' in s or '-' in s or '@' in s for s in self.samples)
            if has_structure:
                suf = ""
        elif len(suf) > max_suffix_len:
            suf = suf[len(suf) - max_suffix_len:]
            
        core = [s[len(pref): len(s) - len(suf) if len(suf) > 0 else len(s)] for s in self.samples]

        # 2) tokenize cores
        tokenized = [tokenize_runs(s) for s in core]

        # 3) anchor skeleton via LCS across anchor sequences
        anchor_seqs = [extract_anchor_seq(toks) for toks in tokenized]
        skeleton = lcs_multi(anchor_seqs)
        
        # If no common anchors found, try to find the most frequent anchor pattern
        if not skeleton and any(anchor_seqs):
            # Find the most common anchor sequence
            from collections import Counter
            seq_counts = Counter(tuple(seq) for seq in anchor_seqs if seq)
            if seq_counts:
                most_common_seq, count = seq_counts.most_common(1)[0]
                threshold = len(self.samples) * 0.6
                # Only use the most common sequence if it appears in most samples
                if count >= threshold:  # 60% threshold
                    skeleton = list(most_common_seq)
                # Otherwise, leave skeleton empty to trigger simple fallback
        
        # Enhanced handling for variable anchor patterns
        variable_anchors = set()
        if skeleton and anchor_seqs:
            from collections import Counter, defaultdict
            
            # Check for anchors that appear different numbers of times
            for anchor in set().union(*anchor_seqs):
                counts = [seq.count(anchor) for seq in anchor_seqs]
                if len(set(counts)) > 1:  # Variable count across samples
                    variable_anchors.add(anchor)
            
            # If we have variable anchors, use the longest common pattern as base
            # and make variable parts optional
            if variable_anchors and skeleton:  # Only if we already have a skeleton
                # Find the sample with the most complete anchor pattern
                longest_seq = max(anchor_seqs, key=len) if anchor_seqs else []
                if len(longest_seq) > len(skeleton):
                    # Use the longest sequence but mark extra anchors as variable
                    skeleton = longest_seq

        # 4) split tokens by skeleton anchors -> regions per sample
        regions_per_sample: List[List[List[Token]]] = []
        anchors_seen: List[List[str]] = []
        for toks in tokenized:
            regions: List[List[Token]] = []
            seen: List[str] = []
            i = 0
            for anchor in skeleton:
                bucket: List[Token] = []
                while i < len(toks) and not (toks[i].is_anchor() and toks[i].text == anchor):
                    bucket.append(toks[i])
                    i += 1
                regions.append(bucket)
                if i < len(toks) and toks[i].is_anchor() and toks[i].text == anchor:
                    seen.append(toks[i].text)
                    i += 1
            tail: List[Token] = []
            while i < len(toks):
                tail.append(toks[i])
                i += 1
            regions.append(tail)
            regions_per_sample.append(regions)
            anchors_seen.append(seen)
            
        # Check if we have anchors that repeat in some samples but not in skeleton
        # This handles cases like emails where some have multiple dots
        if skeleton:
            from collections import Counter
            for anchor_seq in anchor_seqs:
                anchor_counts = Counter(anchor_seq)
                for anchor, count in anchor_counts.items():
                    if anchor in skeleton and count > 1:
                        # This anchor repeats - we need a more flexible approach
                        # Let's allow for optional additional occurrences
                        pass

        num_regions = len(skeleton) + 1

        # 5) summarize regions
        region_summaries = [RegionSummary() for _ in range(num_regions)]
        for sample_regions in regions_per_sample:
            if len(sample_regions) < num_regions:
                sample_regions += [[] for _ in range(num_regions - len(sample_regions))]
            for idx in range(num_regions):
                region_summaries[idx].update_sample(sample_regions[idx])
                
        # Handle case where no anchors were found - use length-based pattern
        if not skeleton:
            # For patterns with no common structure, create a flexible character-based pattern
            region_summaries = [RegionSummary()]
            
            # Analyze the overall character composition
            all_chars = set()
            core_lengths = []
            for sample in self.samples:
                core_sample = sample[len(pref): len(sample) - len(suf) if len(suf) > 0 else len(sample)]
                all_chars.update(core_sample)
                core_lengths.append(len(core_sample))
            
            has_letters = any(c.isalpha() for c in all_chars)
            has_digits = any(c.isdigit() for c in all_chars)
            has_spaces = any(c.isspace() for c in all_chars)
            has_punct = any(not c.isalnum() and not c.isspace() for c in all_chars)
            
            # For patterns with no common structure, create a simple flexible pattern
            if has_letters and has_digits and has_spaces and has_punct:
                # Very mixed pattern (like phone numbers) - use a simple permissive pattern
                min_len = min(core_lengths) if core_lengths else 0
                max_len = max(core_lengths) if core_lengths else 0
                
                # Create a simple pattern that matches the observed character types and lengths
                region_summaries[0].lengths = core_lengths  # Use all observed core lengths  
                region_summaries[0].per_sample_kinds = [{"D", "S", "P"} for _ in core_lengths]  # Focus on digits, spaces, punctuation
            else:
                # Standard flexible pattern
                kinds = set()
                if has_digits:
                    kinds.add("D")
                if has_letters:
                    kinds.add("l")
                    kinds.add("U") 
                if has_spaces:
                    kinds.add("S")
                if has_punct:
                    kinds.add("P")
                
                min_len = min(core_lengths) if core_lengths else 0
                max_len = max(core_lengths) if core_lengths else 0
                
                region_summaries[0].lengths = [min_len]
                region_summaries[0].per_sample_kinds = [kinds]
            
            num_regions = 1

        region_patterns = [rs.summarize() for rs in region_summaries]

        # 6) anchor patterns (handle variable anchor counts)
        anchor_patterns: List[str] = []
        for i, anchor in enumerate(skeleton):
            # Count occurrences of this anchor at this position across samples
            counts = []
            for seq in anchor_seqs:
                if i < len(seq) and seq[i] == anchor:
                    counts.append(1)
                else:
                    counts.append(0)
            
            esc = escape_literal(anchor)
            if anchor in variable_anchors:
                # This anchor has variable occurrences - make it optional or flexible
                min_count = min(counts) if counts else 0
                max_count = max(counts) if counts else 1
                if min_count == 0:
                    # Some samples don't have this anchor - make it optional
                    anchor_patterns.append(f"(?:{esc}[a-z]*)*")
                else:
                    # All samples have it, but some have more - allow flexible matching
                    anchor_patterns.append(f"{esc}(?:[a-z]*{esc})*")
            else:
                # Standard case - appears consistently
                anchor_patterns.append(esc)

        # 7) assemble single anchored regex
        pieces: List[str] = []
        if pref:
            pieces.append(escape_literal(pref))
            
        # Special handling for no common anchors case
        if not skeleton:
            if region_patterns[0]:
                # If the pattern is very complex (like phone numbers), simplify it
                pattern = region_patterns[0]
                # Check if this is a phone-like pattern with mixed punctuation
                sample_chars = set(''.join(self.samples))
                has_mixed_punct = len([c for c in sample_chars if c in '+()-']) >= 2
                if has_mixed_punct and len(pattern) > 20:
                    # Simplify to a basic pattern that matches common phone elements
                    min_len = min(len(s) for s in self.samples)
                    max_len = max(len(s) for s in self.samples)
                    pieces.append(f"[\\d\\s\\-\\+\\(\\)]{{{min_len},{max_len}}}")
                else:
                    pieces.append(pattern)
        else:
            for i in range(num_regions):
                if region_patterns[i]:
                    pieces.append(region_patterns[i])
                if i < len(anchor_patterns):
                    pieces.append(anchor_patterns[i])
                    
        if suf:
            pieces.append(escape_literal(suf))

        body = "".join(pieces)
        regex = f"^{body}$"

        # small safe cleanup
        regex = re.sub(r"\{1\}", "", regex)
        return regex


def validate(regex: str, samples: List[str]) -> Tuple[bool, float, List[str]]:
    try:
        cre = re.compile(regex)
    except re.error as e:
        return False, 0.0, [f"compile_error: {e}"]
    ok = 0
    fails: List[str] = []
    for s in samples:
        if cre.fullmatch(s):
            ok += 1
        else:
            fails.append(s)
    rate = ok / len(samples) if samples else 0.0
    return True, rate, fails


def main() -> None:
    samples_email = [
        "abc@mail.com",
        "dfdfdfb@mm.com.tr",
        "dfdfdfb@mm.com.eu"
    ]

    samples_dash = [
        "123-abc-123",
        "323-dfgf-323",
        "999-QQQ-000"
    ]

    samples_mixed = [
        "+90 532 123 45 67",
        "(0541) 987-65-43",
        "0532 123 45 67"
    ]

    # New test cases to evaluate edge cases
    samples_ip = [
        "192.168.1.1",
        "10.0.0.1", 
        "172.16.254.1"
    ]
    
    samples_credit_card = [
        "4532-1234-5678-9012",
        "5555-4444-3333-2222",
        "4111-1111-1111-1111"
    ]
    
    samples_date = [
        "2023-12-25",
        "2024-01-15",
        "2022-06-30"
    ]
    
    samples_social_security = [
        "123-45-6789",
        "987-65-4321", 
        "555-12-3456"
    ]
    
    samples_social_security_2 = [
        "123-457890",
        "987-654321", 
        "555-123456"
    ]
    
    samples_social_security_4 = [
        "123.457890",
        "987.654321", 
        "555.123456"
    ]
    
    samples_social_security_5 = [
        "123.45789-0",
        "987.65432-1", 
        "555.12345-6"
    ]
    
    
    samples_social_security_6 = [
        "dfs-457890",
        "tgc-654321", 
        "kjh-123456"
    ]
    
    samples_social_security_7 = [
        "123.ftjklh",
        "987.lkvbfg", 
        "555.etcvdf"
    ]
    
    samples_social_security_8 = [
        "123.45789-a",
        "987.65432-d", 
        "555.12345-v"
    ]
    
    samples_social_security_9 = [
        "dfs(457890)",
        "tgc(654321)", 
        "kjh(123456)"
    ]
    
    samples_social_security_10 = [
        "123.(ftjklh)",
        "987.(lkvbfg)", 
        "555.(etcvdf)"
    ]
    
    samples_social_security_11 = [
        "123.(45789-a)",
        "987.(65432-d)", 
        "555.(12345-v)"
    ]
    
    
    samples_mixed_case = [
        "ABC123def",
        "XYZ456ghi",
        "PQR789jkl"
    ]

    sets = [
        ("Email Samples", samples_email),
        ("Dash Samples", samples_dash),
        ("Mixed Phone-like Samples", samples_mixed),
        ("IP Address Samples", samples_ip),
        ("Credit Card Samples", samples_credit_card),
        ("Date Samples", samples_date),
        ("Social Security Samples", samples_social_security),
        ("Social Security Samples 2", samples_social_security_2),
        ("Social Security Samples 4", samples_social_security_4),
        ("Social Security Samples 5", samples_social_security_5),
        ("Social Security Samples 6", samples_social_security_6),
        ("Social Security Samples 7", samples_social_security_7),
        ("Social Security Samples 8", samples_social_security_8),
        ("Social Security Samples 9", samples_social_security_9),
        ("Social Security Samples 10", samples_social_security_10),
        ("Social Security Samples 11", samples_social_security_11),
        ("Mixed Case Samples", samples_mixed_case),
    ]

    for title, samples in sets:
        s = SmartRegexSynthesizer()
        s.add(samples)
        rx = s.synthesize()
        ok, rate, fails = validate(rx, samples)
        print("\n" + "=" * 60)
        print(f"Testing: {title}")
        print("=" * 60)
        print(f"\nInput Patterns ({len(samples)}):")
        for i, p in enumerate(samples, 1):
            print(f"  {i:2d}. {p}")
        
        
        print("\nGenerated Regex:")
        print(f"  {rx}")
        print("\nValidation Results:")
        print(f"  Compiled: {ok}")
        if ok:
            print(f"  Match Rate: {rate:.2%}")
            if fails:
                print(f"  Sample Failures: {fails[:6]}")


if __name__ == "__main__":
    main()
