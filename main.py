#!/usr/bin/env python3
"""Generate simplified assessment PDFs (Gen2 pipeline).

Builds LaTeX, PDFs, metadata, and gold labels for configurable documents
using cached datasets (MMLU + MBPP). Questions are formatted explicitly as
MCQ, True/False, and Long-form prompts with consistent marks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import re
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_config  # type: ignore
from src.utils.logger import get_logger, setup_logging  # type: ignore
from src.data_processing.dataset_downloader import DatasetDownloader  # type: ignore
from src.pdf_generation.pdf_compiler import PDFCompiler  # type: ignore
from src.utils.llm_converter import LLMConverter  # type: ignore

MCQ_MARKS = 2
TF_MARKS = 2
LONG_MARKS = 10
LETTER_OPTIONS = ["A", "B", "C", "D", "E"]


@dataclass
class Question:
    qid: str
    qtype: str  # 'mcq', 'tf', 'long'
    prompt: str
    marks: int
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    source_id: Optional[int] = None
    source_dataset: Optional[str] = None
    image_path: Optional[str] = None


class DatasetPool:
    """Facilitates sampling across cached datasets with reproducible order."""

    def __init__(self, seed: Optional[int] = None, domain: Optional[str] = None, level: Optional[str] = None, config=None) -> None:
        # Create local Random instance instead of using global random (thread-safe)
        self.rng = random.Random(seed)
        self.config = config
        self.domain = domain
        self.level = level
        
        # Load all datasets
        self.mmlu_items = self._load_items("data/raw/mmlu_all.json")
        self.mbpp_items = self._load_items("data/raw/mbpp_plus.json")
        self.gsm8k_items = self._load_gsm8k_items("data/gsm_mcq")
        self.mmlu_pro_items = self._load_items("data/raw/mmlu_pro.json")
        self.ai2_arc_items = self._load_items("data/raw/ai2_arc.json")
        self.pubmedqa_items = self._load_items("data/raw/pubmedqa.json")
        self.squad_items = self._load_items("data/raw/squad.json")
        
        if not self.mmlu_items:
            raise RuntimeError("mmlu_all dataset missing or empty. Run without --skip-download once.")
        if not self.mbpp_items:
            raise RuntimeError("mbpp_plus dataset missing or empty. Run without --skip-download once.")
        if not self.gsm8k_items:
            raise RuntimeError("gsm8k dataset missing or empty. Check data/gsm_mcq directory.")
        if not self.mmlu_pro_items:
            raise RuntimeError("mmlu_pro dataset missing or empty. Run without --skip-download once.")
        if not self.ai2_arc_items:
            raise RuntimeError("ai2_arc dataset missing or empty. Run without --skip-download once.")
        # PubMedQA is optional - only needed for biology graduate TF questions
        # SQuAD is optional - only needed for domains configured to use it
        
        # Filter MMLU and MMLU-Pro items by domain and level if specified
        if domain and config:
            self.mmlu_items = self._filter_mmlu_by_domain_level(self.mmlu_items, domain, level)
            self.mmlu_pro_items = self._filter_mmlu_by_domain_level(self.mmlu_pro_items, domain, level)
            
            # Validate filtered lists are not empty
            logger = get_logger()
            if not self.mmlu_pro_items:
                subjects = config.get_subjects_for_domain_level(domain, level) if level else []
                logger.warning(
                    f"MMLU-Pro dataset filtered to empty list for domain '{domain}', level '{level}'. "
                    f"Configured subjects: {subjects}. "
                    f"Check if subjects match MMLU-Pro dataset items or configure additional subjects."
                )
            if not self.mmlu_items:
                subjects = config.get_subjects_for_domain_level(domain, level) if level else []
                logger.warning(
                    f"MMLU dataset filtered to empty list for domain '{domain}', level '{level}'. "
                    f"Configured subjects: {subjects}."
                )
            
            # Filter SQuAD items by domain if available
            if self.squad_items:
                self.squad_items = self._filter_squad_by_domain(self.squad_items, domain, level)
        
        self.rng.shuffle(self.mmlu_items)
        self.rng.shuffle(self.mbpp_items)
        self.rng.shuffle(self.gsm8k_items)
        self.rng.shuffle(self.mmlu_pro_items)
        self.rng.shuffle(self.ai2_arc_items)
        if self.pubmedqa_items:
            self.rng.shuffle(self.pubmedqa_items)
        if self.squad_items:
            self.rng.shuffle(self.squad_items)
        self._mmlu_idx = 0
        self._mbpp_idx = 0
        self._gsm8k_idx = 0
        self._mmlu_pro_idx = 0
        self._arc_idx = 0
        self._pubmedqa_idx = 0
        self._squad_idx = 0

    @staticmethod
    def _load_items(path: str) -> List[Dict[str, Any]]:
        file_path = Path(path)
        if not file_path.exists():
            return []
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("items", [])
    
    @staticmethod
    def _load_gsm8k_items(gsm8k_dir: str) -> List[Dict[str, Any]]:
        """Load GSM8K items from JSONL files."""
        gsm8k_path = Path(gsm8k_dir)
        if not gsm8k_path.exists():
            return []
        
        items = []
        for jsonl_file in gsm8k_path.glob("*.jsonl"):
            with jsonl_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        data = json.loads(line)
                        # Convert GSM8K format to our format
                        item = {
                            "id": len(items),
                            "problem": data.get("problem", ""),
                            "solution": data.get("solution", ""),
                            "candidates": data.get("candidates", []),
                            "level": data.get("level", "unknown"),
                            "type": data.get("type", "math"),
                            "source_file": str(jsonl_file)
                        }
                        items.append(item)
        return items
    
    def _filter_mmlu_by_domain_level(self, mmlu_items: List[Dict[str, Any]], domain: str, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter MMLU items by domain-specific subjects and academic level."""
        if not self.config:
            return mmlu_items
        
        logger = get_logger()
        original_count = len(mmlu_items)
        
        # Get subjects for this domain and level
        if level:
            domain_subjects = self.config.get_subjects_for_domain_level(domain, level)
        else:
            # Fallback to domain-only filtering (no level specified)
            # When no level is specified, return empty list to indicate no filtering
            domain_subjects = []
        
        if not domain_subjects:
            logger.debug(f"No subjects configured for domain '{domain}', level '{level}'. Returning all {original_count} items.")
            return mmlu_items
        
        # Filter items that match the domain subjects
        filtered_items = []
        for item in mmlu_items:
            subject = item.get("subject", "")
            if subject in domain_subjects:
                filtered_items.append(item)
        
        logger.info(
            f"Filtered MMLU items for domain '{domain}', level '{level}': "
            f"{len(filtered_items)} out of {original_count} items matched subjects {domain_subjects}"
        )
        
        if len(filtered_items) == 0:
            logger.warning(
                f"No MMLU items matched subjects {domain_subjects} for domain '{domain}', level '{level}'. "
                f"Original dataset had {original_count} items."
            )
        
        return filtered_items
    
    def _filter_squad_by_domain(self, squad_items: List[Dict[str, Any]], domain: str, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter SQuAD items by domain using Wikipedia article titles and keywords."""
        if not self.config:
            return squad_items
        
        # Get SQuAD keywords for this domain from config
        try:
            logger = get_logger()
            keywords_dict = self.config.get_squad_keywords_for_domain(domain)
            if not keywords_dict:
                # No keywords configured for this domain, return all items
                logger.info(f"No SQuAD keywords configured for domain '{domain}'. Returning all {len(squad_items)} SQuAD items.")
                return squad_items
            
            title_keywords = [kw.lower() for kw in keywords_dict.get("title_keywords", [])]
            context_keywords = [kw.lower() for kw in keywords_dict.get("context_keywords", [])]
            
            if not title_keywords and not context_keywords:
                logger.info(f"SQuAD keywords for domain '{domain}' are empty. Returning all {len(squad_items)} SQuAD items.")
                return squad_items
            
            # Filter items that match domain keywords
            filtered_items = []
            for item in squad_items:
                title = item.get("title", "").lower()
                context = item.get("context", "").lower()
                question = item.get("question", "").lower()
                
                # Check if any keyword matches the title, context, or question
                matches_title = any(keyword in title for keyword in title_keywords) if title_keywords else False
                matches_context = any(keyword in context or keyword in question for keyword in context_keywords) if context_keywords else False
                
                if matches_title or matches_context:
                    filtered_items.append(item)
            
            logger.info(f"Filtered SQuAD items for domain '{domain}': {len(filtered_items)} out of {len(squad_items)} original items.")
            if len(filtered_items) == 0:
                logger.warning(f"No SQuAD items matched keywords for domain '{domain}'. This may cause errors when generating questions.")
            
            return filtered_items
        except Exception as e:
            # If there's an error getting keywords, return all items
            # Log warning if possible, but don't fail
            try:
                logger = get_logger()
                logger.warning(f"Error filtering SQuAD by domain '{domain}': {e}. Returning all items.")
            except:
                pass
            return squad_items
    
    def next_mmlu(self) -> Dict[str, Any]:
        logger = get_logger()
        # Check if list is empty before any access (defensive programming)
        if not self.mmlu_items:
            logger.error(f"MMLU dataset is empty for domain '{self.domain}', level '{self.level}'")
            raise RuntimeError(
                f"MMLU dataset is empty for domain '{self.domain}', level '{self.level}'. "
                f"Check if subjects are configured in config.yaml for this domain-level combination, "
                f"or if the MMLU dataset contains items matching those subjects."
            )
        
        if self._mmlu_idx >= len(self.mmlu_items):
            logger.info(f"MMLU pool exhausted at index {self._mmlu_idx}/{len(self.mmlu_items)}, resetting and reshuffling")
            # Reset index to cycle through questions again
            self._mmlu_idx = 0
            # Reshuffle for variety
            self.rng.shuffle(self.mmlu_items)
            # Double-check after reset (shouldn't be empty, but defensive programming)
            if not self.mmlu_items:
                logger.error(f"MMLU dataset became empty after reset for domain '{self.domain}', level '{self.level}'")
                raise RuntimeError(
                    f"MMLU dataset is empty for domain '{self.domain}', level '{self.level}'"
                )
        
        item = self.mmlu_items[self._mmlu_idx]
        logger.debug(f"Extracted MMLU item {self._mmlu_idx+1}/{len(self.mmlu_items)}: id={item.get('id', 'unknown')}, subject={item.get('subject', 'unknown')}")
        self._mmlu_idx += 1
        return item

    def next_mbpp(self) -> Dict[str, Any]:
        logger = get_logger()
        if self._mbpp_idx >= len(self.mbpp_items):
            logger.error(f"MBPP pool exhausted at index {self._mbpp_idx}/{len(self.mbpp_items)}")
            raise RuntimeError("Ran out of MBPP prompts")
        item = self.mbpp_items[self._mbpp_idx]
        logger.debug(f"Extracted MBPP item {self._mbpp_idx+1}/{len(self.mbpp_items)}: id={item.get('id', 'unknown')}")
        self._mbpp_idx += 1
        return item
    
    def next_gsm8k(self) -> Dict[str, Any]:
        logger = get_logger()
        if self._gsm8k_idx >= len(self.gsm8k_items):
            logger.error(f"GSM8K pool exhausted at index {self._gsm8k_idx}/{len(self.gsm8k_items)}")
            raise RuntimeError("Ran out of GSM8K problems")
        item = self.gsm8k_items[self._gsm8k_idx]
        logger.debug(f"Extracted GSM8K item {self._gsm8k_idx+1}/{len(self.gsm8k_items)}: id={item.get('id', 'unknown')}")
        self._gsm8k_idx += 1
        return item

    def next_mmlu_pro(self) -> Dict[str, Any]:
        logger = get_logger()
        # Check if list is empty before any access
        if not self.mmlu_pro_items:
            logger.error(f"MMLU-Pro dataset is empty for domain '{self.domain}', level '{self.level}'")
            raise RuntimeError(
                f"MMLU-Pro dataset is empty for domain '{self.domain}', level '{self.level}'. "
                f"Check if subjects are configured in config.yaml for this domain-level combination, "
                f"or if the MMLU-Pro dataset contains items matching those subjects."
            )
        
        if self._mmlu_pro_idx >= len(self.mmlu_pro_items):
            logger.info(f"MMLU-Pro pool exhausted at index {self._mmlu_pro_idx}/{len(self.mmlu_pro_items)}, resetting and reshuffling")
            # Reset index to cycle through questions again
            self._mmlu_pro_idx = 0
            # Reshuffle for variety
            self.rng.shuffle(self.mmlu_pro_items)
            # Double-check after reset (shouldn't be empty, but defensive programming)
            if not self.mmlu_pro_items:
                logger.error(f"MMLU-Pro dataset became empty after reset for domain '{self.domain}', level '{self.level}'")
                raise RuntimeError(
                    f"MMLU-Pro dataset is empty for domain '{self.domain}', level '{self.level}'"
                )
        
        item = self.mmlu_pro_items[self._mmlu_pro_idx]
        logger.debug(f"Extracted MMLU-Pro item {self._mmlu_pro_idx+1}/{len(self.mmlu_pro_items)}: id={item.get('id', 'unknown')}, subject={item.get('subject', 'unknown')}")
        self._mmlu_pro_idx += 1
        return item

    def next_arc(self) -> Dict[str, Any]:
        logger = get_logger()
        # Check if list is empty before any access
        if not self.ai2_arc_items:
            logger.error(f"AI2-ARC dataset is empty for domain '{self.domain}', level '{self.level}'")
            raise RuntimeError(
                f"AI2-ARC dataset is empty for domain '{self.domain}', level '{self.level}'. "
                f"Check if subjects are configured in config.yaml for this domain-level combination, "
                f"or if the AI2-ARC dataset contains items matching those subjects."
            )
        
        if self._arc_idx >= len(self.ai2_arc_items):
            logger.info(f"AI2-ARC pool exhausted at index {self._arc_idx}/{len(self.ai2_arc_items)}, resetting and reshuffling")
            # Reset index to cycle through questions again
            self._arc_idx = 0
            # Reshuffle for variety
            self.rng.shuffle(self.ai2_arc_items)
            # Double-check after reset (shouldn't be empty, but defensive programming)
            if not self.ai2_arc_items:
                logger.error(f"AI2-ARC dataset became empty after reset for domain '{self.domain}', level '{self.level}'")
                raise RuntimeError(
                    f"AI2-ARC dataset is empty for domain '{self.domain}', level '{self.level}'"
                )
        
        item = self.ai2_arc_items[self._arc_idx]
        logger.debug(f"Extracted AI2-ARC item {self._arc_idx+1}/{len(self.ai2_arc_items)}: id={item.get('id', 'unknown')}, subject={item.get('subject', 'unknown')}")
        self._arc_idx += 1
        return item

    def next_pubmedqa(self) -> Dict[str, Any]:
        logger = get_logger()
        if not self.pubmedqa_items:
            logger.error(f"PubMedQA dataset missing or empty for domain '{self.domain}', level '{self.level}'")
            raise RuntimeError(
                f"PubMedQA dataset missing or empty for domain '{self.domain}', level '{self.level}'. "
                f"Run without --skip-download once."
            )
        if self._pubmedqa_idx >= len(self.pubmedqa_items):
            logger.info(f"PubMedQA pool exhausted at index {self._pubmedqa_idx}/{len(self.pubmedqa_items)}, resetting and reshuffling")
            # Reset index to cycle through questions again
            self._pubmedqa_idx = 0
            # Reshuffle for variety
            self.rng.shuffle(self.pubmedqa_items)
            # Double-check after reset (shouldn't be empty, but defensive programming)
            if not self.pubmedqa_items:
                logger.error(f"PubMedQA dataset became empty after reset for domain '{self.domain}', level '{self.level}'")
                raise RuntimeError(
                    f"PubMedQA dataset is empty for domain '{self.domain}', level '{self.level}'"
                )
        item = self.pubmedqa_items[self._pubmedqa_idx]
        logger.debug(f"Extracted PubMedQA item {self._pubmedqa_idx+1}/{len(self.pubmedqa_items)}: id={item.get('id', 'unknown')}")
        self._pubmedqa_idx += 1
        return item
    
    def next_squad(self) -> Dict[str, Any]:
        logger = get_logger()
        if not self.squad_items:
            logger.error(f"SQuAD dataset missing or empty for domain '{self.domain}'")
            raise RuntimeError(f"SQuAD dataset missing or empty for domain '{self.domain}'. Run without --skip-download once, or check if SQuAD keywords are too restrictive.")
        if self._squad_idx >= len(self.squad_items):
            logger.info(f"SQuAD pool exhausted at index {self._squad_idx}, resetting and reshuffling for domain '{self.domain}'")
            # Reset index to cycle through questions again
            self._squad_idx = 0
            # Reshuffle for variety
            self.rng.shuffle(self.squad_items)
        if self._squad_idx >= len(self.squad_items):
            logger.error(f"SQuAD dataset exhausted for domain '{self.domain}'. Only {len(self.squad_items)} items available after filtering.")
            raise RuntimeError(f"SQuAD dataset exhausted for domain '{self.domain}'. Only {len(self.squad_items)} items available after filtering. Consider expanding keywords or reducing question count.")
        item = self.squad_items[self._squad_idx]
        logger.debug(f"Extracted SQuAD item {self._squad_idx+1}/{len(self.squad_items)} for domain '{self.domain}': id={item.get('id', 'unknown')}, title={item.get('title', 'unknown')[:50]}")
        self._squad_idx += 1
        return item

def escape_latex(text: str) -> str:
    """Escape LaTeX special characters, but preserve mathematical expressions."""
    import re

    if text is None:
        return ""

    text = str(text)

    # Protect display math blocks first (\[ ... \] and \( ... \))
    display_blocks: List[Tuple[str, str]] = []

    def protect_display(pattern: str, prefix: str) -> None:
        nonlocal text

        def _replace(match: re.Match) -> str:
            idx = len(display_blocks)
            placeholder = f"@@{prefix}{idx}@@"
            display_blocks.append((placeholder, match.group(0)))
            return placeholder

        text = re.sub(pattern, _replace, text, flags=re.DOTALL)

    protect_display(r'\\\[(.+?)\\\]', "DSPMATH")
    protect_display(r'\\\((.+?)\\\)', "INLMATH")

    # Protect inline math $...$
    math_expressions: List[str] = []
    math_placeholders: List[str] = []

    def replace_math(match: re.Match) -> str:
        idx = len(math_expressions)
        math_expressions.append(match.group(1))
        placeholder = f"@@MATHPH{idx}@@"
        math_placeholders.append(placeholder)
        return placeholder

    text = re.sub(r'\$([^$]+)\$', replace_math, text)

    # Protect bare LaTeX command sequences like \frac{...}{...}
    latex_commands: List[str] = []
    command_placeholders: List[str] = []

    def replace_command(match: re.Match) -> str:
        idx = len(latex_commands)
        command = match.group(0)
        placeholder = f"@@CMDPH{idx}@@"
        latex_commands.append(command)
        command_placeholders.append(placeholder)
        return placeholder

    # Capture commands with optional brace arguments
    text = re.sub(r'(\\[a-zA-Z]+(?:\s*\{[^{}]*\})*)', replace_command, text)

    # NEW: Detect and convert chemical formulas to LaTeX subscripts BEFORE escaping
    # Pattern matches: letter(s) followed by digit(s) - e.g., H2, SO4, H2SO4, C27H46O
    # Also handles spaced versions like "H 2 SO 4" -> "H2SO4" -> subscripts
    chemical_formulas: List[str] = []
    formula_placeholders: List[str] = []

    def replace_chemical_formula(match: re.Match) -> str:
        idx = len(chemical_formulas)
        # Get the full match (e.g., "H2SO4" or "H 2 SO 4")
        formula = match.group(0)
        # Remove spaces to normalize (e.g., "H 2 SO 4" -> "H2SO4")
        formula_clean = re.sub(r'\s+', '', formula)
        
        # Convert to LaTeX subscript format: H2SO4 -> H$_2$SO$_4$
        # Pattern: find letter(s) followed by digit(s) and wrap digits in subscript
        def subscript_replace(m: re.Match) -> str:
            element = m.group(1)  # Letters (element symbol)
            number = m.group(2)   # Digits (subscript)
            return f"{element}$_{{{number}}}$"
        
        # Convert each element+number pair to subscript notation
        formula_latex = re.sub(r'([A-Z][a-z]?)(\d+)', subscript_replace, formula_clean)
        chemical_formulas.append(formula_latex)
        placeholder = f"@@CHEMPH{idx}@@"
        formula_placeholders.append(placeholder)
        return placeholder

    # Match chemical formula patterns:
    # - One or more letters (element symbols) followed by one or more digits
    # - Can have spaces between element and number (e.g., "H 2 SO 4")
    # - Pattern: letter(s) + optional space + digit(s), repeated
    # - Must start with a capital letter (chemical element)
    # - Exclude if already inside a placeholder
    text = re.sub(
        r'\b([A-Z][a-z]?(?:\s*\d+)+)(?:\s*[A-Z][a-z]?(?:\s*\d+)*)*', 
        replace_chemical_formula, 
        text
    )

    # Detect and protect exponent patterns (e.g., t^2, x^3, 2^t) BEFORE escaping
    # Pattern matches: variable/number^exponent (not already in math mode)
    exponent_expressions: List[str] = []
    exponent_placeholders: List[str] = []

    def replace_exponent(match: re.Match) -> str:
        idx = len(exponent_expressions)
        base = match.group(1)
        exp = match.group(2)
        # Wrap in math mode
        math_expr = f"${base}^{exp}$"
        exponent_expressions.append(math_expr)
        placeholder = f"@@EXPPH{idx}@@"
        exponent_placeholders.append(placeholder)
        return placeholder

    # Match exponent patterns: alphanumeric/parens^alphanumeric/parens
    # Exclude if already inside a placeholder or math mode
    text = re.sub(r'([a-zA-Z0-9\)]+)\^([a-zA-Z0-9\(]+)', replace_exponent, text)

    # Escape the remaining text (NOTE: ^ is NOT in replacements - handled above)
    replacements = {
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "~": r"\textasciitilde{}",
        "%": r"\%",
        "<": r"\textless{}",
        ">": r"\textgreater{}",
    }

    for char, repl in replacements.items():
        text = text.replace(char, repl)

    # Restore placeholders in reverse order of capture, wrapping math commands
    TEXT_PREFIXES = ("\\text", "\\begin", "\\end", "\\label", "\\ref", "\\item", "\\ensuremath")

    for placeholder, command in zip(command_placeholders, latex_commands):
        command_name_match = re.match(r'\\[a-zA-Z]+', command)
        command_name = command_name_match.group(0) if command_name_match else ""
        if any(command_name.startswith(prefix) for prefix in TEXT_PREFIXES):
            restored = command
        else:
            restored = rf"\ensuremath{{{command}}}"
        text = text.replace(placeholder, restored)

    # Restore chemical formulas (already in LaTeX subscript format)
    for placeholder, formula in zip(formula_placeholders, chemical_formulas):
        text = text.replace(placeholder, formula)

    # Restore exponent expressions (already wrapped in $...$)
    for placeholder, exp_expr in zip(exponent_placeholders, exponent_expressions):
        text = text.replace(placeholder, exp_expr)

    for placeholder, math_expr in zip(math_placeholders, math_expressions):
        text = text.replace(placeholder, f"${math_expr}$")

    # Restore display math blocks
    for placeholder, block in display_blocks:
        text = text.replace(placeholder, block)

    # Clean up accidental literal-dollar + math delimiter sequences
    text = text.replace(r"\$$", "$")

    return text


UNICODE_REPLACEMENTS = {
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    " ": " "
}

def normalize_whitespace(text: str) -> str:
    for bad, repl in UNICODE_REPLACEMENTS.items():
        text = text.replace(bad, repl)
    return re.sub(r"\s+", " ", text).strip()


def fix_missing_spaces(text: str) -> str:
    """Fix missing spaces around numbers, currency, and common patterns."""
    if not text:
        return text
    
    # Fix missing space after numbers followed by letters (e.g., "60,000insurance" -> "60,000 insurance")
    text = re.sub(r'(\d+(?:,\d{3})*(?:\.\d+)?)([a-zA-Z])', r'\1 \2', text)
    
    # Fix missing space before numbers followed by letters in certain contexts
    # (e.g., "policyatarateof 0.2065" -> "policy at a rate of 0.2065")
    text = re.sub(r'([a-zA-Z])(\d+(?:,\d{3})*(?:\.\d+)?)', r'\1 \2', text)
    
    # Fix common concatenated patterns
    text = re.sub(r'atarateof', 'at a rate of', text, flags=re.IGNORECASE)
    text = re.sub(r'yearendowment', 'year endowment', text, flags=re.IGNORECASE)
    text = re.sub(r'policywitha', 'policy with a', text, flags=re.IGNORECASE)
    text = re.sub(r'anddiscussthe', 'and discuss the', text, flags=re.IGNORECASE)
    
    # Normalize whitespace after fixes
    return normalize_whitespace(text)


def sanitize_text(text: str) -> str:
    """Remove control characters and normalize Unicode math symbols."""
    if not text:
        return text
    
    text = str(text)
    
    # Remove control characters (ord < 32, except \n, \t, \r)
    text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\t\r')
    
    # Normalize Unicode math symbols to LaTeX equivalents
    unicode_math_replacements = {
        '√': r'\sqrt',
        '²': '^2',
        '³': '^3',
        '°': r'^\circ',
        '×': r'\times',
        '÷': r'\div',
        '±': r'\pm',
        '≤': r'\leq',
        '≥': r'\geq',
        '≠': r'\neq',
        '≈': r'\approx',
    }
    
    for unicode_char, latex_equiv in unicode_math_replacements.items():
        text = text.replace(unicode_char, latex_equiv)
    
    return text


def clean_option_text(text: str) -> str:
    """Remove leading numeric list markers (e.g., '4. text', '(4) text') but keep math content like '4/9' or '48,000'."""
    if not text:
        return text
    
    text = str(text)
    # Remove numbered prefixes: optional bracket/paren + digits + period/paren/bracket + whitespace
    # This now handles both "4. Answer" and "4. 22.1 MeV" patterns
    # Still preserves "4/9", "48,000" (no punctuation+space pattern)
    text = re.sub(r'^[\s]*[\(\[]?\d+[\.\)\]]\s+', '', text)
    return text.strip()


def clean_question_text(text: str) -> str:
    """Remove leading numeric list markers from question text (e.g., '4. ', '(4) ', '[4] ')."""
    if not text:
        return text
    
    text = str(text)
    # Remove patterns like "4. ", "(4) ", "[4] " at the start
    # More permissive than clean_option_text - removes any numeric prefix followed by punctuation and space
    text = re.sub(r'^[\s]*[\(\[]?\d+[\.\)\]]\s+', '', text)
    return text.strip()


def validate_domain_match(item: Dict[str, Any], domain: str, pool: Optional[DatasetPool] = None, config=None) -> bool:
    """Validate that an MMLU item matches the expected domain."""
    if not config:
        return True  # No config means no filtering
    
    # Get expected subjects for this domain
    level = getattr(pool, 'level', None) if pool else None
    expected_subjects = config.get_subjects_for_domain_level(domain, level) if level else []
    
    if not expected_subjects:
        return True  # No subjects configured means accept all
    
    item_subject = item.get("subject", "").lower()
    domain_lower = domain.lower()
    
    # Check if subject matches expected subjects
    if item_subject in [s.lower() for s in expected_subjects]:
        return True
    
    # Additional check: reject obvious mismatches
    math_keywords = ["math", "mathematics", "algebra", "calculus", "geometry", "statistics"]
    chemistry_keywords = ["chemistry", "chemical", "molecular", "atomic"]
    
    if domain_lower == "chemistry":
        # Reject if subject contains math keywords
        if any(kw in item_subject for kw in math_keywords):
            return False
    
    if domain_lower == "mathematics":
        # Reject if subject contains chemistry keywords (unless it's math chemistry)
        if any(kw in item_subject for kw in chemistry_keywords) and "math" not in item_subject:
            return False
    
    return True  # Default: accept if no obvious mismatch


def has_only_ascii(text: str) -> bool:
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def get_fixed_question_distribution() -> Dict[str, int]:
    """
    Return fixed question distribution for all papers.
    
    All generated papers use the same distribution:
    - 5 MCQ questions (5 * 2 = 10 marks)
    - 5 True/False questions (5 * 2 = 10 marks)
    - 2 Long-form questions (2 * 10 = 20 marks)
    Total: 40 marks
    
    Returns:
        Dictionary with fixed counts: {"mcq": 5, "tf": 5, "long": 2}
    """
    return {"mcq": 5, "tf": 5, "long": 2}


def question_counts(combination: Iterable[str]) -> Dict[str, int]:
    """
    Return question counts for a combination.
    
    Note: This function now always returns the fixed distribution
    (5 MCQ, 5 TF, 2 Long-form) regardless of the combination.
    The combination parameter is kept for backward compatibility.
    """
    return get_fixed_question_distribution()


def parse_answer_index(answer: Any, choices_len: int) -> int:
    logger = get_logger()
    original_answer = answer
    if isinstance(answer, str) and answer.upper() in LETTER_OPTIONS:
        idx = LETTER_OPTIONS.index(answer.upper())
        logger.debug(f"Parsed answer index from letter '{answer.upper()}' -> {idx}")
    else:
        try:
            idx = int(answer)
            logger.debug(f"Parsed answer index from integer '{answer}' -> {idx}")
        except Exception:
            idx = 0
            logger.debug(f"Failed to parse answer '{answer}' as integer, defaulting to 0")
    if choices_len == 0:
        logger.warning(f"Answer index parsing: choices_len is 0, returning 0")
        return 0
    result = max(0, min(idx, choices_len - 1))
    if result != idx:
        logger.debug(f"Answer index adjusted: {original_answer} -> {idx} -> clamped to {result} (choices_len: {choices_len})")
    return result


def build_mcq(pool: DatasetPool, total: int, doc_id: str) -> List[Question]:
    logger = get_logger()
    logger.info(f"Building {total} MCQ questions, doc_id: {doc_id}")
    questions: List[Question] = []
    
    # Initialize LLM converter for generating explanations
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
    except Exception as e:
        logger.warning(f"LLM converter unavailable for MCQ explanations: {e}. Using fallback.")
        llm_converter = None
    
    for idx in range(total):
        logger.debug(f"MCQ {idx+1}/{total}: Starting question extraction")
        attempts = 0
        item = None
        processed_options = None
        while attempts < 500:
            candidate = pool.next_mmlu()
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices_raw = candidate.get("choices", [])[: len(LETTER_OPTIONS)]
            
            if question_raw and choices_raw and all(normalize_whitespace(c) for c in choices_raw):
                trimmed_question = normalize_whitespace(question_raw)
                trimmed_question = trimmed_question.rstrip('.')
                
                # Process options INSIDE the loop to validate them
                processed_choices = [normalize_whitespace(c) for c in choices_raw]
                options = [clean_option_text(fix_missing_spaces(sanitize_text(choice))) for choice in processed_choices]
                options = [opt for opt in options if opt and opt.strip()]  # Filter empty options
                
                # Validate question AND processed options
                if (len(trimmed_question) <= 180
                        and has_only_ascii(trimmed_question)
                        and len(options) >= 2  # Need at least 2 valid options
                        and all(len(opt) <= 90 and has_only_ascii(opt) for opt in options)):
                    item = candidate
                    processed_options = options  # Store processed options
                    break
            attempts += 1
        
        if item is None:
            logger.error(f"MCQ {idx+1}/{total}: Unable to find suitable MCQ item from MMLU dataset after {attempts} attempts")
            raise RuntimeError("Unable to find suitable MCQ item from MMLU dataset")
        
        logger.debug(f"MCQ {idx+1}/{total}: Selected item id={item.get('id', 'unknown')}, attempts={attempts}")
        question_raw = normalize_whitespace(item.get("question", ""))
        logger.debug(f"MCQ {idx+1}/{total}: Raw question length: {len(question_raw)}")
        question_text = textwrap.fill(fix_missing_spaces(sanitize_text(question_raw)), width=90)
        
        # Use stored processed_options instead of processing again
        options = processed_options
        logger.debug(f"MCQ {idx+1}/{total}: Using {len(options)} valid processed options")
        
        answer_idx = parse_answer_index(item.get("answer"), len(options))
        logger.debug(f"MCQ {idx+1}/{total}: Parsed answer index: {answer_idx} from answer: {item.get('answer')}, options count: {len(options)}")
        correct_letter = LETTER_OPTIONS[answer_idx] if options and answer_idx < len(LETTER_OPTIONS) else "A"
        correct_text = options[answer_idx] if options and answer_idx < len(options) else ""
        logger.debug(f"MCQ {idx+1}/{total}: Selected answer - letter: {correct_letter}, text: {correct_text[:50] if correct_text else 'empty'}...")
        formatted_options = options
        subject = item.get("subject") or "mmlu_all"
        
        # Generate explanation using LLM
        explanation = None
        if llm_converter:
            try:
                explanation = llm_converter.generate_answer_explanation(
                    question_text=question_text,
                    answer=correct_text,
                    question_type="MCQ",
                    domain=subject,
                    academic_level="K-12"
                )
            except Exception as e:
                logger.warning(f"Error generating MCQ explanation: {e}")
        # Fallback if LLM fails
        if not explanation:
            explanation = f"Correct option: {correct_letter} - {correct_text}"
        
        questions.append(
            Question(
                qid=f"{doc_id}_mcq_{idx+1}",
                qtype="mcq",
                prompt=question_text or "Answer the question.",
                marks=MCQ_MARKS,
                options=formatted_options,
                correct_answer=correct_letter,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=subject,
            )
        )
        logger.debug(f"MCQ {idx+1}/{total}: Created question qid={questions[-1].qid}")
    logger.info(f"Successfully built {len(questions)}/{total} MCQ questions")
    return questions


def build_tf(pool: DatasetPool, total: int, doc_id: str) -> List[Question]:
    questions: List[Question] = []
    logger = get_logger()
    
    # Initialize LLM converter (with fallback if unavailable)
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
    except ValueError as e:
        logger.warning(f"LLM converter unavailable for TF generation: {e}. Using fallback method.")
        llm_converter = None
    except Exception as e:
        logger.warning(f"Error initializing LLM converter for TF: {e}. Using fallback method.")
        llm_converter = None
    
    # Get domain and level from pool if available
    domain = getattr(pool, 'domain', None) or "general"
    academic_level = getattr(pool, 'level', None) or "K-12"
    
    logger.info(f"Building {total} TF questions, doc_id: {doc_id}, domain: {domain}, level: {academic_level}")
    for idx in range(total):
        logger.debug(f"TF {idx+1}/{total}: Starting question extraction")
        attempts = 0
        item = None
        while attempts < 500:
            candidate = pool.next_mmlu()
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices = candidate.get("choices", [])
            if question_raw and choices:
                trimmed_question = normalize_whitespace(question_raw)
                if (len(trimmed_question) <= 160
                        and has_only_ascii(trimmed_question)
                        and all(has_only_ascii(normalize_whitespace(c)) for c in choices)):
                    item = candidate
                    break
            attempts += 1
        if item is None:
            logger.error(f"TF {idx+1}/{total}: Unable to find suitable TF item from MMLU dataset after {attempts} attempts")
            raise RuntimeError("Unable to find suitable TF item from MMLU dataset")
        logger.debug(f"TF {idx+1}/{total}: Selected item id={item.get('id', 'unknown')}, attempts={attempts}")
        question_text = normalize_whitespace(item.get("question", ""))
        logger.debug(f"TF {idx+1}/{total}: Raw question length: {len(question_text)}")
        choices = [normalize_whitespace(c) for c in item.get("choices", [])]
        logger.debug(f"TF {idx+1}/{total}: Raw choices count: {len(choices)}")
        
        # Ensure choices list is not empty - retry if empty
        if not choices:
            # Retry with a new item
            attempts = 0
            while attempts < 100:
                candidate = pool.next_mmlu()
                choices = candidate.get("choices", [])
                if choices:
                    item = candidate
                    question_text = normalize_whitespace(item.get("question", ""))
                    choices = [normalize_whitespace(c) for c in choices]
                    break
                attempts += 1
            if not choices:
                raise RuntimeError("Unable to find suitable TF item with valid choices from MMLU dataset")
        
        answer_idx = parse_answer_index(item.get("answer"), len(choices))
        
        # Ensure answer_idx is within bounds (should never happen due to parse_answer_index, but double-check)
        if answer_idx < 0 or answer_idx >= len(choices):
            # This should never happen if parse_answer_index works correctly, but if it does, clamp it
            answer_idx = max(0, min(answer_idx, len(choices) - 1))
        
        # Double-check that answer_idx is valid before accessing
        if answer_idx >= len(choices) or answer_idx < 0:
            # Retry with a new item
            attempts = 0
            while attempts < 100:
                candidate = pool.next_mmlu()
                choices = candidate.get("choices", [])
                if choices and len(choices) > 0:
                    item = candidate
                    question_text = normalize_whitespace(item.get("question", ""))
                    choices = [normalize_whitespace(c) for c in choices]
                    answer_idx = parse_answer_index(item.get("answer"), len(choices))
                    if 0 <= answer_idx < len(choices):
                        break
                attempts += 1
            if answer_idx >= len(choices) or answer_idx < 0:
                raise RuntimeError("Unable to find suitable TF item with valid answer index from MMLU dataset")
        
        correct_text = choices[answer_idx]
        
        # Try LLM conversion first
        llm_statements = None
        if llm_converter:
            try:
                llm_statements = llm_converter.convert_mcq_to_tf(
                    mcq_stem=question_text,
                    choices=choices,
                    correct_answer=correct_text,
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.warning(f"Error during LLM TF conversion: {e}. Using fallback method.")
        
        # Use LLM-generated statements if available, otherwise use fallback
        if llm_statements:
            # Randomly choose between true and false statement
            make_true = random.random() < 0.5
            if make_true:
                statement = llm_statements["true_statement"]
                correct_answer = "True"
            else:
                statement = llm_statements["false_statement"]
                correct_answer = "False"
        else:
            # Fallback to original method
            false_text = None
            if choices and len(choices) > 1:
                incorrect = [c for i, c in enumerate(choices) if i != answer_idx and i < len(choices)]
                if incorrect:
                    false_text = random.choice(incorrect)
            make_true = random.random() < 0.5 or not false_text
            question_text_short = textwrap.shorten(question_text, width=140, placeholder="...")
            if make_true:
                statement = f"The correct answer to '{question_text_short}' is '{correct_text}'."
                correct_answer = "True"
            else:
                statement = f"The correct answer to '{question_text_short}' is '{false_text}'."
                correct_answer = "False"
        
        # Generate explanation using LLM
        explanation = None
        if llm_converter:
            try:
                explanation = llm_converter.generate_answer_explanation(
                    question_text=statement,
                    answer=correct_answer,
                    question_type="TF",
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.warning(f"Error generating TF explanation: {e}")
        # Fallback if LLM fails
        if not explanation:
            if correct_answer == "True":
                explanation = f"This statement is true based on the correct answer: '{correct_text}'."
            else:
                explanation = f"This statement is false. The correct answer is: '{correct_text}'."
        
        subject = item.get("subject") or "mmlu_all"
        logger.debug(f"TF {idx+1}/{total}: Statement: {statement[:100] if statement else 'empty'}..., correct_answer: {correct_answer}")
        questions.append(
            Question(
                qid=f"{doc_id}_tf_{idx+1}",
                qtype="tf",
                prompt=fix_missing_spaces(sanitize_text(statement)),
                marks=TF_MARKS,
                options=["True", "False"],
                correct_answer=correct_answer,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=subject,
            )
        )
        logger.debug(f"TF {idx+1}/{total}: Created question qid={questions[-1].qid}")
    logger.info(f"Successfully built {len(questions)}/{total} TF questions")
    return questions


def summarise_mbpp_answer(item: Dict[str, Any]) -> str:
    code = item.get("code", "")
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    doc_lines = [line for line in lines if line.startswith("def ") or line.startswith("return ")]
    if not doc_lines:
        doc_lines = lines[:3]
    return " | ".join(doc_lines)[:300]


def build_long(pool: DatasetPool, total: int, doc_id: str) -> List[Question]:
    logger = get_logger()
    questions: List[Question] = []
    
    # Initialize LLM converter for generating explanations
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
    except Exception as e:
        logger.warning(f"LLM converter unavailable for LONG explanations: {e}. Using fallback.")
        llm_converter = None
    
    for idx in range(total):
        # Alternate between MBPP and GSM8K for variety
        if idx % 2 == 0 and len(pool.gsm8k_items) > 0:
            # Use GSM8K for math word problems
            item = pool.next_gsm8k()
            problem = normalize_whitespace(item.get("problem", ""))
            solution = normalize_whitespace(item.get("solution", ""))
            
            if problem and solution:
                # Fix missing spaces and sanitize
                problem = fix_missing_spaces(sanitize_text(problem))
                solution = fix_missing_spaces(sanitize_text(solution))
                prompt = textwrap.fill(problem, width=90)
                answer_summary = textwrap.fill(solution, width=90)
                
                # Generate explanation using LLM
                explanation = None
                if llm_converter:
                    try:
                        explanation = llm_converter.generate_answer_explanation(
                            question_text=prompt,
                            answer=answer_summary[:500],
                            question_type="LONG",
                            domain="mathematics",
                            academic_level="K-12"
                        )
                    except Exception as e:
                        logger.warning(f"Error generating GSM8K explanation: {e}")
                if not explanation:
                    explanation = "This solution demonstrates correct mathematical reasoning with clear step-by-step calculations leading to the final answer."
                
                questions.append(
                    Question(
                        qid=f"{doc_id}_long_{idx+1}",
                        qtype="long",
                        prompt=prompt,
                        marks=LONG_MARKS,
                        correct_answer=answer_summary,
                        explanation=explanation,
                        source_id=item.get("id"),
                        source_dataset="gsm8k_math",
                    )
                )
            else:
                # Fallback to MBPP if GSM8K item is invalid
                item = pool.next_mbpp()
                text = normalize_whitespace(item.get("text", "Explain the solution."))
                text = fix_missing_spaces(sanitize_text(text))
                prompt = textwrap.fill(text, width=90)
                answer_summary = sanitize_text(summarise_mbpp_answer(item) or "Refer to reference implementation in MBPP dataset.")
                split = item.get("split")
                dataset_name = f"mbpp_plus:{split}" if split else "mbpp_plus"
                
                # Generate explanation using LLM
                explanation = None
                if llm_converter:
                    try:
                        explanation = llm_converter.generate_answer_explanation(
                            question_text=prompt,
                            answer=answer_summary[:500],
                            question_type="LONG",
                            domain="computer_science",
                            academic_level="K-12"
                        )
                    except Exception as e:
                        logger.warning(f"Error generating MBPP explanation: {e}")
                if not explanation:
                    explanation = "This solution correctly implements the required algorithm with proper logic and code structure."
                
                questions.append(
                    Question(
                        qid=f"{doc_id}_long_{idx+1}",
                        qtype="long",
                        prompt=prompt,
                        marks=LONG_MARKS,
                        correct_answer=answer_summary,
                        explanation=explanation,
                        source_id=item.get("id"),
                        source_dataset=dataset_name,
                    )
                )
        else:
            # Use MBPP for coding problems
            item = pool.next_mbpp()
            text = normalize_whitespace(item.get("text", "Explain the solution."))
            text = fix_missing_spaces(sanitize_text(text))
            prompt = textwrap.fill(text, width=90)
            answer_summary = sanitize_text(summarise_mbpp_answer(item) or "Refer to reference implementation in MBPP dataset.")
            split = item.get("split")
            dataset_name = f"mbpp_plus:{split}" if split else "mbpp_plus"
            
            # Generate explanation using LLM
            explanation = None
            if llm_converter:
                try:
                    explanation = llm_converter.generate_answer_explanation(
                        question_text=prompt,
                        answer=answer_summary[:500],
                        question_type="LONG",
                        domain="computer_science",
                        academic_level="K-12"
                    )
                except Exception as e:
                    logger.warning(f"Error generating MBPP explanation: {e}")
            if not explanation:
                explanation = "This solution correctly implements the required algorithm with proper logic and code structure."
            
            questions.append(
                Question(
                    qid=f"{doc_id}_long_{idx+1}",
                    qtype="long",
                    prompt=prompt,
                    marks=LONG_MARKS,
                    correct_answer=answer_summary,
                    explanation=explanation,
                    source_id=item.get("id"),
                    source_dataset=dataset_name,
                )
            )
    return questions


def build_domain_mcq(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build domain-specific MCQ questions from MMLU dataset."""
    logger = get_logger()
    logger.info(f"Building {total} MCQ questions for domain '{domain}', doc_id: {doc_id}")
    questions: List[Question] = []
    
    # Get academic level from pool if available
    academic_level = getattr(pool, 'level', None) or "K-12"
    
    # Initialize LLM converter for generating explanations
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
    except Exception as e:
        logger.warning(f"LLM converter unavailable for MCQ explanations: {e}. Using fallback.")
        llm_converter = None
    
    for idx in range(total):
        logger.debug(f"MCQ {idx+1}/{total}: Starting question extraction for domain '{domain}'")
        attempts = 0
        item = None
        processed_options = None
        while attempts < 500:
            candidate = pool.next_mmlu()
            # Validate domain match
            if not validate_domain_match(candidate, domain, pool, pool.config if hasattr(pool, 'config') else None):
                attempts += 1
                continue
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices_raw = candidate.get("choices", [])[: len(LETTER_OPTIONS)]
            
            if question_raw and choices_raw and all(normalize_whitespace(c) for c in choices_raw):
                trimmed_question = normalize_whitespace(question_raw)
                trimmed_question = trimmed_question.rstrip('.')
                
                # Process options INSIDE the loop to validate them
                processed_choices = [normalize_whitespace(c) for c in choices_raw]
                options = [clean_option_text(fix_missing_spaces(sanitize_text(choice))) for choice in processed_choices]
                options = [opt for opt in options if opt and opt.strip()]  # Filter empty options
                
                # Validate question AND processed options
                if (len(trimmed_question) <= 180
                        and has_only_ascii(trimmed_question)
                        and len(options) >= 2  # Need at least 2 valid options
                        and all(len(opt) <= 90 and has_only_ascii(opt) for opt in options)):
                    item = candidate
                    processed_options = options  # Store processed options
                    break
            attempts += 1
        
        if item is None:
            logger.error(f"MCQ {idx+1}/{total}: Unable to find suitable MCQ item from {domain} domain after {attempts} attempts")
            raise RuntimeError(f"Unable to find suitable MCQ item from {domain} domain")
        
        logger.debug(f"MCQ {idx+1}/{total}: Selected item id={item.get('id', 'unknown')}, attempts={attempts}")
        question_raw = normalize_whitespace(item.get("question", ""))
        logger.debug(f"MCQ {idx+1}/{total}: Raw question length: {len(question_raw)}")
        question_text = textwrap.fill(fix_missing_spaces(sanitize_text(question_raw)), width=90)
        
        # Use stored processed_options instead of processing again
        options = processed_options
        logger.debug(f"MCQ {idx+1}/{total}: Using {len(options)} valid processed options")
        answer_idx = parse_answer_index(item.get("answer"), len(options))
        logger.debug(f"MCQ {idx+1}/{total}: Parsed answer index: {answer_idx} from answer: {item.get('answer')}, options count: {len(options)}")
        if answer_idx < 0 or answer_idx >= len(options):
            logger.error(f"MCQ {idx+1}/{total}: Invalid answer index {answer_idx} for {len(options)} options in {domain} domain (item id: {item.get('id', 'unknown')})")
            raise RuntimeError(f"Invalid answer index {answer_idx} for {len(options)} options in {domain} domain (item id: {item.get('id', 'unknown')})")
        if answer_idx >= len(LETTER_OPTIONS):
            logger.error(f"MCQ {idx+1}/{total}: Answer index {answer_idx} exceeds LETTER_OPTIONS length {len(LETTER_OPTIONS)} in {domain} domain (item id: {item.get('id', 'unknown')})")
            raise RuntimeError(f"Answer index {answer_idx} exceeds LETTER_OPTIONS length {len(LETTER_OPTIONS)} in {domain} domain (item id: {item.get('id', 'unknown')})")
        correct_letter = LETTER_OPTIONS[answer_idx]
        correct_text = options[answer_idx]
        logger.debug(f"MCQ {idx+1}/{total}: Selected answer - letter: {correct_letter}, text: {correct_text[:50] if correct_text else 'empty'}...")
        formatted_options = options
        subject = item.get("subject") or domain
        
        # Generate explanation using LLM
        explanation = None
        if llm_converter:
            try:
                explanation = llm_converter.generate_answer_explanation(
                    question_text=question_text,
                    answer=correct_text,
                    question_type="MCQ",
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.warning(f"Error generating MCQ explanation: {e}")
        # Fallback if LLM fails
        if not explanation:
            explanation = f"Correct option: {correct_letter} - {correct_text}"
        
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_mcq_{idx+1}",
                qtype="mcq",
                prompt=question_text or "Answer the question.",
                marks=MCQ_MARKS,
                options=formatted_options,
                correct_answer=correct_letter,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=f"{domain}_{subject}",
            )
        )
        logger.debug(f"MCQ {idx+1}/{total}: Created question qid={questions[-1].qid}")
    logger.info(f"Successfully built {len(questions)}/{total} MCQ questions for domain '{domain}'")
    return questions


def build_domain_mmlu_pro_mcq(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build domain-specific MCQ questions from MMLU-Pro dataset."""
    logger = get_logger()
    logger.info(f"Building {total} MMLU-Pro MCQ questions for domain '{domain}', doc_id: {doc_id}")
    
    # Check if MMLU-Pro pool is empty
    if not pool.mmlu_pro_items:
        logger.warning(
            f"MMLU-Pro pool is empty for domain '{domain}'. "
            f"Falling back to regular MMLU dataset."
        )
        return build_domain_mcq(pool, total, doc_id, domain)
    
    # Get academic level from pool if available
    academic_level = getattr(pool, 'level', None) or "K-12"
    
    # Initialize LLM converter for generating explanations
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
    except Exception as e:
        logger.warning(f"LLM converter unavailable for MMLU-Pro MCQ explanations: {e}. Using fallback.")
        llm_converter = None
    
    questions: List[Question] = []
    for idx in range(total):
        logger.debug(f"MMLU-Pro MCQ {idx+1}/{total}: Starting question extraction for domain '{domain}'")
        attempts = 0
        item = None
        processed_options = None
        while attempts < 500:
            candidate = pool.next_mmlu_pro()
            # Validate domain match
            if not validate_domain_match(candidate, domain, pool, pool.config if hasattr(pool, 'config') else None):
                attempts += 1
                continue
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices_raw = candidate.get("choices", [])[: len(LETTER_OPTIONS)]
            
            if question_raw and choices_raw and all(normalize_whitespace(c) for c in choices_raw):
                trimmed_question = normalize_whitespace(question_raw)
                trimmed_question = trimmed_question.rstrip('.')
                
                # Process options INSIDE the loop to validate them
                processed_choices = [normalize_whitespace(c) for c in choices_raw]
                options = [clean_option_text(fix_missing_spaces(sanitize_text(choice))) for choice in processed_choices]
                options = [opt for opt in options if opt and opt.strip()]  # Filter empty options
                
                # Validate question AND processed options
                if (len(trimmed_question) <= 180
                        and has_only_ascii(trimmed_question)
                        and len(options) >= 2  # Need at least 2 valid options
                        and all(len(opt) <= 90 and has_only_ascii(opt) for opt in options)):
                    item = candidate
                    processed_options = options  # Store processed options
                    break
            attempts += 1
        
        if item is None:
            logger.error(f"MMLU-Pro MCQ {idx+1}/{total}: Unable to find suitable MMLU-Pro MCQ item from {domain} domain after {attempts} attempts")
            raise RuntimeError(f"Unable to find suitable MMLU-Pro MCQ item from {domain} domain")
        
        logger.debug(f"MMLU-Pro MCQ {idx+1}/{total}: Selected item id={item.get('id', 'unknown')}, attempts={attempts}")
        question_raw = normalize_whitespace(item.get("question", ""))
        logger.debug(f"MMLU-Pro MCQ {idx+1}/{total}: Raw question length: {len(question_raw)}")
        question_text = textwrap.fill(fix_missing_spaces(sanitize_text(question_raw)), width=90)
        question_text = clean_question_text(question_text)
        
        # Use stored processed_options instead of processing again
        options = processed_options
        logger.debug(f"MMLU-Pro MCQ {idx+1}/{total}: Using {len(options)} valid processed options")
        answer_idx = parse_answer_index(item.get("answer"), len(options))
        logger.debug(f"MMLU-Pro MCQ {idx+1}/{total}: Parsed answer index: {answer_idx} from answer: {item.get('answer')}, options count: {len(options)}")
        if answer_idx < 0 or answer_idx >= len(options):
            logger.error(f"MMLU-Pro MCQ {idx+1}/{total}: Invalid answer index {answer_idx} for {len(options)} options in {domain} domain (item id: {item.get('id', 'unknown')})")
            raise RuntimeError(f"Invalid answer index {answer_idx} for {len(options)} options in {domain} domain (item id: {item.get('id', 'unknown')})")
        if answer_idx >= len(LETTER_OPTIONS):
            logger.error(f"MMLU-Pro MCQ {idx+1}/{total}: Answer index {answer_idx} exceeds LETTER_OPTIONS length {len(LETTER_OPTIONS)} in {domain} domain (item id: {item.get('id', 'unknown')})")
            raise RuntimeError(f"Answer index {answer_idx} exceeds LETTER_OPTIONS length {len(LETTER_OPTIONS)} in {domain} domain (item id: {item.get('id', 'unknown')})")
        correct_letter = LETTER_OPTIONS[answer_idx]
        correct_text = options[answer_idx]
        logger.debug(f"MMLU-Pro MCQ {idx+1}/{total}: Selected answer - letter: {correct_letter}, text: {correct_text[:50] if correct_text else 'empty'}...")
        formatted_options = options
        subject = item.get("subject") or domain
        
        # Generate explanation using LLM
        explanation = None
        if llm_converter:
            try:
                explanation = llm_converter.generate_answer_explanation(
                    question_text=question_text,
                    answer=correct_text,
                    question_type="MCQ",
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.warning(f"Error generating MMLU-Pro MCQ explanation: {e}")
        # Fallback if LLM fails
        if not explanation:
            explanation = f"Correct option: {correct_letter} - {correct_text}"
        
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_mmlu_pro_mcq_{idx+1}",
                qtype="mcq",
                prompt=question_text or "Answer the question.",
                marks=MCQ_MARKS,
                options=formatted_options,
                correct_answer=correct_letter,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=f"mmlu_pro_{domain}_{subject}",
            )
        )
        logger.debug(f"MMLU-Pro MCQ {idx+1}/{total}: Created question qid={questions[-1].qid}")
    logger.info(f"Successfully built {len(questions)}/{total} MMLU-Pro MCQ questions for domain '{domain}'")
    return questions


def build_domain_tf(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build domain-specific True/False questions from MMLU dataset."""
    questions: List[Question] = []
    logger = get_logger()
    
    # Get academic level from pool if available
    academic_level = getattr(pool, 'level', None) or "K-12"
    
    logger.info(f"Building {total} TF questions for {domain} {academic_level}")
    
    # Initialize LLM converter (with fallback if unavailable)
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        logger.info(f"LLM converter initialized successfully for {domain} {academic_level}")
    except ValueError as e:
        logger.warning(f"LLM converter unavailable for TF generation: {e}. Using fallback method.")
        llm_converter = None
    except Exception as e:
        logger.warning(f"Error initializing LLM converter for TF: {e}. Using fallback method.")
        llm_converter = None
    
    # Track LLM vs fallback usage
    llm_count = 0
    fallback_count = 0
    
    if llm_converter is None:
        logger.info(f"LLM converter not available, using fallback for all TF questions (domain: {domain})")
    
    for idx in range(total):
        attempts = 0
        item = None
        while attempts < 500:
            candidate = pool.next_mmlu()
            # Validate domain match
            if not validate_domain_match(candidate, domain, pool, pool.config if hasattr(pool, 'config') else None):
                attempts += 1
                continue
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices = candidate.get("choices", [])
            if question_raw and choices:
                trimmed_question = normalize_whitespace(question_raw)
                trimmed_question = trimmed_question.rstrip('.')
                if (len(trimmed_question) <= 180
                        and has_only_ascii(trimmed_question)
                        and all(len(normalize_whitespace(c)) <= 90 and has_only_ascii(normalize_whitespace(c)) for c in choices)):
                    item = candidate
                    break
            attempts += 1
        if item is None:
            raise RuntimeError(f"Unable to find suitable True/False item from {domain} domain")
        question_raw = normalize_whitespace(item.get("question", ""))
        question_text = fix_missing_spaces(question_raw)
        question_text = clean_question_text(question_text)
        choices = [fix_missing_spaces(normalize_whitespace(c)) for c in item.get("choices", [])]
        
        # Ensure choices list is not empty - retry if empty
        if not choices:
            # Retry with a new item
            attempts_retry = 0
            while attempts_retry < 100:
                candidate = pool.next_mmlu()
                choices = candidate.get("choices", [])
                if choices:
                    item = candidate
                    question_text = normalize_whitespace(item.get("question", ""))
                    choices = [normalize_whitespace(c) for c in choices]
                    break
                attempts_retry += 1
            if not choices:
                raise RuntimeError(f"Unable to find suitable True/False item with valid choices from {domain} domain")
        
        answer_idx = parse_answer_index(item.get("answer"), len(choices))
        
        # Ensure answer_idx is within bounds (should never happen due to parse_answer_index, but double-check)
        if answer_idx < 0 or answer_idx >= len(choices):
            # This should never happen if parse_answer_index works correctly, but if it does, clamp it
            answer_idx = max(0, min(answer_idx, len(choices) - 1))
        
        # Double-check that answer_idx is valid before accessing
        if answer_idx >= len(choices) or answer_idx < 0:
            # Retry with a new item
            attempts_retry = 0
            while attempts_retry < 100:
                candidate = pool.next_mmlu()
                choices = candidate.get("choices", [])
                if choices and len(choices) > 0:
                    item = candidate
                    question_text = normalize_whitespace(item.get("question", ""))
                    choices = [normalize_whitespace(c) for c in choices]
                    answer_idx = parse_answer_index(item.get("answer"), len(choices))
                    if 0 <= answer_idx < len(choices):
                        break
                attempts_retry += 1
            if answer_idx >= len(choices) or answer_idx < 0:
                raise RuntimeError(f"Unable to find suitable True/False item with valid answer index from {domain} domain")
        
        correct_text = choices[answer_idx]
        
        # Try LLM conversion first
        llm_statements = None
        if llm_converter:
            logger.info(f"Attempting LLM conversion for TF question {idx+1}/{total} (domain: {domain}, level: {academic_level})")
            try:
                llm_statements = llm_converter.convert_mcq_to_tf(
                    mcq_stem=question_text,
                    choices=choices,
                    correct_answer=correct_text,
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.info(f"Exception during LLM TF conversion for question {idx+1}: {e}. Using fallback method (domain: {domain})")
        
        # Use LLM-generated statements if available, otherwise use fallback
        if llm_statements:
            logger.info(f"LLM conversion successful for TF question {idx+1} (domain: {domain})")
            llm_count += 1
            # Randomly choose between true and false statement
            make_true = random.random() < 0.5
            if make_true:
                statement = llm_statements["true_statement"]
                correct_answer = "True"
            else:
                statement = llm_statements["false_statement"]
                correct_answer = "False"
        else:
            if llm_converter:
                logger.info(f"LLM conversion returned None, using fallback for TF question {idx+1} (domain: {domain})")
            fallback_count += 1
            # Fallback to original method
            false_text = None
            if choices and len(choices) > 1:
                incorrect = [c for i, c in enumerate(choices) if i != answer_idx and i < len(choices)]
                if incorrect:
                    false_text = random.choice(incorrect)
            make_true = random.random() < 0.5 or not false_text
            question_text_short = textwrap.shorten(question_text, width=140, placeholder="...")
            if make_true:
                statement = f"The correct answer to '{question_text_short}' is '{correct_text}'."
                correct_answer = "True"
            else:
                statement = f"The correct answer to '{question_text_short}' is '{false_text}'."
                correct_answer = "False"
        
        # Generate explanation using LLM
        explanation = None
        if llm_converter:
            try:
                explanation = llm_converter.generate_answer_explanation(
                    question_text=statement,
                    answer=correct_answer,
                    question_type="TF",
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.warning(f"Error generating TF explanation: {e}")
        # Fallback if LLM fails
        if not explanation:
            if correct_answer == "True":
                explanation = f"This statement is true based on the correct answer: '{correct_text}'."
            else:
                explanation = f"This statement is false. The correct answer is: '{correct_text}'."
        
        subject = item.get("subject") or domain
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_tf_{idx+1}",
                qtype="tf",
                prompt=fix_missing_spaces(sanitize_text(statement)),
                marks=TF_MARKS,
                options=["True", "False"],
                correct_answer=correct_answer,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=f"{domain}_{subject}",
            )
        )
    
    logger.info(f"Generated {len(questions)} TF questions for {domain} {academic_level} ({llm_count} from LLM, {fallback_count} from fallback)")
    return questions


def build_domain_pubmedqa_tf(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build True/False questions from PubMedQA dataset for biology graduate papers."""
    questions: List[Question] = []
    logger = get_logger()
    
    # Safety check - this should only be called for biology domain
    if domain != "biology":
        logger.warning(f"build_domain_pubmedqa_tf called for domain '{domain}', expected 'biology'. Using PubMedQA anyway.")
    
    logger.info(f"Building {total} TF questions from PubMedQA for {domain}")
    
    for idx in range(total):
        # Get next PubMedQA item
        item = pool.next_pubmedqa()
        
        # Extract question and final_decision
        question_raw = normalize_whitespace(item.get("question", ""))
        question_text = fix_missing_spaces(question_raw)
        final_decision = item.get("final_decision", "").lower()
        long_answer = item.get("long_answer", "")
        
        # Skip items with "maybe" (should be filtered during loading, but double-check)
        if final_decision not in ["yes", "no"]:
            logger.warning(f"Skipping PubMedQA item with final_decision='{final_decision}' (expected 'yes' or 'no')")
            # Try next item
            item = pool.next_pubmedqa()
            question_raw = normalize_whitespace(item.get("question", ""))
            question_text = fix_missing_spaces(question_raw)
            final_decision = item.get("final_decision", "").lower()
            long_answer = item.get("long_answer", "")
        
        # Convert final_decision to True/False
        # Direct conversion: "yes" -> True, "no" -> False
        if final_decision == "yes":
            correct_answer = "True"
            explanation = long_answer if long_answer else "This statement is correct based on PubMed research."
        elif final_decision == "no":
            correct_answer = "False"
            explanation = long_answer if long_answer else "This statement is incorrect based on PubMed research."
        else:
            # Fallback (should not happen if filtering works)
            logger.warning(f"Unexpected final_decision '{final_decision}', defaulting to False")
            correct_answer = "False"
            explanation = "Unable to determine correctness from PubMedQA data."
        
        # Use question text directly as the True/False statement
        statement = question_text
        
        # Sanitize text (no escaping - done at render time)
        statement_cleaned = fix_missing_spaces(sanitize_text(statement))
        
        # Get source_id - convert to int if possible, otherwise use index
        source_id = item.get("id")
        if source_id is not None:
            try:
                source_id = int(source_id) if not isinstance(source_id, int) else source_id
            except (ValueError, TypeError):
                source_id = idx  # Fallback to index if conversion fails
        
        questions.append(
            Question(
                qid=f"{doc_id}_biology_pubmedqa_tf_{idx+1}",
                qtype="tf",
                prompt=statement_cleaned,
                marks=TF_MARKS,
                options=["True", "False"],
                correct_answer=correct_answer,
                explanation=fix_missing_spaces(sanitize_text(explanation)) if explanation else "Based on PubMed research.",
                source_id=source_id,
                source_dataset="pubmedqa_labeled",
            )
        )
    
    logger.info(f"Generated {len(questions)} TF questions from PubMedQA for {domain}")
    return questions
def _convert_question_to_statement(question: str, answer: str) -> str:
    """Convert a question to a declarative statement using pattern matching."""
    question = question.strip().rstrip('?')
    question_lower = question.lower()
    
    # Pattern-based conversions
    if question_lower.startswith("how many"):
        # "How many X did Y" -> "Y [verb] [answer] X"
        parts = question.split()
        if len(parts) >= 4:
            # Find verb (did, does, do, has, have, sent, etc.)
            try:
                verb_idx = next(i for i, w in enumerate(parts) if w.lower() in ["did", "does", "do", "has", "have", "sent", "sends", "send"])
                verb = parts[verb_idx].lower()
                subject = " ".join(parts[verb_idx+1:])
                
                # Convert verb to past tense if needed
                verb_map = {
                    "did": "did",
                    "does": "did",
                    "do": "did",
                    "has": "had",
                    "have": "had",
                    "sent": "sent",
                    "sends": "sent",
                    "send": "sent"
                }
                past_verb = verb_map.get(verb, "did")
                
                # Extract the noun (X in "how many X")
                noun_parts = parts[2:verb_idx]  # Everything between "many" and the verb
                noun = " ".join(noun_parts) if noun_parts else ""
                
                if noun:
                    statement = f"{subject} {past_verb} {answer} {noun}."
                else:
                    statement = f"{subject} {past_verb} {answer}."
            except StopIteration:
                statement = f"{answer}."
        else:
            statement = f"{answer}."
    
    elif question_lower.startswith("what"):
        # "What X" -> "The X is [answer]" or "[answer] is the X"
        if "is" in question_lower or "was" in question_lower:
            # "What is X" -> "X is [answer]"
            parts = question.split()
            try:
                is_idx = next(i for i, w in enumerate(parts) if w.lower() in ["is", "was"])
                subject = " ".join(parts[is_idx+1:])
                statement = f"{subject} is {answer}."
            except StopIteration:
                statement = f"{answer}."
        else:
            # "What style of X emerged" -> "The [answer] style of X emerged"
            # Extract the noun phrase after "what"
            parts = question.split()
            if len(parts) > 1:
                noun_phrase = " ".join(parts[1:])
                # Check if answer should be inserted before a key word
                if "style" in noun_phrase.lower() and answer.lower() not in noun_phrase.lower():
                    # Insert answer before "style"
                    noun_phrase = noun_phrase.replace("style", f"{answer} style", 1)
                    statement = f"The {noun_phrase}."
                else:
                    statement = f"The {noun_phrase} is {answer}."
            else:
                statement = f"{answer}."
    
    elif question_lower.startswith("where"):
        # "Where did X go" -> "X went to [answer]"
        parts = question.split()
        if len(parts) >= 3:
            subject = " ".join(parts[1:])
            # Remove question words and common verbs, convert to past tense
            subject = subject.replace("did ", "").replace("does ", "").replace("do ", "")
            
            # Check if there's already a verb
            if "went" in subject.lower():
                # Replace "went where" or "went" with "went to [answer]"
                statement = subject.replace("where", answer, 1).replace("went", f"went to {answer}", 1)
                if "to" not in statement.lower():
                    statement = f"{subject} to {answer}."
                else:
                    statement = f"{statement}."
            elif "go" in subject.lower():
                statement = subject.replace("go", f"went to {answer}", 1) + "."
            else:
                # No explicit verb, add "went to"
                statement = f"{subject} went to {answer}."
        else:
            statement = f"{answer}."
    
    elif question_lower.startswith("when"):
        # "When did X" -> "X happened in [answer]"
        parts = question.split()
        if len(parts) >= 3:
            subject = " ".join(parts[1:])
            subject = subject.replace("did ", "").replace("does ", "").replace("do ", "")
            statement = f"{subject} in {answer}."
        else:
            statement = f"{answer}."
    
    elif question_lower.startswith("who"):
        # "Who did X" -> "[answer] did X"
        parts = question.split()
        if len(parts) >= 3:
            action = " ".join(parts[1:])
            statement = f"{answer} {action}."
        else:
            statement = f"{answer}."
    
    elif question_lower.startswith("was") or question_lower.startswith("were"):
        # "Was X Y" -> "X was Y" or "X was [answer]"
        parts = question.split()
        if len(parts) >= 3:
            subject = parts[1]
            statement = f"{subject} was {answer}."
        else:
            statement = f"{answer}."
    
    else:
        # Generic fallback: try to make it a statement
        # Remove question words and add answer
        question_clean = question
        for qword in ["what", "where", "when", "who", "how", "why", "which"]:
            if question_clean.lower().startswith(qword):
                question_clean = question_clean[len(qword):].strip()
                break
        
        statement = f"{question_clean} {answer}."
    
    # Clean up the statement
    statement = normalize_whitespace(statement)
    if not statement.endswith('.'):
        statement += '.'
    
    return statement


def build_domain_squad_tf(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build True/False questions from SQuAD dataset."""
    questions: List[Question] = []
    logger = get_logger()
    
    logger.info(f"Building {total} TF questions from SQuAD for {domain}")
    
    for idx in range(total):
        logger.debug(f"SQuAD TF {idx+1}/{total}: Starting question extraction for domain '{domain}'")
        # Get next SQuAD item
        item = pool.next_squad()
        logger.debug(f"SQuAD TF {idx+1}/{total}: Extracted item id={item.get('id', 'unknown')}, title={item.get('title', 'unknown')[:50]}")
        
        # Extract question, answer, and context
        question_text = normalize_whitespace(item.get("question", ""))
        answer_text = normalize_whitespace(item.get("answer_text", ""))
        # Get original context first to preserve answer_start accuracy
        original_context = item.get("context", "")
        context = normalize_whitespace(original_context)
        answer_start = item.get("answer_start", 0)
        logger.debug(f"SQuAD TF {idx+1}/{total}: Question length: {len(question_text)}, answer length: {len(answer_text)}, context length: {len(context)}")
        
        # Skip items with missing data
        if not question_text or not answer_text or not context:
            logger.warning(f"SQuAD TF {idx+1}/{total}: Skipping SQuAD item with missing question, answer, or context")
            # Try next item
            item = pool.next_squad()
            question_text = normalize_whitespace(item.get("question", ""))
            answer_text = normalize_whitespace(item.get("answer_text", ""))
            original_context = item.get("context", "")
            context = normalize_whitespace(original_context)
            answer_start = item.get("answer_start", 0)
        
        # Fix missing spaces
        question_text = fix_missing_spaces(question_text)
        answer_text = fix_missing_spaces(answer_text)
        context = fix_missing_spaces(context)
        
        # Strategy: Extract the sentence from context that contains the answer
        # This gives us a natural declarative statement
        statement = None
        
        # Try to find the sentence containing the answer
        # Note: answer_start is for original_context, but we use normalized context
        # So we need to find the answer position in the normalized context
        if answer_text.lower() in context.lower():
            # Find answer position in normalized context (since normalization might change positions)
            answer_pos = context.lower().find(answer_text.lower())
            if answer_pos >= 0 and answer_pos < len(context):
                # Use the found position instead of original answer_start
                effective_answer_start = answer_pos
                effective_answer_end = min(len(context), answer_pos + len(answer_text))
                
                # Find sentence boundaries around the answer
                # Look for sentence start (capital letter or start of text) before answer
                sentence_start = 0
                search_start = max(0, effective_answer_start - 300)  # Look back up to 300 chars
                for i in range(search_start, effective_answer_start):
                    if i == 0:
                        sentence_start = 0
                        break
                    elif i > 0 and i < len(context) and context[i-1] in '.!?\n':
                        # Check if next char is capital or whitespace+capital
                        if i < len(context) - 1:
                            if context[i].isupper() or (context[i].isspace() and i+1 < len(context) and context[i+1].isupper()):
                                sentence_start = i
                                break
                
                # Find sentence end (period, exclamation, question mark) after answer
                sentence_end = len(context)
                search_end = min(len(context), effective_answer_end + 300)  # Look forward up to 300 chars
                for i in range(effective_answer_end, search_end):
                    if i < len(context) and context[i] in '.!?\n':
                        sentence_end = i + 1
                        break
                
                # Extract the sentence
                if sentence_end > sentence_start and sentence_start < len(context) and sentence_end <= len(context):
                    extracted = context[sentence_start:sentence_end].strip()
                    # Verify the answer is in this sentence
                    if answer_text.lower() in extracted.lower():
                        statement = extracted
                        # Clean up the statement
                        statement = statement.strip('.!?')
                        if not statement.endswith('.'):
                            statement += '.'
        
        # If we couldn't extract a good sentence, use pattern-based conversion
        if not statement or len(statement) < 10:
            statement = _convert_question_to_statement(question_text, answer_text)
        
        # Randomly decide whether to make it True or False
        make_true = random.random() < 0.5
        
        if make_true:
            # Use the statement as-is (it's true)
            correct_answer = "True"
            explanation = f"According to the passage: {answer_text}" if answer_text else "This statement is correct based on the reading passage."
        else:
            # Create a false statement by modifying the answer
            # Try to find and replace the answer with a plausible alternative
            statement_lower = statement.lower()
            answer_lower = answer_text.lower()
            
            if answer_lower in statement_lower:
                # Replace answer with a modified version or negation
                # Simple approach: add "not" or change number/name
                if answer_text.isdigit() or (answer_text.replace(',', '').replace('.', '').isdigit()):
                    # For numeric answers, change the number
                    try:
                        # Handle numbers with commas
                        num_str = answer_text.replace(',', '').replace('.', '')
                        num = int(num_str)
                        false_num = str(num + random.randint(1, max(10, num // 10)))
                        statement = statement.replace(answer_text, false_num, 1)
                    except:
                        statement = statement.replace(answer_text, "a different value", 1)
                elif len(answer_text.split()) == 1:  # Single word answer
                    # Try to negate or modify
                    if answer_text.lower() in ["yes", "no", "not"]:
                        # For yes/no, flip it
                        if answer_text.lower() == "yes":
                            statement = statement.replace(answer_text, "no", 1)
                        elif answer_text.lower() == "no":
                            statement = statement.replace(answer_text, "yes", 1)
                        else:
                            statement = statement.replace(answer_text, f"not {answer_text}", 1)
                    else:
                        # For other single words, try to replace with similar but wrong
                        statement = statement.replace(answer_text, "a different answer", 1)
                else:
                    # Multi-word answer - replace with generic false
                    statement = statement.replace(answer_text, "a different answer", 1)
            else:
                # Answer not found in statement, add negation
                statement = f"It is not true that {statement.lower().rstrip('.')}."
            
            correct_answer = "False"
            explanation = f"The correct answer is: {answer_text}" if answer_text else "This statement is incorrect based on the reading passage."
        
        # Sanitize text (no escaping - done at render time)
        statement_cleaned = fix_missing_spaces(sanitize_text(statement))
        explanation_cleaned = fix_missing_spaces(sanitize_text(explanation))
        
        # Get source_id
        source_id = item.get("id")
        if source_id is not None:
            try:
                source_id = str(source_id) if not isinstance(source_id, str) else source_id
            except (ValueError, TypeError):
                source_id = str(idx)
        else:
            source_id = str(idx)
        
        logger.debug(f"SQuAD TF {idx+1}/{total}: Statement: {statement[:100] if statement else 'empty'}..., correct_answer: {correct_answer}")
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_squad_tf_{idx+1}",
                qtype="tf",
                prompt=statement_cleaned,
                marks=TF_MARKS,
                options=["True", "False"],
                correct_answer=correct_answer,
                explanation=explanation_cleaned,
                source_id=source_id,
                source_dataset="squad",
            )
        )
        logger.debug(f"SQuAD TF {idx+1}/{total}: Created question qid={questions[-1].qid}")
    
    logger.info(f"Successfully built {len(questions)}/{total} TF questions from SQuAD for {domain}")
    return questions


def build_domain_squad_long(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build Long-form questions from SQuAD dataset."""
    questions: List[Question] = []
    logger = get_logger()
    
    logger.info(f"Building {total} long-form questions from SQuAD for {domain}")
    
    for idx in range(total):
        logger.debug(f"SQuAD Long {idx+1}/{total}: Starting question extraction for domain '{domain}'")
        # Get next SQuAD item
        item = pool.next_squad()
        logger.debug(f"SQuAD Long {idx+1}/{total}: Extracted item id={item.get('id', 'unknown')}, title={item.get('title', 'unknown')[:50]}")
        
        # Extract question, answer, and context
        question_text = normalize_whitespace(item.get("question", ""))
        answer_text = normalize_whitespace(item.get("answer_text", ""))
        context = normalize_whitespace(item.get("context", ""))
        logger.debug(f"SQuAD Long {idx+1}/{total}: Question length: {len(question_text)}, answer length: {len(answer_text)}, context length: {len(context)}")
        
        # Skip items with missing data
        if not question_text or not context:
            logger.warning(f"SQuAD Long {idx+1}/{total}: Skipping SQuAD item with missing question or context")
            # Try next item
            item = pool.next_squad()
            question_text = normalize_whitespace(item.get("question", ""))
            answer_text = normalize_whitespace(item.get("answer_text", ""))
            context = normalize_whitespace(item.get("context", ""))
        
        # Fix missing spaces
        question_text = fix_missing_spaces(question_text)
        answer_text = fix_missing_spaces(answer_text)
        context = fix_missing_spaces(context)
        
        # Truncate context if too long (keep it reasonable for LaTeX)
        max_context_length = 800
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Build the prompt: include context passage and question
        # Format: "Read the following passage and answer the question: [context] Question: [question]"
        prompt = f"Read the following passage and answer the question:\n\n{context}\n\nQuestion: {question_text}"
        
        # Use answer text as part of the expected answer/explanation
        explanation = f"Expected answer: {answer_text}" if answer_text else "Answer based on the information provided in the passage."
        
        # Sanitize text (no escaping - done at render time)
        prompt_cleaned = fix_missing_spaces(sanitize_text(prompt))
        explanation_cleaned = fix_missing_spaces(sanitize_text(explanation))
        
        # Get source_id
        source_id = item.get("id")
        if source_id is not None:
            try:
                source_id = str(source_id) if not isinstance(source_id, str) else source_id
            except (ValueError, TypeError):
                source_id = str(idx)
        else:
            source_id = str(idx)
        
        logger.debug(f"SQuAD Long {idx+1}/{total}: Created prompt length: {len(prompt)}, answer: {answer_text[:50] if answer_text else 'empty'}...")
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_squad_long_{idx+1}",
                qtype="long",
                prompt=prompt_cleaned,
                marks=LONG_MARKS,
                options=None,
                correct_answer=answer_text if answer_text else "",
                explanation=explanation_cleaned,
                source_id=source_id,
                source_dataset="squad",
            )
        )
        logger.debug(f"SQuAD Long {idx+1}/{total}: Created question qid={questions[-1].qid}")
    
    logger.info(f"Successfully built {len(questions)}/{total} long-form questions from SQuAD for {domain}")
    return questions


def build_domain_mmlu_pro_tf(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build domain-specific True/False questions from MMLU-Pro dataset."""
    logger = get_logger()
    
    # Get academic level from pool if available
    academic_level = getattr(pool, 'level', None) or "K-12"
    
    logger.info(f"Building {total} TF questions for {domain} {academic_level}")
    
    # Validate pool size
    if pool.mmlu_pro_items and len(pool.mmlu_pro_items) < total * 2:  # Need at least 2x questions for selection
        logger.warning(
            f"Filtered MMLU-Pro pool for domain '{domain}' has only {len(pool.mmlu_pro_items)} items, "
            f"requested {total}. May result in domain mismatches."
        )
    
    # Check if MMLU-Pro pool is empty
    if not pool.mmlu_pro_items:
        logger.warning(
            f"MMLU-Pro pool is empty for domain '{domain}'. "
            f"Falling back to regular MMLU dataset."
        )
        return build_domain_tf(pool, total, doc_id, domain)
    
    questions: List[Question] = []
    
    # Initialize LLM converter (with fallback if unavailable)
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        logger.info(f"LLM converter initialized successfully for {domain} {academic_level}")
    except ValueError as e:
        logger.warning(f"LLM converter unavailable for TF generation: {e}. Using fallback method.")
        llm_converter = None
    except Exception as e:
        logger.warning(f"Error initializing LLM converter for TF: {e}. Using fallback method.")
        llm_converter = None
    
    # Track LLM vs fallback usage
    llm_count = 0
    fallback_count = 0
    
    if llm_converter is None:
        logger.info(f"LLM converter not available, using fallback for all TF questions (domain: {domain})")
    
    for idx in range(total):
        attempts = 0
        item = None
        while attempts < 500:
            candidate = pool.next_mmlu_pro()
            # Validate domain match
            if not validate_domain_match(candidate, domain, pool, pool.config if hasattr(pool, 'config') else None):
                attempts += 1
                continue
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices = candidate.get("choices", [])
            if question_raw and choices:
                trimmed_question = normalize_whitespace(question_raw)
                trimmed_question = trimmed_question.rstrip('.')
                if (len(trimmed_question) <= 180
                        and has_only_ascii(trimmed_question)
                        and all(len(normalize_whitespace(c)) <= 90 and has_only_ascii(normalize_whitespace(c)) for c in choices)):
                    item = candidate
                    break
            attempts += 1
        if item is None:
            raise RuntimeError(f"Unable to find suitable MMLU-Pro True/False item from {domain} domain")
        question_raw = normalize_whitespace(item.get("question", ""))
        question_text = fix_missing_spaces(question_raw)
        question_text = clean_question_text(question_text)
        choices = [fix_missing_spaces(normalize_whitespace(c)) for c in item.get("choices", [])]
        
        # Ensure choices list is not empty - retry if empty
        if not choices:
            # Retry with a new item
            attempts_retry = 0
            while attempts_retry < 100:
                candidate = pool.next_mmlu_pro()
                choices = candidate.get("choices", [])
                if choices:
                    item = candidate
                    question_raw = normalize_whitespace(item.get("question", ""))
                    question_text = fix_missing_spaces(question_raw)
                    choices = [fix_missing_spaces(normalize_whitespace(c)) for c in choices]
                    break
                attempts_retry += 1
            if not choices:
                raise RuntimeError(f"Unable to find suitable MMLU-Pro True/False item with valid choices from {domain} domain")
        
        answer_idx = parse_answer_index(item.get("answer"), len(choices))
        
        # Ensure answer_idx is within bounds (should never happen due to parse_answer_index, but double-check)
        if answer_idx < 0 or answer_idx >= len(choices):
            # This should never happen if parse_answer_index works correctly, but if it does, clamp it
            answer_idx = max(0, min(answer_idx, len(choices) - 1))
        
        # Double-check that answer_idx is valid before accessing
        if answer_idx >= len(choices) or answer_idx < 0:
            # Retry with a new item
            attempts_retry = 0
            while attempts_retry < 100:
                candidate = pool.next_mmlu_pro()
                choices = candidate.get("choices", [])
                if choices and len(choices) > 0:
                    item = candidate
                    question_text = normalize_whitespace(item.get("question", ""))
                    choices = [normalize_whitespace(c) for c in choices]
                    answer_idx = parse_answer_index(item.get("answer"), len(choices))
                    if 0 <= answer_idx < len(choices):
                        break
                attempts_retry += 1
            if answer_idx >= len(choices) or answer_idx < 0:
                raise RuntimeError(f"Unable to find suitable MMLU-Pro True/False item with valid answer index from {domain} domain")
        
        correct_text = choices[answer_idx]
        
        # Try LLM conversion first
        llm_statements = None
        if llm_converter:
            logger.info(f"Attempting LLM conversion for TF question {idx+1}/{total} (domain: {domain}, level: {academic_level})")
            try:
                llm_statements = llm_converter.convert_mcq_to_tf(
                    mcq_stem=question_text,
                    choices=choices,
                    correct_answer=correct_text,
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.info(f"Exception during LLM TF conversion for question {idx+1}: {e}. Using fallback method (domain: {domain})")
        
        # Use LLM-generated statements if available, otherwise use fallback
        if llm_statements:
            logger.info(f"LLM conversion successful for TF question {idx+1} (domain: {domain})")
            llm_count += 1
            # Randomly choose between true and false statement
            make_true = random.random() < 0.5
            if make_true:
                statement = llm_statements["true_statement"]
                correct_answer = "True"
            else:
                statement = llm_statements["false_statement"]
                correct_answer = "False"
        else:
            if llm_converter:
                logger.info(f"LLM conversion returned None, using fallback for TF question {idx+1} (domain: {domain})")
            fallback_count += 1
            # Fallback to original method
            false_text = None
            if choices and len(choices) > 1:
                incorrect = [c for i, c in enumerate(choices) if i != answer_idx and i < len(choices)]
                if incorrect:
                    false_text = random.choice(incorrect)
            make_true = random.random() < 0.5 or not false_text
            question_text_short = textwrap.shorten(question_text, width=140, placeholder="...")
            if make_true:
                statement = f"The correct answer to '{question_text_short}' is '{correct_text}'."
                correct_answer = "True"
            else:
                statement = f"The correct answer to '{question_text_short}' is '{false_text}'."
                correct_answer = "False"
        
        # Generate explanation using LLM
        explanation = None
        if llm_converter:
            try:
                explanation = llm_converter.generate_answer_explanation(
                    question_text=statement,
                    answer=correct_answer,
                    question_type="TF",
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.warning(f"Error generating MMLU-Pro TF explanation: {e}")
        # Fallback if LLM fails
        if not explanation:
            if correct_answer == "True":
                explanation = f"This statement is true based on the correct answer: '{correct_text}'."
            else:
                explanation = f"This statement is false. The correct answer is: '{correct_text}'."
        
        subject = item.get("subject") or domain
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_mmlu_pro_tf_{idx+1}",
                qtype="tf",
                prompt=fix_missing_spaces(sanitize_text(statement)),
                marks=TF_MARKS,
                options=["True", "False"],
                correct_answer=correct_answer,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=f"mmlu_pro_{domain}_{subject}",
            )
        )
    
    logger.info(f"Generated {len(questions)} TF questions for {domain} {academic_level} ({llm_count} from LLM, {fallback_count} from fallback)")
    return questions


def build_domain_arc_mcq(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build domain-specific MCQ questions from AI2-ARC dataset."""
    logger = get_logger()
    logger.info(f"Building {total} ARC MCQ questions for domain '{domain}', doc_id: {doc_id}")
    
    # Get academic level from pool if available
    academic_level = getattr(pool, 'level', None) or "K-12"
    
    # Initialize LLM converter for generating explanations
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
    except Exception as e:
        logger.warning(f"LLM converter unavailable for ARC MCQ explanations: {e}. Using fallback.")
        llm_converter = None
    
    questions: List[Question] = []
    for idx in range(total):
        logger.debug(f"ARC MCQ {idx+1}/{total}: Starting question extraction for domain '{domain}'")
        attempts = 0
        item = None
        processed_options = None
        while attempts < 500:
            candidate = pool.next_arc()
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices_dict = candidate.get("choices", {})
            choices_raw = choices_dict.get("text", [])[:len(LETTER_OPTIONS)] if isinstance(choices_dict, dict) else choices_dict[:len(LETTER_OPTIONS)]
            
            if question_raw and choices_raw and all(normalize_whitespace(c) for c in choices_raw):
                trimmed_question = normalize_whitespace(question_raw)
                trimmed_question = trimmed_question.rstrip('.')
                
                # Process options INSIDE the loop to validate them
                processed_choices = [normalize_whitespace(c) for c in choices_raw]
                options = [clean_option_text(fix_missing_spaces(sanitize_text(choice))) for choice in processed_choices]
                options = [opt for opt in options if opt and opt.strip()]  # Filter empty options
                
                # Validate question AND processed options
                if (len(trimmed_question) <= 180
                        and has_only_ascii(trimmed_question)
                        and len(options) >= 2  # Need at least 2 valid options
                        and all(len(opt) <= 90 and has_only_ascii(opt) for opt in options)):
                    item = candidate
                    processed_options = options  # Store processed options
                    break
            attempts += 1
        
        if item is None:
            logger.error(f"ARC MCQ {idx+1}/{total}: Unable to find suitable AI2-ARC MCQ item from {domain} domain after {attempts} attempts")
            raise RuntimeError(f"Unable to find suitable AI2-ARC MCQ item from {domain} domain")
        
        logger.debug(f"ARC MCQ {idx+1}/{total}: Selected item id={item.get('id', 'unknown')}, attempts={attempts}")
        question_raw = normalize_whitespace(item.get("question", ""))
        logger.debug(f"ARC MCQ {idx+1}/{total}: Raw question length: {len(question_raw)}")
        question_text = textwrap.fill(fix_missing_spaces(sanitize_text(question_raw)), width=90)
        
        # Use stored processed_options instead of processing again
        options = processed_options
        logger.debug(f"ARC MCQ {idx+1}/{total}: Using {len(options)} valid processed options")
        
        answer_idx = item.get("answer_index", 0)
        logger.debug(f"ARC MCQ {idx+1}/{total}: Answer index from item: {answer_idx}, options count: {len(options)}")
        
        # Validate answer_idx is within bounds
        if answer_idx < 0 or answer_idx >= len(options):
            logger.warning(f"ARC MCQ {idx+1}/{total}: Invalid answer_index {answer_idx} for {len(options)} options, clamping to 0")
            answer_idx = 0
        
        correct_letter = LETTER_OPTIONS[answer_idx] if answer_idx < len(LETTER_OPTIONS) else "A"
        # FIX: Add empty check before accessing
        correct_text = options[answer_idx] if options and answer_idx < len(options) else ""
        logger.debug(f"ARC MCQ {idx+1}/{total}: Selected answer - letter: {correct_letter}, text: {correct_text[:50] if correct_text else 'empty'}...")
        formatted_options = options
        subject = item.get("subject") or domain
        
        # Generate explanation using LLM
        explanation = None
        if llm_converter:
            try:
                explanation = llm_converter.generate_answer_explanation(
                    question_text=question_text,
                    answer=correct_text,
                    question_type="MCQ",
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.warning(f"Error generating ARC MCQ explanation: {e}")
        # Fallback if LLM fails
        if not explanation:
            explanation = f"Correct option: {correct_letter} - {correct_text}"
        
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_arc_mcq_{idx+1}",
                qtype="mcq",
                prompt=question_text or "Answer the question.",
                marks=MCQ_MARKS,
                options=formatted_options,
                correct_answer=correct_letter,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=f"arc_{domain}_{subject}",
            )
        )
        logger.debug(f"ARC MCQ {idx+1}/{total}: Created question qid={questions[-1].qid}")
    logger.info(f"Successfully built {len(questions)}/{total} ARC MCQ questions for domain '{domain}'")
    return questions


def build_domain_arc_tf(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build domain-specific True/False questions from AI2-ARC dataset."""
    questions: List[Question] = []
    logger = get_logger()
    
    # Get academic level from pool if available
    academic_level = getattr(pool, 'level', None) or "K-12"
    
    logger.info(f"Building {total} TF questions for {domain} {academic_level}")
    
    # Initialize LLM converter (with fallback if unavailable)
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        logger.info(f"LLM converter initialized successfully for {domain} {academic_level}")
    except ValueError as e:
        logger.warning(f"LLM converter unavailable for TF generation: {e}. Using fallback method.")
        llm_converter = None
    except Exception as e:
        logger.warning(f"Error initializing LLM converter for TF: {e}. Using fallback method.")
        llm_converter = None
    
    # Track LLM vs fallback usage
    llm_count = 0
    fallback_count = 0
    
    if llm_converter is None:
        logger.info(f"LLM converter not available, using fallback for all TF questions (domain: {domain})")
    
    for idx in range(total):
        attempts = 0
        item = None
        while attempts < 500:
            candidate = pool.next_arc()
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices_dict = candidate.get("choices", {})
            choices = choices_dict.get("text", []) if isinstance(choices_dict, dict) else choices_dict
            if question_raw and choices:
                trimmed_question = normalize_whitespace(question_raw)
                trimmed_question = trimmed_question.rstrip('.')
                if (len(trimmed_question) <= 180
                        and has_only_ascii(trimmed_question)
                        and all(len(normalize_whitespace(c)) <= 90 and has_only_ascii(normalize_whitespace(c)) for c in choices)):
                    item = candidate
                    break
            attempts += 1
        if item is None:
            raise RuntimeError(f"Unable to find suitable AI2-ARC True/False item from {domain} domain")
        question_raw = normalize_whitespace(item.get("question", ""))
        question_text = fix_missing_spaces(question_raw)
        question_text = clean_question_text(question_text)
        choices_dict = item.get("choices", {})
        choices_list = choices_dict.get("text", []) if isinstance(choices_dict, dict) else choices_dict
        choices = [fix_missing_spaces(normalize_whitespace(c)) for c in choices_list]
        
        # Ensure choices list is not empty - retry if empty
        if not choices:
            # Retry with a new item
            attempts_retry = 0
            while attempts_retry < 100:
                candidate = pool.next_arc()
                choices_dict_retry = candidate.get("choices", {})
                choices_list_retry = choices_dict_retry.get("text", []) if isinstance(choices_dict_retry, dict) else choices_dict_retry
                if choices_list_retry:
                    item = candidate
                    question_raw = normalize_whitespace(item.get("question", ""))
                    question_text = fix_missing_spaces(question_raw)
                    choices = [fix_missing_spaces(normalize_whitespace(c)) for c in choices_list_retry]
                    break
                attempts_retry += 1
            if not choices:
                raise RuntimeError(f"Unable to find suitable AI2-ARC True/False item with valid choices from {domain} domain")
        
        answer_idx = item.get("answer_index", 0)
        # Ensure answer_idx is within bounds
        if answer_idx < 0 or answer_idx >= len(choices):
            answer_idx = max(0, min(answer_idx, len(choices) - 1))
        
        correct_text = choices[answer_idx] if answer_idx < len(choices) and len(choices) > 0 else "N/A"
        
        # Try LLM conversion first
        llm_statements = None
        if llm_converter:
            logger.info(f"Attempting LLM conversion for TF question {idx+1}/{total} (domain: {domain}, level: {academic_level})")
            try:
                llm_statements = llm_converter.convert_mcq_to_tf(
                    mcq_stem=question_text,
                    choices=choices,
                    correct_answer=correct_text,
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.info(f"Exception during LLM TF conversion for question {idx+1}: {e}. Using fallback method (domain: {domain})")
        
        # Use LLM-generated statements if available, otherwise use fallback
        if llm_statements:
            logger.info(f"LLM conversion successful for TF question {idx+1} (domain: {domain})")
            llm_count += 1
            # Randomly choose between true and false statement
            make_true = random.random() < 0.5
            if make_true:
                statement = llm_statements["true_statement"]
                correct_answer = "True"
            else:
                statement = llm_statements["false_statement"]
                correct_answer = "False"
        else:
            if llm_converter:
                logger.info(f"LLM conversion returned None, using fallback for TF question {idx+1} (domain: {domain})")
            fallback_count += 1
            # Fallback to original method
            false_text = None
            if choices:
                incorrect = [c for i, c in enumerate(choices) if i != answer_idx]
                if incorrect:
                    false_text = random.choice(incorrect)
            make_true = random.random() < 0.5 or not false_text
            question_text_short = textwrap.shorten(question_text, width=140, placeholder="...")
            if make_true:
                statement = f"The correct answer to '{question_text_short}' is '{correct_text}'."
                correct_answer = "True"
            else:
                statement = f"The correct answer to '{question_text_short}' is '{false_text}'."
                correct_answer = "False"
        
        # Generate explanation using LLM
        explanation = None
        if llm_converter:
            try:
                explanation = llm_converter.generate_answer_explanation(
                    question_text=statement,
                    answer=correct_answer,
                    question_type="TF",
                    domain=domain,
                    academic_level=academic_level
                )
            except Exception as e:
                logger.warning(f"Error generating ARC TF explanation: {e}")
        # Fallback if LLM fails
        if not explanation:
            if correct_answer == "True":
                explanation = f"This statement is true based on the correct answer: '{correct_text}'."
            else:
                explanation = f"This statement is false. The correct answer is: '{correct_text}'."
        
        subject = item.get("subject") or domain
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_arc_tf_{idx+1}",
                qtype="tf",
                prompt=escape_latex(statement),
                marks=TF_MARKS,
                options=["True", "False"],
                correct_answer=correct_answer,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=f"arc_{domain}_{subject}",
            )
        )
    
    logger.info(f"Generated {len(questions)} TF questions for {domain} {academic_level} ({llm_count} from LLM, {fallback_count} from fallback)")
    return questions


def generate_dynamic_question_distribution(
    total_marks: int = 40,
    available_types: List[str] = None,
    require_all_types: bool = False
) -> Dict[str, int]:
    """
    Return fixed question distribution for all papers.
    
    Note: This function now always returns the fixed distribution
    (5 MCQ, 5 TF, 2 Long-form) regardless of parameters.
    Parameters are kept for backward compatibility but are ignored.
    
    Args:
        total_marks: Total marks for the paper (default: 40, ignored)
        available_types: List of available question types (ignored)
        require_all_types: If True, ensure every available type appears (ignored)
    
    Returns:
        Dictionary with fixed counts: {"mcq": 5, "tf": 5, "long": 2}
    """
    return get_fixed_question_distribution()


def infer_mcq_source_from_combination(combination: Iterable[str]) -> str:
    """Decide which MCQ dataset the combination is anchored on."""
    combo_list = list(combination)
    if any(item.startswith("mmlu_pro_") for item in combo_list):
        return "mmlu_pro"
    if any(item.startswith("arc_") for item in combo_list):
        return "arc"
    return "mmlu"


def _next_mcq_from_source(pool: DatasetPool, source: str) -> Dict[str, Any]:
    logger = get_logger()
    if source == "mmlu_pro":
        # Check if MMLU-Pro pool is empty, fallback to regular MMLU
        if not pool.mmlu_pro_items:
            logger.warning(
                f"MMLU-Pro pool is empty, falling back to regular MMLU for long-form question generation"
            )
            return pool.next_mmlu()
        return pool.next_mmlu_pro()
    if source == "arc":
        return pool.next_arc()
    return pool.next_mmlu()


def extract_question_from_stem(stem: str) -> Tuple[str, Optional[str]]:
    """
    Extract the actual question part from a stem that may contain a passage.
    
    Returns:
        Tuple of (question_part, passage_summary) where passage_summary is optional.
    """
    stem = normalize_whitespace(stem)
    
    # Check if stem contains a passage (long text or passage indicators)
    has_passage = (
        len(stem) > 200 or
        "This question refers to" in stem or
        "The following passage" in stem.lower() or
        "The following information" in stem.lower() or
        stem.count('"') >= 2  # Likely has quoted passage
    )
    
    if not has_passage:
        # Simple question, return as-is
        return (stem, None)
    
    # Try to find the actual question part
    question_markers = [
        "Which of the following",
        "What",
        "Who",
        "When",
        "Where",
        "How",
        "Why",
        "According to",
        "Based on"
    ]
    
    question_part = None
    passage_text = None
    
    # Look for question markers
    for marker in question_markers:
        marker_lower = marker.lower()
        stem_lower = stem.lower()
        idx = stem_lower.find(marker_lower)
        if idx >= 0:
            # Found a question marker, extract from here
            question_part = stem[idx:].strip()
            passage_text = stem[:idx].strip()
            break
    
    # If no marker found, look for the last question mark
    if question_part is None:
        qmark_idx = stem.rfind("?")
        if qmark_idx > 0:
            # Take text around the question mark (last 200 chars before it, or from start)
            start_idx = max(0, qmark_idx - 200)
            question_part = stem[start_idx:qmark_idx + 1].strip()
            if start_idx > 0:
                passage_text = stem[:start_idx].strip()
        else:
            # No question mark, take last 200 characters
            question_part = stem[-200:].strip()
            if len(stem) > 200:
                passage_text = stem[:-200].strip()
    
    # Clean up question part
    if question_part:
        # Remove common passage prefixes
        question_part = re.sub(r'^This question refers to.*?\.\s*', '', question_part, flags=re.IGNORECASE)
        question_part = re.sub(r'^The following passage.*?\.\s*', '', question_part, flags=re.IGNORECASE)
        question_part = normalize_whitespace(question_part)
    
    # Generate passage summary if passage exists
    passage_summary = None
    if passage_text and len(passage_text) > 50:
        passage_summary = summarize_passage(passage_text)
    
    return (question_part or stem[:200], passage_summary)


def summarize_passage(passage: str) -> str:
    """
    Create a 1-sentence summary of a passage for context.
    
    Returns a concise summary like "Based on a passage about [topic] by [author]..."
    """
    passage = normalize_whitespace(passage)
    
    # Try to extract author name (look for patterns like "-Author Name" or "by Author Name")
    author_match = re.search(r'[-–—]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', passage)
    author = None
    if author_match:
        author = author_match.group(1)
    else:
        # Try "by Author Name" pattern
        by_match = re.search(r'\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', passage, re.IGNORECASE)
        if by_match:
            author = by_match.group(1)
    
    # Try to extract topic/keywords (look for quoted text or key phrases)
    topic = None
    # Look for quoted text (often the title or key concept)
    quote_match = re.search(r'"([^"]{10,60})"', passage)
    if quote_match:
        topic = quote_match.group(1)
    else:
        # Look for common topic indicators
        topic_patterns = [
            r'about\s+([^.,]{10,40})',
            r'regarding\s+([^.,]{10,40})',
            r'on\s+([^.,]{10,40})',
        ]
        for pattern in topic_patterns:
            match = re.search(pattern, passage, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                break
    
    # Try to extract time period/date
    date_match = re.search(r'\b(\d{4})\b', passage)
    time_period = date_match.group(1) if date_match else None
    
    # Build summary
    parts = []
    if author:
        parts.append(f"by {author}")
    if topic:
        parts.append(f"about {topic[:40]}")
    if time_period:
        parts.append(f"from {time_period}")
    
    if parts:
        summary = "Based on a passage " + ", ".join(parts) + "."
    else:
        # Fallback: generic summary
        summary = "Based on the provided passage."
    
    # Ensure summary is not too long
    if len(summary) > 80:
        summary = summary[:77] + "..."
    
    return summary


def synthesise_long_from_mcq(
    pool: DatasetPool,
    total: int,
    doc_id: str,
    domain: str,
    source: str = "mmlu",
    start_index: int = 0,
    academic_level: Optional[str] = None,
    config=None,
) -> List[Question]:
    """Create long-form questions derived from MCQ items using LLM conversion."""
    questions: List[Question] = []
    logger = get_logger()
    
    # Extract academic level from doc_id if not provided (needed for validation)
    if academic_level is None:
        # Try to extract from doc_id format: "domain_level_doc_XX"
        parts = doc_id.split("_")
        if len(parts) >= 2:
            level_part = parts[1]
            if level_part.lower() in ["k-12", "k12", "undergraduate", "graduate"]:
                academic_level = level_part.replace("k12", "K-12").title()
            else:
                academic_level = pool.level if hasattr(pool, 'level') and pool.level else "K-12"
        else:
            academic_level = pool.level if hasattr(pool, 'level') and pool.level else "K-12"
    
    # Validate source availability and fallback if needed
    if source == "mmlu_pro" and not pool.mmlu_pro_items:
        logger.warning(
            f"MMLU-Pro pool is empty for domain '{domain}', level '{academic_level}'. "
            f"Falling back to regular MMLU for synthetic long-form questions."
        )
        source = "mmlu"
    elif source == "arc" and not pool.ai2_arc_items:
        logger.warning(
            f"AI2-ARC pool is empty for domain '{domain}', level '{academic_level}'. "
            f"Falling back to regular MMLU for synthetic long-form questions."
        )
        source = "mmlu"
    
    # Initialize LLM converter
    try:
        if config is None:
            config = get_config()
        llm_config = config.get_llm_config()
        llm_converter = LLMConverter(llm_config)
    except ValueError as e:
        logger.error(f"Failed to initialize LLM converter: {e}")
        logger.warning("Falling back to rule-based conversion (questions may be lower quality)")
        llm_converter = None
    except Exception as e:
        logger.error(f"Unexpected error initializing LLM converter: {e}")
        llm_converter = None
    
    logger.info(f"Building {total} synthetic long-form questions from {source.upper()} for domain '{domain}', doc_id: {doc_id}, academic_level: {academic_level}")
    for idx in range(total):
        logger.debug(f"Long-form {idx+1}/{total}: Starting question extraction from {source.upper()} for domain '{domain}'")
        attempts = 0
        item = None
        question_part = None
        passage_summary = None
        
        while attempts < 300:
            try:
                candidate = _next_mcq_from_source(pool, source)
            except RuntimeError as e:
                # If the requested source is empty, fallback to regular MMLU
                if "empty" in str(e).lower() and source != "mmlu":
                    logger.warning(
                        f"Long-form {idx+1}/{total}: {source.upper()} pool is empty, "
                        f"falling back to regular MMLU"
                    )
                    source = "mmlu"
                    try:
                        candidate = _next_mcq_from_source(pool, source)
                    except RuntimeError as fallback_error:
                        logger.error(
                            f"Long-form {idx+1}/{total}: Both {source.upper()} and MMLU pools are empty. "
                            f"Cannot generate long-form question."
                        )
                        candidate = None
                        break
                else:
                    # Re-raise if it's a different error
                    raise
            
            # Skip processing if candidate is None (both pools empty)
            if candidate is None:
                break
            
            stem = normalize_whitespace(candidate.get("question", ""))
            original_choices_count = len(candidate.get("choices", []))
            choices = [normalize_whitespace(c) for c in candidate.get("choices", []) if normalize_whitespace(c)]
            logger.debug(f"Long-form {idx+1}/{total}: Extracted {len(choices)} choices from candidate (original: {original_choices_count})")
            
            if not stem or not choices:
                logger.debug(f"Long-form {idx+1}/{total}: Skipping candidate - stem empty: {not stem}, choices empty: {not choices}")
                attempts += 1
                continue
            
            # Filter by length: skip stems > 300 characters (likely have long passages)
            if len(stem) > 300:
                logger.debug(f"Long-form {idx+1}/{total}: Skipping candidate - stem too long: {len(stem)} chars")
                attempts += 1
                continue
            
            # Prefer questions with stems between 50-250 characters
            if len(stem) < 50:
                logger.debug(f"Long-form {idx+1}/{total}: Skipping candidate - stem too short: {len(stem)} chars")
                attempts += 1
                continue
            
            # Extract question part and passage summary (for context if needed)
            try:
                question_part, passage_summary = extract_question_from_stem(stem)
                
                # Validate extracted question
                if not question_part or len(question_part.strip()) == 0:
                    logger.debug(f"Long-form {idx+1}/{total}: Skipping candidate - extracted question part is empty")
                    attempts += 1
                    continue
                
                # Check if question makes sense (contains at least one question word)
                question_words = ["which", "what", "who", "when", "where", "how", "why"]
                question_lower = question_part.lower()
                if not any(word in question_lower for word in question_words) and "?" not in question_part:
                    logger.debug(f"Long-form {idx+1}/{total}: Skipping candidate - question doesn't contain question words")
                    attempts += 1
                    continue
                
                item = candidate
                break
            except Exception as e:
                logger.debug(f"Long-form {idx+1}/{total}: Error extracting question from stem: {e}")
                attempts += 1
                continue
        
        if not item or not question_part:
            logger.warning(f"Long-form {idx+1}/{total}: Could not find suitable MCQ item for long-form question after {attempts} attempts")
            continue
        
        # Re-validate choices haven't become empty
        if not choices or len(choices) == 0:
            logger.warning(f"Long-form {idx+1}/{total}: Choices became empty after processing, skipping item")
            continue
        
        logger.info(f"Long-form {idx+1}/{total}: Selected MCQ item with {len(choices)} choices, stem length: {len(stem)}, item_id: {item.get('id', 'unknown')}")
        
        answer_idx = parse_answer_index(item.get("answer"), len(choices))
        logger.debug(f"Long-form {idx+1}/{total}: Parsed answer index: {answer_idx} from answer: {item.get('answer')}, choices count: {len(choices)}")
        
        if answer_idx < 0 or answer_idx >= len(choices):
            logger.warning(f"Long-form {idx+1}/{total}: Invalid answer_idx {answer_idx} for {len(choices)} choices, clamping to 0")
            answer_idx = 0
        
        # Add validation before accessing
        if not choices or answer_idx >= len(choices):
            logger.error(f"Long-form {idx+1}/{total}: Cannot access choices[{answer_idx}], choices length: {len(choices) if choices else 0}")
            continue
        
        correct_text = choices[answer_idx] if choices else ""
        logger.debug(f"Long-form {idx+1}/{total}: Selected correct answer text: {correct_text[:50] if correct_text else 'empty'}...")
        
        # Use LLM to convert MCQ to long-form question
        if llm_converter:
            try:
                # Use the full stem for LLM conversion (it handles passages intelligently)
                converted_question = llm_converter.convert_mcq_to_long_form(
                    mcq_stem=stem,
                    correct_answer=correct_text,
                    domain=domain,
                    academic_level=academic_level
                )
                
                if converted_question:
                    long_form_question = converted_question
                    logger.debug(f"Successfully converted MCQ to long-form using LLM")
                else:
                    # Fallback: use extracted question part
                    logger.warning(f"LLM conversion returned None, using extracted question part")
                    long_form_question = question_part
                    if passage_summary:
                        long_form_question = f"{passage_summary} {question_part}"
            except Exception as e:
                logger.warning(f"Error during LLM conversion: {e}. Using extracted question part as fallback.")
                long_form_question = question_part
                if passage_summary:
                    long_form_question = f"{passage_summary} {question_part}"
        else:
            # Fallback to rule-based conversion
            long_form_question = question_part
            if passage_summary:
                long_form_question = f"{passage_summary} {question_part}"
        
        # Use the question text directly without additional instructions
        long_prompt = long_form_question
        # Fix missing spaces before escaping
        long_prompt = clean_question_text(fix_missing_spaces(long_prompt))
        
        # Generate meaningful long-form answer using LLM
        long_answer = None
        if llm_converter:
            try:
                long_answer = llm_converter.generate_long_form_answer(
                    long_form_question=long_prompt,
                    correct_answer_text=correct_text,
                    domain=domain,
                    academic_level=academic_level or "K-12"
                )
            except Exception as e:
                logger.warning(f"Error generating long-form answer via LLM: {e}")
        
        # Fallback to template if LLM fails
        if not long_answer:
            if domain == "mathematics":
                long_answer = (
                    f"{correct_text}. Explain why this is the correct answer and provide supporting evidence. "
                    f"Show the calculation or reasoning that leads to this conclusion."
                )
            else:
                long_answer = (
                    f"{correct_text}. Explain why this is the correct answer and provide supporting evidence. "
                    f"Reference key facts or concepts that support this choice."
                )
        
        # Generate explanation using LLM
        explanation = None
        if llm_converter:
            try:
                explanation = llm_converter.generate_answer_explanation(
                    question_text=long_prompt,
                    answer=long_answer[:500],  # Truncate long answers for explanation prompt
                    question_type="LONG",
                    domain=domain,
                    academic_level=academic_level or "K-12"
                )
            except Exception as e:
                logger.warning(f"Error generating LONG explanation: {e}")
        # Fallback if LLM fails
        if not explanation:
            explanation = f"This answer correctly addresses the key concept: {correct_text}. Award credit for demonstrating understanding and providing supporting reasoning."
        
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_long_{start_index + idx + 1}",
                qtype="long",
                prompt=textwrap.fill(fix_missing_spaces(sanitize_text(long_prompt)), width=90),
                marks=LONG_MARKS,
                correct_answer=textwrap.fill(fix_missing_spaces(sanitize_text(long_answer)), width=90),
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=f"{source}_mcq_to_long",
            )
        )
        logger.debug(f"Long-form {idx+1}/{total}: Created question qid={questions[-1].qid}")
    
    logger.info(f"Successfully built {len(questions)}/{total} synthetic long-form questions from {source.upper()} for domain '{domain}'")
    return questions


def build_domain_long(
    pool: DatasetPool,
    total: int,
    doc_id: str,
    domain: str,
    mcq_source: str = "mmlu",
    use_native: bool = True,
    allow_synthetic: bool = True,
) -> List[Question]:
    """Build domain-specific long-form questions."""
    logger = get_logger()
    logger.info(f"Building {total} long-form questions for domain '{domain}', doc_id: {doc_id}, use_native: {use_native}, allow_synthetic: {allow_synthetic}")
    questions: List[Question] = []
    
    # Get academic level from pool if available
    academic_level = getattr(pool, 'level', None) or "K-12"
    
    # Initialize LLM converter for generating explanations
    llm_converter = None
    try:
        if hasattr(pool, 'config') and pool.config:
            llm_config = pool.config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
        else:
            config = get_config()
            llm_config = config.get_llm_config()
            llm_converter = LLMConverter(llm_config)
    except Exception as e:
        logger.warning(f"LLM converter unavailable for LONG explanations: {e}. Using fallback.")
        llm_converter = None
    
    programming_domains = {"computer_science", "cybersecurity", "machine_learning"}
    
    if use_native:
        for idx in range(total):
            logger.debug(f"Long-form {idx+1}/{total}: Starting native question extraction for domain '{domain}'")
            if domain == "mathematics":
                # Use GSM8K for math long-form questions
                item = pool.next_gsm8k()
                logger.debug(f"Long-form {idx+1}/{total}: Extracted GSM8K item id={item.get('id', 'unknown')}")
                problem = normalize_whitespace(item.get("problem", ""))
                solution = normalize_whitespace(item.get("solution", ""))
                logger.debug(f"Long-form {idx+1}/{total}: Problem length: {len(problem)}, solution length: {len(solution)}")
                
                if problem and solution:
                    # Fix missing spaces and sanitize
                    problem = clean_question_text(fix_missing_spaces(sanitize_text(problem)))
                    solution = fix_missing_spaces(sanitize_text(solution))
                    prompt = textwrap.fill(problem, width=90)
                    answer_summary = textwrap.fill(solution, width=90)
                    
                    # Generate explanation using LLM
                    explanation = None
                    if llm_converter:
                        try:
                            explanation = llm_converter.generate_answer_explanation(
                                question_text=prompt,
                                answer=answer_summary[:500],
                                question_type="LONG",
                                domain=domain,
                                academic_level=academic_level
                            )
                        except Exception as e:
                            logger.warning(f"Error generating GSM8K explanation: {e}")
                    if not explanation:
                        explanation = "This solution demonstrates correct mathematical reasoning with clear step-by-step calculations leading to the final answer."
                    
                    questions.append(
                        Question(
                            qid=f"{doc_id}_{domain}_long_{idx+1}",
                            qtype="long",
                            prompt=prompt,
                            marks=LONG_MARKS,
                            correct_answer=answer_summary,
                            explanation=explanation,
                            source_id=item.get("id"),
                            source_dataset="gsm8k_math",
                        )
                    )
                    logger.debug(f"Long-form {idx+1}/{total}: Created GSM8K question qid={questions[-1].qid}")
            elif domain in programming_domains:
                # Use MBPP+ for programming-related domains only
                item = pool.next_mbpp()
                logger.debug(f"Long-form {idx+1}/{total}: Extracted MBPP item id={item.get('id', 'unknown')}")
                text = normalize_whitespace(item.get("text", "Explain the solution."))
                logger.debug(f"Long-form {idx+1}/{total}: Text length: {len(text)}")
                text = fix_missing_spaces(sanitize_text(text))
                prompt = textwrap.fill(text, width=90)
                answer_summary = sanitize_text(summarise_mbpp_answer(item) or "Refer to reference implementation in MBPP dataset.")
                split = item.get("split")
                dataset_name = f"mbpp_plus:{split}" if split else "mbpp_plus"
                
                # Generate explanation using LLM
                explanation = None
                if llm_converter:
                    try:
                        explanation = llm_converter.generate_answer_explanation(
                            question_text=prompt,
                            answer=answer_summary[:500],
                            question_type="LONG",
                            domain=domain,
                            academic_level=academic_level
                        )
                    except Exception as e:
                        logger.warning(f"Error generating MBPP explanation: {e}")
                if not explanation:
                    explanation = "This solution correctly implements the required algorithm with proper logic and code structure."
                
                questions.append(
                    Question(
                        qid=f"{doc_id}_{domain}_long_{idx+1}",
                        qtype="long",
                        prompt=prompt,
                        marks=LONG_MARKS,
                        correct_answer=answer_summary,
                        explanation=explanation,
                        source_id=item.get("id"),
                        source_dataset=dataset_name,
                    )
                )
                logger.debug(f"Long-form {idx+1}/{total}: Created MBPP question qid={questions[-1].qid}")
            else:
                # No native long-form dataset for this domain
                logger.debug(f"Long-form {idx+1}/{total}: No native dataset for domain '{domain}', skipping")
                continue
    
    if allow_synthetic and len(questions) < total:
        remaining = total - len(questions)
        logger.info(f"Long-form: Built {len(questions)}/{total} native questions, generating {remaining} synthetic questions from {mcq_source.upper()}")
        # Get academic level from pool if available
        academic_level = pool.level if hasattr(pool, 'level') and pool.level else None
        questions.extend(
            synthesise_long_from_mcq(
                pool,
                remaining,
                doc_id,
                domain,
                source=mcq_source,
                start_index=len(questions),
                academic_level=academic_level,
                config=pool.config if hasattr(pool, 'config') else None,
            )
        )
    
    logger.info(f"Successfully built {len(questions)}/{total} long-form questions for domain '{domain}'")
    return questions


def render_latex(title: str, total_marks: int, sections: Dict[str, List[Question]]) -> str:
    logger = get_logger()
    logger.info(f"Rendering LaTeX for document: {title}, total marks: {total_marks}")
    
    question_counts = {qtype: len(questions) for qtype, questions in sections.items()}
    logger.debug(f"Question counts: {question_counts}")
    
    section_titles = {"mcq": "Multiple Choice", "tf": "True / False", "long": "Long Form Response"}
    latex_parts = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{textcomp}",
        r"\usepackage{enumitem}",
        r"\usepackage{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{amsmath, amssymb}",
        r"\geometry{margin=1in}",
        r"\begin{document}",
        r"\begin{center}",
        f"{title} \\",
        f"Total Marks: {total_marks} \\",
        "Answer all questions.",
        r"\end{center}",
        ""
    ]
    
    # Track cumulative question count across all sections
    question_counter = 0
    
    for qtype in ["mcq", "tf", "long"]:
        questions = sections.get(qtype, [])
        if not questions:
            logger.debug(f"Skipping {qtype} section - no questions")
            continue
        
        logger.debug(f"Rendering {len(questions)} {qtype} questions")
        section_marks = sum(q.marks for q in questions)
        latex_parts.append(f"\\section*{{{section_titles[qtype]} ({section_marks} marks)}}")
        if qtype == "tf":
            latex_parts.append(r"\textit{Answer True or False}")
        if qtype == "long":
            latex_parts.append(r"\textit{Answer each question with a clear and complete explanation.}")
        
        # Set counter to continue from previous section
        # LaTeX enumerate counters are 0-indexed: counter=0 shows as 1, counter=6 shows as 7
        # So if we've processed 7 questions, set counter to 7 so next item shows as 8
        latex_parts.append(r"\begin{enumerate}[label=\arabic*.]")
        if question_counter > 0:
            latex_parts.append(fr"\setcounter{{enumi}}{{{question_counter}}}")

        for question in questions:
            logger.debug(f"Adding {qtype} question {question.qid} to LaTeX (question {question_counter + 1})")
            # If the question has an associated image, insert it before the prompt
            if getattr(question, 'image_path', None):
                img_name = Path(question.image_path).name
                logger.debug(f"Question {question.qid} has associated image: {img_name}")
                latex_parts.append(r"\begin{center}")
                latex_parts.append(f"\\includegraphics[width=0.6\\textwidth]{{{img_name}}}")
                latex_parts.append(r"\end{center}")
            
            # Handle prompt - always escape at render time (text is raw in Question objects)
            prompt_text = escape_latex(fix_missing_spaces(question.prompt))
            
            if qtype == "tf":
                latex_parts.append(f"\\item True or False: {prompt_text}")
            else:
                latex_parts.append(f"\\item {prompt_text}")
            if qtype == "mcq" and question.options:
                latex_parts.append(r"\begin{enumerate}[label=(\alph*)]")
                for opt in question.options:
                    # Always escape options at render time
                    opt_text = escape_latex(fix_missing_spaces(opt))
                    latex_parts.append(f"    \\item {opt_text}")
                latex_parts.append(r"\end{enumerate}")
            question_counter += 1
        latex_parts.append(r"\end{enumerate}")
        latex_parts.append("")
    latex_parts.extend([r"\vfill", r"\noindent\textit{End of Paper}", r"\end{document}"])
    logger.info(f"LaTeX rendering completed for {title} - total questions: {question_counter}, total marks: {total_marks}")
    return "\n".join(latex_parts)


def render_answer_key_latex(title: str, sections: Dict[str, List[Question]]) -> str:
    """Generate LaTeX for an answer key that mirrors the question ordering."""
    section_titles = {"mcq": "Multiple Choice", "tf": "True / False", "long": "Long Form Response"}
    latex_parts = [
        r"\documentclass[12pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{textcomp}",
        r"\usepackage{enumitem}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1in}",
        r"\begin{document}",
        r"\begin{center}",
        f"Answer Key - {title} \\",
        r"\small (Answers are ordered exactly as in the question paper.)",
        r"\end{center}",
        ""
    ]

    question_counter = 0
    for qtype in ["mcq", "tf", "long"]:
        questions = sections.get(qtype, [])
        if not questions:
            continue

        latex_parts.append(f"\\section*{{{section_titles[qtype]}}}")
        latex_parts.append(r"\begin{enumerate}[label=\arabic*.]")
        if question_counter > 0:
            latex_parts.append(fr"\setcounter{{enumi}}{{{question_counter}}}")

        for question in questions:
            answer_text = question.correct_answer or "Not provided"
            explanation = question.explanation or ""
            # Escape text for LaTeX
            answer_text_escaped = escape_latex(fix_missing_spaces(str(answer_text)))
            latex_parts.append(r"\item " + f"\\textbf{{Answer:}} {answer_text_escaped}")

            if question.options:
                option_strings = []
                for idx, opt in enumerate(question.options):
                    label = LETTER_OPTIONS[idx] if idx < len(LETTER_OPTIONS) else f"Option {idx+1}"
                    opt_escaped = escape_latex(fix_missing_spaces(str(opt)))
                    option_strings.append(f"{label}) {opt_escaped}")
                latex_parts.append(f"\\\\ \\textit{{Options:}} {'; '.join(option_strings)}")

            if explanation:
                explanation_escaped = escape_latex(fix_missing_spaces(str(explanation)))
                latex_parts.append(f"\\\\ \\textit{{Note:}} {explanation_escaped}")

            question_counter += 1

        latex_parts.append(r"\end{enumerate}")
        latex_parts.append("")

    latex_parts.extend([r"\vfill", r"\noindent\textit{End of Answer Key}", r"\end{document}"])
    return "\n".join(latex_parts)


def write_answer_key(
    doc_id: str,
    title: str,
    sections: Dict[str, List[Question]],
    compiler: PDFCompiler,
    latex_dir: Path,
    pdf_dir: Path,
) -> Tuple[Path, Path]:
    """Create answer-key LaTeX/PDF alongside the question paper."""
    latex_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    latex_content = render_answer_key_latex(title, sections)
    latex_file = latex_dir / f"{doc_id}_answer_key.tex"
    latex_file.write_text(latex_content, encoding="utf-8")

    pdf_file = compiler.compile_latex_to_pdf(latex_file, output_dir=pdf_dir)
    return latex_file, pdf_file


def total_marks(sections: Dict[str, List[Question]]) -> int:
    return sum(q.marks for questions in sections.values() for q in questions)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def get_pdf_page_count(pdf_file: Path, compiler: PDFCompiler) -> Optional[int]:
    """Get page count from PDF file using PDFCompiler or fallback libraries."""
    try:
        pdf_info = compiler.get_pdf_info(pdf_file)
        return pdf_info.get("page_count")
    except Exception:
        # Fallback to PyPDF2 or PyMuPDF if available
        try:
            import PyPDF2
            with open(pdf_file, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                return len(pdf_reader.pages)
        except ImportError:
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_file)
                page_count = len(doc)
                doc.close()
                return page_count
            except ImportError:
                return None


def format_question_options(question: Question) -> Optional[Dict[str, str]]:
    """Format question options as a dictionary for structured JSON output."""
    if not question.options:
        return None
    
    if question.qtype == "mcq":
        # Format MCQ options as {"A": "...", "B": "..."}
        formatted = {}
        for idx, option in enumerate(question.options):
            if idx < len(LETTER_OPTIONS):
                letter = LETTER_OPTIONS[idx]
                formatted[letter] = option
        return formatted if formatted else None
    elif question.qtype == "tf":
        # Format TF options as {"True": "True", "False": "False"}
        return {"True": "True", "False": "False"}
    else:
        # Long-form questions don't have options
        return None


def generate_structured_json(
    doc_id: str,
    title: str,
    domain: str,
    level: str,
    subjects: List[str],
    combination_used: List[str],
    sections: Dict[str, List[Question]],
    gold_data: Dict[str, Any],
    metadata: Dict[str, Any],
    latex_file: Path,
    pdf_file: Path,
    compiler: PDFCompiler,
    answer_key_latex_file: Optional[Path] = None,
    answer_key_pdf_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate structured JSON output for a document."""
    # Get all questions in order (mcq, tf, long)
    all_questions = []
    question_number = 1
    
    # Collect all questions from sections
    for section_type in ["mcq", "tf", "long"]:
        if section_type in sections:
            for q in sections[section_type]:
                all_questions.append((question_number, q))
                question_number += 1
    
    # Create a mapping of question_id to gold answer data
    gold_answers_map = {
        ans["question_id"]: ans 
        for ans in gold_data.get("answers", [])
    }
    
    # Build questions array
    questions_array = []
    for q_num, question in all_questions:
        gold_answer_data = gold_answers_map.get(question.qid, {})
        
        question_data = {
            "question_number": q_num,
            "question_id": question.qid,
            "question_type": question.qtype.upper() if question.qtype == "mcq" else (
                "TF" if question.qtype == "tf" else "LONG"
            ),
            "stem_text": question.prompt,
            "options": format_question_options(question),
            "marks": question.marks,
            "gold_answer": gold_answer_data.get("correct_answer") or question.correct_answer,
            "gold_confidence": 1.0,
            "answer_explanation": gold_answer_data.get("explanation") or question.explanation or "",
            "source": {
                "dataset": gold_answer_data.get("source_dataset") or question.source_dataset or "",
                "source_id": str(gold_answer_data.get("source_id") or question.source_id or "")
            },
            "has_image": bool(question.image_path),
            "image_path": str(question.image_path) if question.image_path else None
        }
        questions_array.append(question_data)
    
    # Get page count from PDF
    page_count = get_pdf_page_count(pdf_file, compiler)
    
    # Calculate statistics
    stats = {
        "by_type": {},
        "by_marks": {},
        "total_marks_by_type": {}
    }
    
    type_counts = {}
    marks_by_type = {}
    marks_distribution = {}
    
    for q_num, question in all_questions:
        qtype_upper = question.qtype.upper() if question.qtype == "mcq" else (
            "TF" if question.qtype == "tf" else "LONG"
        )
        type_counts[qtype_upper] = type_counts.get(qtype_upper, 0) + 1
        marks_by_type[qtype_upper] = marks_by_type.get(qtype_upper, 0) + question.marks
        marks_distribution[str(question.marks)] = marks_distribution.get(str(question.marks), 0) + 1
    
    stats["by_type"] = type_counts
    stats["by_marks"] = marks_distribution
    stats["total_marks_by_type"] = marks_by_type
    
    # Build final structured JSON
    structured_json = {
        "docid": doc_id,
        "document_name": title,
        "domain": domain,
        "academic_level": level,
        "subjects": subjects,
        "combination_used": combination_used,
        "total_marks": gold_data.get("total_marks", 0),
        "number_of_questions": len(all_questions),
        "number_of_pages": page_count,
        "generated_at": metadata.get("generated_at", "2023-01-01"),
        "version": metadata.get("version", "1.0"),
        "file_paths": {
            "latex_file": str(latex_file),
            "pdf_file": str(pdf_file),
            "metadata_file": f"data/metadata_hierarchical/{doc_id}_metadata.json",
            "gold_labels_file": f"data/gold_labels_hierarchical/{doc_id}_gold.json",
            "answer_key_latex_file": str(answer_key_latex_file) if answer_key_latex_file else None,
            "answer_key_pdf_file": str(answer_key_pdf_file) if answer_key_pdf_file else None,
        },
        "questions": questions_array,
        "question_statistics": stats
    }
    
    return structured_json


def generate_domain_specific_documents(args: argparse.Namespace) -> None:
    """Generate domain-specific question papers."""
    config = get_config(args.config)
    logger = get_logger()

    if not args.skip_download:
        downloader = DatasetDownloader(config, force_download=args.refresh_data)
        downloader.download_all_datasets()

    # Get domain generation settings
    domains_to_generate = config.get_domains_to_generate()
    papers_per_domain = config.get_papers_per_domain_level()
    
    # Create base output directory
    base_output = Path("output")
    metadata_dir = Path("data/metadata_domain_specific")
    gold_dir = Path("data/gold_labels_domain_specific")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    compiler = PDFCompiler(config)
    summary: List[Dict[str, Any]] = []
    
    doc_counter = 1
    
    for domain in domains_to_generate:
        logger.info(f"Generating papers for domain: {domain}")
        
        # Get domain-specific combinations
        domain_combinations = config.get_hierarchical_combination(domain, "K-12")
        if not domain_combinations:
            logger.warning(f"No combinations found for domain: {domain}")
            continue
        
        # Create domain-specific output directories
        domain_output_dir = base_output / f"{domain}_papers"
        domain_latex_dir = domain_output_dir / "latex_documents"
        domain_pdf_dir = domain_output_dir / "pdf_documents"
        domain_answer_latex_dir = domain_output_dir / "answer_keys" / "latex"
        domain_answer_pdf_dir = domain_output_dir / "answer_keys" / "pdf"
        
        domain_latex_dir.mkdir(parents=True, exist_ok=True)
        domain_pdf_dir.mkdir(parents=True, exist_ok=True)
        domain_answer_latex_dir.mkdir(parents=True, exist_ok=True)
        domain_answer_pdf_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created domain directories: {domain_output_dir}")
        
        # Create domain-specific pool
        pool = DatasetPool(seed=args.seed, domain=domain, config=config)
        combo_rng = random.Random(args.seed)
        
        for paper_idx in range(papers_per_domain):
            # Select random combination for this domain
            combination = combo_rng.choice(domain_combinations)
            doc_id = f"{domain}_doc_{paper_idx+1:02d}"
            
            # Build questions using fixed distribution (5 MCQ, 5 TF, 2 Long-form)
            sections: Dict[str, List[Question]] = {}
            
            # Determine available question types from the combination
            has_native_long = any(qtype.endswith("_long") for qtype in combination)
            available_types = ["mcq", "tf", "long"]
            mcq_source = infer_mcq_source_from_combination(combination)
            
            # Get fixed question distribution (always returns 5 MCQ, 5 TF, 2 Long-form)
            question_distribution = generate_dynamic_question_distribution(40, available_types, require_all_types=True)
            
            # Build questions based on fixed distribution
            for qtype, count in question_distribution.items():
                if qtype == "mcq":
                    if mcq_source == "mmlu_pro":
                        sections["mcq"] = build_domain_mmlu_pro_mcq(pool, count, doc_id, domain)
                    elif mcq_source == "arc":
                        sections["mcq"] = build_domain_arc_mcq(pool, count, doc_id, domain)
                    else:
                        sections["mcq"] = build_domain_mcq(pool, count, doc_id, domain)
                elif qtype == "tf":
                    if mcq_source == "mmlu_pro":
                        sections["tf"] = build_domain_mmlu_pro_tf(pool, count, doc_id, domain)
                    elif mcq_source == "arc":
                        sections["tf"] = build_domain_arc_tf(pool, count, doc_id, domain)
                    else:
                        sections["tf"] = build_domain_tf(pool, count, doc_id, domain)
                elif qtype == "long":
                    sections["long"] = build_domain_long(
                        pool,
                        count,
                        doc_id,
                        domain,
                        mcq_source=mcq_source,
                        use_native=has_native_long,
                        allow_synthetic=True,
                    )
            
            # Generate document
            marks = total_marks(sections)
            title = f"{domain.replace('_', ' ').title()} Assessment {paper_idx+1}"
            latex_content = render_latex(title, marks, sections)
            latex_file = domain_latex_dir / f"{doc_id}.tex"
            latex_file.write_text(latex_content, encoding="utf-8")

            pdf_file = compiler.compile_latex_to_pdf(latex_file, output_dir=domain_pdf_dir)
            answer_latex_file, answer_pdf_file = write_answer_key(
                doc_id, title, sections, compiler, domain_answer_latex_dir, domain_answer_pdf_dir
            )

            # Generate gold labels
            gold = {
                "document_id": doc_id,
                "domain": domain,
                "total_marks": marks,
                "answers": [
                    {
                        "question_id": q.qid,
                        "type": q.qtype,
                        "correct_answer": q.correct_answer,
                        "marks": q.marks,
                        "explanation": q.explanation,
                        "source_dataset": q.source_dataset,
                        "source_id": q.source_id,
                    }
                    for questions in sections.values()
                    for q in questions
                ],
            }
            save_json(gold, gold_dir / f"{doc_id}_gold.json")

            # Generate metadata
            metadata = {
                "document_id": doc_id,
                "domain": domain,
                "title": title,
                "combination": combination,
                "question_counts": {k: len(v) for k, v in sections.items()},
                "total_marks": marks,
                "latex_file": str(latex_file.resolve()),
                "pdf_file": str(pdf_file.resolve()),
                "answer_key_latex_file": str(answer_latex_file.resolve()),
                "answer_key_pdf_file": str(answer_pdf_file.resolve()),
                "datasets": {
                    "mmlu_subjects": [],  # Domain-specific documents don't use level-based filtering
                    "gsm8k_used": domain == "mathematics",
                    "mbpp_used": domain in ["computer_science_theory", "cybersecurity", "ai_ml"]
                }
            }
            save_json(metadata, metadata_dir / f"{doc_id}_metadata.json")
            
            summary.append({
                "document_id": doc_id,
                "domain": domain,
                "title": title,
                "total_marks": marks,
                "question_counts": {k: len(v) for k, v in sections.items()},
                "pdf_file": str(pdf_file.resolve()),
                "answer_key_pdf_file": str(answer_pdf_file.resolve())
            })
            
            doc_counter += 1
    
    # Save generation summary
    save_json(summary, base_output / "domain_generation_summary.json")
    logger.info(f"Generated {len(summary)} domain-specific documents")
    logger.info("Domain-specific papers organized in separate folders:")
    for domain in domains_to_generate:
        domain_folder = base_output / f"{domain}_papers"
        if domain_folder.exists():
            latex_count = len(list((domain_folder / "latex_documents").glob("*.tex")))
            pdf_count = len(list((domain_folder / "pdf_documents").glob("*.pdf")))
            logger.info(f"  {domain}: {latex_count} LaTeX files, {pdf_count} PDF files")


def generate_domain_level_papers(args_tuple: Tuple) -> Dict[str, Any]:
    """Generate all papers for a single (domain, level) combination.
    
    This function is designed to be called in parallel by ThreadPoolExecutor.
    It encapsulates all logic for generating papers for one (domain, level) pair.
    
    Args:
        args_tuple: Tuple containing (domain, level, combination, config, args, 
                    output_dirs, compiler)
    
    Returns:
        Dictionary with generation results:
        {
            "domain": str,
            "level": str,
            "successful": int,
            "failed": int,
            "error": Optional[str]
        }
    """
    (domain, level, combination, config, args, output_dirs, compiler) = args_tuple
    
    logger = get_logger()
    error_msg = None
    
    try:
        # Check if combination exists
        if not combination:
            error_msg = "No combination found"
            logger.warning(f"{domain} {level}: {error_msg}")
            return {
                "domain": domain,
                "level": level,
                "successful": 0,
                "failed": 0,
                "error": error_msg
            }
        
        # Check if subjects exist
        subjects = config.get_subjects_for_domain_level(domain, level)
        if not subjects:
            error_msg = "No subjects found"
            logger.warning(f"{domain} {level}: {error_msg}")
            return {
                "domain": domain,
                "level": level,
                "successful": 0,
                "failed": 0,
                "error": error_msg
            }
        
        # Setup output directories
        base_output = output_dirs['base']
        metadata_dir = output_dirs['metadata']
        gold_dir = output_dirs['gold']
        
        level_output_dir = base_output / domain / level.lower()
        level_latex_dir = level_output_dir / "latex_documents"
        level_pdf_dir = level_output_dir / "pdf_documents"
        level_answer_latex_dir = level_output_dir / "answer_keys" / "latex"
        level_answer_pdf_dir = level_output_dir / "answer_keys" / "pdf"
        
        level_latex_dir.mkdir(parents=True, exist_ok=True)
        level_pdf_dir.mkdir(parents=True, exist_ok=True)
        level_answer_latex_dir.mkdir(parents=True, exist_ok=True)
        level_answer_pdf_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique seed per domain-level combination to prevent collisions
        # Hash domain+level to get deterministic but unique seed per worker
        seed_hash = int(hashlib.md5(f"{domain}_{level}_{args.seed}".encode()).hexdigest()[:8], 16) % (2**31)
        unique_seed = args.seed + seed_hash
        
        # Create level-specific dataset pool (reused for all papers in this domain-level)
        # Use unique seed for DatasetPool to ensure thread-safe random state
        pool = DatasetPool(seed=unique_seed, domain=domain, level=level, config=config)
        
        # Use same unique seed for combination selection
        combo_rng = random.Random(unique_seed)
        
        papers_per_domain_level = config.get_papers_per_domain_level()
        
        # Track paper generation results for this domain-level
        papers_successful = 0
        papers_failed = 0
        
        # Generate all papers for this domain-level combination
        for paper_idx in range(papers_per_domain_level):
            try:
                # Select random combination for this domain-level
                selected_combination = combo_rng.choice(combination)
                doc_id = f"{domain}_{level.lower()}_doc_{paper_idx+1:02d}"
                mcq_source = infer_mcq_source_from_combination(selected_combination)
                has_native_long = any(qtype.endswith("_long") for qtype in selected_combination)
                
                # Build questions using fixed distribution (5 MCQ, 5 TF, 2 Long-form)
                sections: Dict[str, List[Question]] = {}
                
                # Get fixed question distribution (always returns 5 MCQ, 5 TF, 2 Long-form)
                question_distribution = generate_dynamic_question_distribution(
                    40,
                    ["mcq", "tf", "long"],
                    require_all_types=True,
                )
                
                # Build questions based on fixed distribution
                for qtype, count in question_distribution.items():
                    if qtype == "mcq":
                        # Find the MCQ type from combination
                        mcq_type = next((q for q in selected_combination if q.endswith("_mcq")), None)
                        if mcq_type and mcq_type.startswith("mmlu_pro_"):
                            sections["mcq"] = build_domain_mmlu_pro_mcq(pool, count, doc_id, domain)
                        elif mcq_type and mcq_type.startswith("arc_"):
                            sections["mcq"] = build_domain_arc_mcq(pool, count, doc_id, domain)
                        elif mcq_source == "mmlu_pro":
                            sections["mcq"] = build_domain_mmlu_pro_mcq(pool, count, doc_id, domain)
                        elif mcq_source == "arc":
                            sections["mcq"] = build_domain_arc_mcq(pool, count, doc_id, domain)
                        else:
                            sections["mcq"] = build_domain_mcq(pool, count, doc_id, domain)
                    elif qtype == "tf":
                        # Check for dataset override in config first
                        override_dataset = config.get_dataset_override(domain, level, "tf")
                        if override_dataset == "pubmedqa":
                            sections["tf"] = build_domain_pubmedqa_tf(pool, count, doc_id, domain)
                        elif override_dataset == "squad":
                            sections["tf"] = build_domain_squad_tf(pool, count, doc_id, domain)
                        elif override_dataset:
                            logger.warning(f"Unknown override dataset '{override_dataset}' for {domain} {level} TF, using default")
                            # Fall through to default logic
                            tf_type = next((q for q in selected_combination if q.endswith("_tf")), None)
                            if tf_type and tf_type.startswith("mmlu_pro_"):
                                sections["tf"] = build_domain_mmlu_pro_tf(pool, count, doc_id, domain)
                            elif tf_type and tf_type.startswith("arc_"):
                                sections["tf"] = build_domain_arc_tf(pool, count, doc_id, domain)
                            elif mcq_source == "mmlu_pro":
                                sections["tf"] = build_domain_mmlu_pro_tf(pool, count, doc_id, domain)
                            elif mcq_source == "arc":
                                sections["tf"] = build_domain_arc_tf(pool, count, doc_id, domain)
                            else:
                                sections["tf"] = build_domain_tf(pool, count, doc_id, domain)
                        else:
                            # No override - use existing logic based on combination
                            tf_type = next((q for q in selected_combination if q.endswith("_tf")), None)
                            if tf_type and tf_type.startswith("mmlu_pro_"):
                                sections["tf"] = build_domain_mmlu_pro_tf(pool, count, doc_id, domain)
                            elif tf_type and tf_type.startswith("arc_"):
                                sections["tf"] = build_domain_arc_tf(pool, count, doc_id, domain)
                            elif mcq_source == "mmlu_pro":
                                sections["tf"] = build_domain_mmlu_pro_tf(pool, count, doc_id, domain)
                            elif mcq_source == "arc":
                                sections["tf"] = build_domain_arc_tf(pool, count, doc_id, domain)
                            else:
                                sections["tf"] = build_domain_tf(pool, count, doc_id, domain)
                    elif qtype == "long":
                        # Check for dataset override in config first
                        override_dataset = config.get_dataset_override(domain, level, "long")
                        if override_dataset == "squad":
                            sections["long"] = build_domain_squad_long(pool, count, doc_id, domain)
                        else:
                            # No override - use existing logic
                            sections["long"] = build_domain_long(
                                pool,
                                count,
                                doc_id,
                                domain,
                                mcq_source=mcq_source,
                                use_native=has_native_long,
                                allow_synthetic=True,
                            )
                
                # Generate document
                marks = total_marks(sections)
                title = f"{domain.replace('_', ' ').title()} - {level} Level Assessment {paper_idx+1}"
                latex_content = render_latex(title, marks, sections)
                latex_file = level_latex_dir / f"{doc_id}.tex"
                latex_file.write_text(latex_content, encoding="utf-8")
                
                pdf_file = compiler.compile_latex_to_pdf(latex_file, output_dir=level_pdf_dir)
                answer_latex_file, answer_pdf_file = write_answer_key(
                    doc_id, title, sections, compiler, level_answer_latex_dir, level_answer_pdf_dir
                )
                
                # Generate gold labels
                gold = {
                    "document_id": doc_id,
                    "domain": domain,
                    "academic_level": level,
                    "total_marks": marks,
                    "subjects": subjects,
                    "combination_used": selected_combination,
                    "answers": [
                        {
                            "question_id": q.qid,
                            "type": q.qtype,
                            "correct_answer": q.correct_answer,
                            "marks": q.marks,
                            "explanation": q.explanation,
                            "source_dataset": q.source_dataset,
                            "source_id": q.source_id,
                        }
                        for questions in sections.values()
                        for q in questions
                    ],
                }
                save_json(gold, gold_dir / f"{doc_id}_gold.json")
                
                # Generate metadata
                metadata = {
                    "document_id": doc_id,
                    "domain": domain,
                    "academic_level": level,
                    "title": title,
                    "total_marks": marks,
                    "subjects": subjects,
                    "combination_used": selected_combination,
                    "latex_file": str(latex_file),
                    "pdf_file": str(pdf_file),
                    "answer_key_latex_file": str(answer_latex_file),
                    "answer_key_pdf_file": str(answer_pdf_file),
                    "generated_at": "2023-01-01",
                    "version": "1.0"
                }
                save_json(metadata, metadata_dir / f"{doc_id}_metadata.json")
                
                # Generate structured JSON output
                json_output_dir = level_output_dir / "JSON_output"
                json_output_dir.mkdir(parents=True, exist_ok=True)
                
                structured_json = generate_structured_json(
                    doc_id=doc_id,
                    title=title,
                    domain=domain,
                    level=level,
                    subjects=subjects,
                    combination_used=selected_combination,
                    sections=sections,
                    gold_data=gold,
                    metadata=metadata,
                    latex_file=latex_file,
                    pdf_file=pdf_file,
                    compiler=compiler,
                    answer_key_latex_file=answer_latex_file,
                    answer_key_pdf_file=answer_pdf_file,
                )
                save_json(structured_json, json_output_dir / f"{doc_id}.json")
                
                papers_successful += 1
                
            except Exception as e:
                papers_failed += 1
                paper_error_msg = str(e)
                if "empty" in paper_error_msg.lower() or "list index out of range" in paper_error_msg.lower():
                    logger.error(
                        f"Error generating {domain} {level} paper {paper_idx+1}: {paper_error_msg}. "
                        f"This usually means no subjects are configured or no dataset items match the configured subjects."
                    )
                else:
                    logger.error(f"Error generating {domain} {level} paper {paper_idx+1}: {paper_error_msg}")
        
        return {
            "domain": domain,
            "level": level,
            "successful": papers_successful,
            "failed": papers_failed,
            "error": error_msg
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Fatal error generating {domain} {level} papers: {error_msg}")
        papers_per_domain_level = config.get_papers_per_domain_level()
        return {
            "domain": domain,
            "level": level,
            "successful": 0,
            "failed": papers_per_domain_level,
            "error": error_msg
        }


def generate_hierarchical_documents(args: argparse.Namespace) -> None:
    """Generate hierarchical documents for all configured domains and academic levels."""
    config = get_config(args.config)
    logger = get_logger()
    
    if not args.skip_download:
        downloader = DatasetDownloader(config, force_download=args.refresh_data)
        downloader.download_all_datasets()
    
    # Get hierarchical generation settings
    domains_and_levels = config.get_domains_and_levels()
    papers_per_domain_level = config.get_papers_per_domain_level()
    
    logger.info("Hierarchical system enabled - generating papers by domain and academic level")
    
    # Create base output directory
    base_output = Path("output")
    metadata_dir = Path("data/metadata_hierarchical")
    gold_dir = Path("data/gold_labels_hierarchical")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    
    # Create shared semaphore to limit concurrent PDF compilations
    compile_semaphore = threading.Semaphore(args.max_compile_jobs)
    compiler = PDFCompiler(config, compile_semaphore=compile_semaphore)
    
    # Build task list: collect all (domain, level) combinations
    tasks = []
    for domain, academic_levels in domains_and_levels.items():
        for level in academic_levels:
            combination = config.get_hierarchical_combination(domain, level)
            if not combination:
                logger.warning(f"Skipping {domain} {level}: No combination found")
                continue
            
            subjects = config.get_subjects_for_domain_level(domain, level)
            if not subjects:
                logger.warning(f"Skipping {domain} {level}: No subjects found")
                continue
            
            task = (
                domain,
                level,
                combination,
                config,
                args,
                {
                    'base': base_output,
                    'metadata': metadata_dir,
                    'gold': gold_dir
                },
                compiler
            )
            tasks.append(task)
    
    total_tasks = len(tasks)
    
    # Calculate worker count with safeguards (more conservative for laptops)
    requested_workers = getattr(args, 'workers', 6)
    max_workers = min(
        requested_workers,
        total_tasks,  # Don't exceed number of tasks
        (os.cpu_count() or 1) * 2,  # Reasonable upper bound (2x CPU cores, more conservative)
        8  # Hard cap to prevent resource exhaustion (lowered from 20)
    )
    
    print(f"\nStarting parallel generation for {len(domains_and_levels)} domains...")
    print(f"Domains: {', '.join(domains_and_levels.keys())}")
    print(f"Papers per domain-level: {papers_per_domain_level}")
    print(f"Total (domain, level) tasks: {total_tasks}")
    print(f"Using {max_workers} parallel workers")
    print("=" * 80)
    
    logger.info(f"Starting parallel generation: {total_tasks} tasks, {max_workers} workers")
    
    # Track generation results
    successful_generations = 0
    failed_generations = 0
    total_attempts = total_tasks
    
    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(generate_domain_level_papers, task): task 
            for task in tasks
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            result = future.result()
            
            successful_generations += result["successful"]
            failed_generations += result["failed"]
            
            # Progress logging
            status = "✓" if result["failed"] == 0 else "⚠"
            logger.info(
                f"[{completed}/{total_tasks}] {status} {result['domain']} {result['level']}: "
                f"{result['successful']} successful, {result['failed']} failed"
            )
            
            if result["error"]:
                logger.warning(f"{result['domain']} {result['level']}: {result['error']}")
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"GENERATION SUMMARY:")
    print(f"   Successful: {successful_generations}")
    print(f"   Failed: {failed_generations}")
    print(f"   Total Attempts: {total_attempts}")
    if total_attempts > 0:
        print(f"   Success Rate: {successful_generations/total_attempts*100:.1f}%")
    else:
        print(f"   Success Rate: N/A (no attempts)")
    print("=" * 80)
    
    if successful_generations > 0:
        logger.info(f"Successfully generated {successful_generations} papers")
    if failed_generations > 0:
        logger.info(f"Skipped {failed_generations} papers due to various issues")


def generate_documents(args: argparse.Namespace) -> None:
    config = get_config(args.config)
    logger = get_logger()

    # Check if hierarchical system is enabled
    if config.is_hierarchical_system_enabled():
        logger.info("Hierarchical system enabled")
        generate_hierarchical_documents(args)
        return

    if not args.skip_download:
        downloader = DatasetDownloader(config, force_download=args.refresh_data)
        downloader.download_all_datasets()

    pool = DatasetPool(seed=args.seed, config=config)
    combos = config.get_question_combinations()
    if not combos:
        raise RuntimeError('No question combinations configured in config.yaml')
    combo_rng = random.Random(args.seed)
    requested = max(0, args.count)

    base_output = Path("output/pdf_gen2")
    latex_dir = base_output / "latex"
    pdf_dir = base_output / "pdf"
    answer_latex_dir = base_output / "answer_keys" / "latex"
    answer_pdf_dir = base_output / "answer_keys" / "pdf"
    metadata_dir = Path("data/metadata")
    gold_dir = Path("data/gold_labels_gen2")
    latex_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    answer_latex_dir.mkdir(parents=True, exist_ok=True)
    answer_pdf_dir.mkdir(parents=True, exist_ok=True)

    # Create shared semaphore to limit concurrent PDF compilations (for consistency, even in sequential mode)
    compile_semaphore = threading.Semaphore(args.max_compile_jobs)
    compiler = PDFCompiler(config, compile_semaphore=compile_semaphore)
    summary: List[Dict[str, Any]] = []

    for index in range(requested):
        combination = combo_rng.choice(combos)
        counts = question_counts(combination)
        doc_id = f"gen2_doc_{index+1:02d}"

        sections: Dict[str, List[Question]] = {}
        if counts["mcq"]:
            sections["mcq"] = build_mcq(pool, counts["mcq"], doc_id)
        if counts["tf"]:
            sections["tf"] = build_tf(pool, counts["tf"], doc_id)
        if counts["long"]:
            sections["long"] = build_long(pool, counts["long"], doc_id)

        marks = total_marks(sections)
        title = f"IntegrityShield Assessment {index+1}"
        latex_content = render_latex(title, marks, sections)
        latex_file = latex_dir / f"{doc_id}.tex"
        latex_file.write_text(latex_content, encoding="utf-8")

        pdf_file = compiler.compile_latex_to_pdf(latex_file, output_dir=pdf_dir)
        answer_latex_file, answer_pdf_file = write_answer_key(
            doc_id, title, sections, compiler, answer_latex_dir, answer_pdf_dir
        )

        gold = {
            "document_id": doc_id,
            "total_marks": marks,
            "answers": [
                {
                    "question_id": q.qid,
                    "type": q.qtype,
                    "correct_answer": q.correct_answer,
                    "marks": q.marks,
                    "explanation": q.explanation,
                    "source_dataset": q.source_dataset,
                    "source_id": q.source_id,
                }
                for questions in sections.values()
                for q in questions
            ],
        }
        save_json(gold, gold_dir / f"{doc_id}_gold.json")

        metadata = {
            "document_id": doc_id,
            "title": title,
            "combination": combination,
            "question_counts": {k: len(v) for k, v in sections.items()},
            "total_marks": marks,
            "latex_file": str(latex_file.resolve()),
            "pdf_file": str(pdf_file.resolve()),
            "answer_key_latex_file": str(answer_latex_file.resolve()),
            "answer_key_pdf_file": str(answer_pdf_file.resolve()),
            "datasets": {
                "mcq": "mmlu_all" if counts["mcq"] else None,
                "tf": "mmlu_all" if counts["tf"] else None,
                "long": "mbpp_plus" if counts["long"] else None,
            },
            "questions": [
                {
                    "question_id": q.qid,
                    "type": q.qtype,
                    "marks": q.marks,
                    "source_dataset": q.source_dataset,
                    "source_id": q.source_id,
                }
                for questions in sections.values()
                for q in questions
            ],
        }
        save_json(metadata, metadata_dir / f"{doc_id}_metadata.json")

        summary.append(
            {
                "document_id": doc_id,
                "combination": combination,
                "total_marks": marks,
                "latex": str(latex_file.resolve()),
                "pdf": str(pdf_file.resolve()),
                "answer_key_pdf": str(answer_pdf_file.resolve()),
            }
        )

    save_json({"documents": summary}, base_output / "generation_summary.json")
    logger.success(f"pdf_gen2 generated {len(summary)} documents")
    logger.info(f"LaTeX directory: {latex_dir}")
    logger.info(f"PDF directory: {pdf_dir}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simple IntegrityShield PDFs (Gen2)")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--count", type=int, default=5, help="Number of documents to generate")
    parser.add_argument("--skip-download", action="store_true", help="Reuse cached datasets")
    parser.add_argument("--refresh-data", action="store_true", help="Force re-download even if cached")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel workers for question generation (default: 6, max: 20)"
    )
    parser.add_argument(
        "--max-compile-jobs",
        type=int,
        default=4,
        help="Maximum number of concurrent PDF compilations (default: 4, max: 8). Limits pdflatex processes to prevent resource exhaustion."
    )
    args = parser.parse_args(argv)
    
    # Validate workers argument
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.workers > 20:
        parser.error("--workers cannot exceed 20")
    
    # Validate max-compile-jobs argument
    if args.max_compile_jobs < 1:
        parser.error("--max-compile-jobs must be at least 1")
    if args.max_compile_jobs > 8:
        parser.error("--max-compile-jobs cannot exceed 8")
    
    return args


def main() -> int:
    args = parse_args()
    config = get_config(args.config)
    logging_config = config.get_logging_config()
    if args.verbose:
        logging_config = dict(logging_config)
        logging_config["level"] = "DEBUG"
    setup_logging(logging_config)
    try:
        generate_documents(args)
        return 0
    except Exception as exc:
        print(f"X IntegrityShield generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
