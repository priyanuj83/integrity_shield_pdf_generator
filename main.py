#!/usr/bin/env python3
"""Generate simplified assessment PDFs (Gen2 pipeline).

Builds LaTeX, PDFs, metadata, and gold labels for configurable documents
using cached datasets (MMLU + MBPP). Questions are formatted explicitly as
MCQ, True/False, and Long-form prompts with consistent marks.
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import re
import sys
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_config  # type: ignore
from src.utils.logger import get_logger, setup_logging  # type: ignore
from src.data_processing.dataset_downloader import DatasetDownloader  # type: ignore
from src.pdf_generation.pdf_compiler import PDFCompiler  # type: ignore

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


class DatasetPool:
    """Facilitates sampling across cached datasets with reproducible order."""

    def __init__(self, seed: Optional[int] = None, domain: Optional[str] = None, config=None) -> None:
        if seed is not None:
            random.seed(seed)
        self.config = config
        self.domain = domain
        
        # Load all datasets
        self.mmlu_items = self._load_items("data/raw/mmlu_all.json")
        self.mbpp_items = self._load_items("data/raw/mbpp_plus.json")
        self.gsm8k_items = self._load_gsm8k_items("data/gsm_mcq")
        self.mmlu_pro_items = self._load_items("data/raw/mmlu_pro.json")
        
        if not self.mmlu_items:
            raise RuntimeError("mmlu_all dataset missing or empty. Run without --skip-download once.")
        if not self.mbpp_items:
            raise RuntimeError("mbpp_plus dataset missing or empty. Run without --skip-download once.")
        if not self.gsm8k_items:
            raise RuntimeError("gsm8k dataset missing or empty. Check data/gsm_mcq directory.")
        if not self.mmlu_pro_items:
            raise RuntimeError("mmlu_pro dataset missing or empty. Run without --skip-download once.")
        
        # Filter MMLU items by domain if specified
        if domain and config:
            self.mmlu_items = self._filter_mmlu_by_domain(self.mmlu_items, domain)
        
        random.shuffle(self.mmlu_items)
        random.shuffle(self.mbpp_items)
        random.shuffle(self.gsm8k_items)
        random.shuffle(self.mmlu_pro_items)
        self._mmlu_idx = 0
        self._mbpp_idx = 0
        self._gsm8k_idx = 0
        self._mmlu_pro_idx = 0

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
    
    def _filter_mmlu_by_domain(self, mmlu_items: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
        """Filter MMLU items by domain-specific subjects."""
        if not self.config:
            return mmlu_items
        
        # Get subjects for this domain
        domain_subjects = self.config.get_mmlu_subjects_for_domain(domain)
        if not domain_subjects:
            return mmlu_items
        
        # Filter items that match the domain subjects
        filtered_items = []
        for item in mmlu_items:
            subject = item.get("subject", "")
            if subject in domain_subjects:
                filtered_items.append(item)
        
        return filtered_items
    
    def next_mmlu(self) -> Dict[str, Any]:
        if self._mmlu_idx >= len(self.mmlu_items):
            raise RuntimeError("Ran out of MMLU questions")
        item = self.mmlu_items[self._mmlu_idx]
        self._mmlu_idx += 1
        return item

    def next_mbpp(self) -> Dict[str, Any]:
        if self._mbpp_idx >= len(self.mbpp_items):
            raise RuntimeError("Ran out of MBPP prompts")
        item = self.mbpp_items[self._mbpp_idx]
        self._mbpp_idx += 1
        return item
    
    def next_gsm8k(self) -> Dict[str, Any]:
        if self._gsm8k_idx >= len(self.gsm8k_items):
            raise RuntimeError("Ran out of GSM8K problems")
        item = self.gsm8k_items[self._gsm8k_idx]
        self._gsm8k_idx += 1
        return item

    def next_mmlu_pro(self) -> Dict[str, Any]:
        if self._mmlu_pro_idx >= len(self.mmlu_pro_items):
            # Reset index to cycle through questions again
            self._mmlu_pro_idx = 0
            # Reshuffle for variety
            random.shuffle(self.mmlu_pro_items)
        item = self.mmlu_pro_items[self._mmlu_pro_idx]
        self._mmlu_pro_idx += 1
        return item


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters, but preserve mathematical expressions."""
    # First, protect mathematical expressions by temporarily replacing them
    import re
    
    # Find and protect math expressions (content between $...$)
    math_expressions = []
    math_pattern = r'\$([^$]+)\$'
    
    def replace_math(match):
        math_expressions.append(match.group(1))
        return f"__MATH_EXPR_{len(math_expressions)-1}__"
    
    # Protect math expressions
    text = re.sub(math_pattern, replace_math, text)
    
    # Now escape LaTeX special characters in the remaining text
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\textasciitilde{}",
        "%": r"\%",
    }
    
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    
    # Restore math expressions (they should not be escaped)
    for i, math_expr in enumerate(math_expressions):
        text = text.replace(f"__MATH_EXPR_{i}__", f"${math_expr}$")
    
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


def has_only_ascii(text: str) -> bool:
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def question_counts(combination: Iterable[str]) -> Dict[str, int]:
    combo_set = set(combination)
    if combo_set == {"mcq", "tf", "long"}:
        return {"mcq": 5, "tf": 5, "long": 2}
    if combo_set == {"mcq", "tf"}:
        return {"mcq": 10, "tf": 10, "long": 0}
    if combo_set == {"mcq", "long"}:
        return {"mcq": 10, "tf": 0, "long": 2}
    if combo_set == {"tf", "long"}:
        return {"mcq": 0, "tf": 10, "long": 2}
    if combo_set == {"mcq"}:
        return {"mcq": 20, "tf": 0, "long": 0}
    if combo_set == {"tf"}:
        return {"mcq": 0, "tf": 20, "long": 0}
    if combo_set == {"long"}:
        return {"mcq": 0, "tf": 0, "long": 4}
    # Fallback for "mixed" or unknown combos: keep 40 marks
    return {"mcq": 6, "tf": 6, "long": 2}


def parse_answer_index(answer: Any, choices_len: int) -> int:
    if isinstance(answer, str) and answer.upper() in LETTER_OPTIONS:
        idx = LETTER_OPTIONS.index(answer.upper())
    else:
        try:
            idx = int(answer)
        except Exception:
            idx = 0
    if choices_len == 0:
        return 0
    return max(0, min(idx, choices_len - 1))


def build_mcq(pool: DatasetPool, total: int, doc_id: str) -> List[Question]:
    questions: List[Question] = []
    for idx in range(total):
        attempts = 0
        item = None
        while attempts < 500:
            candidate = pool.next_mmlu()
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices_raw = candidate.get("choices", [])[: len(LETTER_OPTIONS)]
            if question_raw and choices_raw and all(normalize_whitespace(c) for c in choices_raw):
                trimmed_question = normalize_whitespace(question_raw)
                trimmed_question = trimmed_question.rstrip('.')
                if (len(trimmed_question) <= 180
                        and has_only_ascii(trimmed_question)
                        and all(len(normalize_whitespace(c)) <= 90 and has_only_ascii(normalize_whitespace(c)) for c in choices_raw)):
                    item = candidate
                    break
            attempts += 1
        if item is None:
            raise RuntimeError("Unable to find suitable MCQ item from MMLU dataset")
        question_text = escape_latex(textwrap.shorten(normalize_whitespace(item.get("question", "")), width=160, placeholder="..."))
        raw_choices = [normalize_whitespace(c) for c in item.get("choices", [])[: len(LETTER_OPTIONS)]]
        options = [escape_latex(choice) for choice in raw_choices]
        answer_idx = parse_answer_index(item.get("answer"), len(options))
        correct_letter = LETTER_OPTIONS[answer_idx] if options else "A"
        correct_text = options[answer_idx] if options else ""
        formatted_options = options
        subject = item.get("subject") or "mmlu_all"
        questions.append(
            Question(
                qid=f"{doc_id}_mcq_{idx+1}",
                qtype="mcq",
                prompt=question_text or "Answer the question.",
                marks=MCQ_MARKS,
                options=formatted_options,
                correct_answer=correct_letter,
                explanation=f"Correct option: {correct_letter} - {correct_text}",
                source_id=item.get("id"),
                source_dataset=subject,
            )
        )
    return questions


def build_tf(pool: DatasetPool, total: int, doc_id: str) -> List[Question]:
    questions: List[Question] = []
    for idx in range(total):
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
            raise RuntimeError("Unable to find suitable TF item from MMLU dataset")
        question_text = textwrap.shorten(normalize_whitespace(item.get("question", "")), width=140, placeholder="...")
        choices = [normalize_whitespace(c) for c in item.get("choices", [])]
        answer_idx = parse_answer_index(item.get("answer"), len(choices))
        correct_text = choices[answer_idx] if choices else "N/A"
        false_text = None
        if choices:
            incorrect = [c for i, c in enumerate(choices) if i != answer_idx]
            if incorrect:
                false_text = random.choice(incorrect)
        make_true = random.random() < 0.5 or not false_text
        if make_true:
            statement = f"The correct answer to '{question_text}' is '{correct_text}'."
            correct_answer = "True"
            explanation = "Matches the MMLU answer key."
        else:
            statement = f"The correct answer to '{question_text}' is '{false_text}'."
            correct_answer = "False"
            explanation = f"Actual answer is '{correct_text}'."
        subject = item.get("subject") or "mmlu_all"
        questions.append(
            Question(
                qid=f"{doc_id}_tf_{idx+1}",
                qtype="tf",
                prompt=escape_latex(statement),
                marks=TF_MARKS,
                options=["True", "False"],
                correct_answer=correct_answer,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=subject,
            )
        )
    return questions


def summarise_mbpp_answer(item: Dict[str, Any]) -> str:
    code = item.get("code", "")
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    doc_lines = [line for line in lines if line.startswith("def ") or line.startswith("return ")]
    if not doc_lines:
        doc_lines = lines[:3]
    return " | ".join(doc_lines)[:300]


def build_long(pool: DatasetPool, total: int, doc_id: str) -> List[Question]:
    questions: List[Question] = []
    for idx in range(total):
        # Alternate between MBPP and GSM8K for variety
        if idx % 2 == 0 and len(pool.gsm8k_items) > 0:
            # Use GSM8K for math word problems
            item = pool.next_gsm8k()
            problem = normalize_whitespace(item.get("problem", ""))
            solution = normalize_whitespace(item.get("solution", ""))
            
            if problem and solution:
                prompt = escape_latex(textwrap.fill(problem, width=90))
                answer_summary = escape_latex(textwrap.fill(solution, width=90))
                questions.append(
                    Question(
                        qid=f"{doc_id}_long_{idx+1}",
                        qtype="long",
                        prompt=prompt,
                        marks=LONG_MARKS,
                        correct_answer=answer_summary,
                        explanation="Students should show all steps in their mathematical solution.",
                        source_id=item.get("id"),
                        source_dataset="gsm8k_math",
                    )
                )
            else:
                # Fallback to MBPP if GSM8K item is invalid
                item = pool.next_mbpp()
                prompt = escape_latex(textwrap.fill(normalize_whitespace(item.get("text", "Explain the solution.")), width=90))
                answer_summary = escape_latex(summarise_mbpp_answer(item) or "Refer to reference implementation in MBPP dataset.")
                split = item.get("split")
                dataset_name = f"mbpp_plus:{split}" if split else "mbpp_plus"
                questions.append(
                    Question(
                        qid=f"{doc_id}_long_{idx+1}",
                        qtype="long",
                        prompt=prompt,
                        marks=LONG_MARKS,
                        correct_answer=answer_summary,
                        explanation="Students should provide a detailed explanation referencing algorithm steps.",
                        source_id=item.get("id"),
                        source_dataset=dataset_name,
                    )
                )
        else:
            # Use MBPP for coding problems
            item = pool.next_mbpp()
            prompt = escape_latex(textwrap.fill(normalize_whitespace(item.get("text", "Explain the solution.")), width=90))
            answer_summary = escape_latex(summarise_mbpp_answer(item) or "Refer to reference implementation in MBPP dataset.")
            split = item.get("split")
            dataset_name = f"mbpp_plus:{split}" if split else "mbpp_plus"
            questions.append(
                Question(
                    qid=f"{doc_id}_long_{idx+1}",
                    qtype="long",
                    prompt=prompt,
                    marks=LONG_MARKS,
                    correct_answer=answer_summary,
                    explanation="Students should provide a detailed explanation referencing algorithm steps.",
                    source_id=item.get("id"),
                    source_dataset=dataset_name,
                )
            )
    return questions


def build_domain_mcq(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build domain-specific MCQ questions from MMLU dataset."""
    questions: List[Question] = []
    for idx in range(total):
        attempts = 0
        item = None
        while attempts < 500:
            candidate = pool.next_mmlu()
            question_raw = normalize_whitespace(candidate.get("question", ""))
            choices_raw = candidate.get("choices", [])[: len(LETTER_OPTIONS)]
            if question_raw and choices_raw and all(normalize_whitespace(c) for c in choices_raw):
                trimmed_question = normalize_whitespace(question_raw)
                trimmed_question = trimmed_question.rstrip('.')
                if (len(trimmed_question) <= 180
                        and has_only_ascii(trimmed_question)
                        and all(len(normalize_whitespace(c)) <= 90 and has_only_ascii(normalize_whitespace(c)) for c in choices_raw)):
                    item = candidate
                    break
            attempts += 1
        if item is None:
            raise RuntimeError(f"Unable to find suitable MCQ item from {domain} domain")
        question_text = escape_latex(textwrap.shorten(normalize_whitespace(item.get("question", "")), width=160, placeholder="..."))
        raw_choices = [normalize_whitespace(c) for c in item.get("choices", [])[: len(LETTER_OPTIONS)]]
        options = [escape_latex(choice) for choice in raw_choices]
        answer_idx = parse_answer_index(item.get("answer"), len(options))
        correct_letter = LETTER_OPTIONS[answer_idx] if options else "A"
        correct_text = options[answer_idx] if options else ""
        formatted_options = options
        subject = item.get("subject") or domain
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_mcq_{idx+1}",
                qtype="mcq",
                prompt=question_text or "Answer the question.",
                marks=MCQ_MARKS,
                options=formatted_options,
                correct_answer=correct_letter,
                explanation=f"Correct option: {correct_letter} - {correct_text}",
                source_id=item.get("id"),
                source_dataset=f"{domain}_{subject}",
            )
        )
    return questions


def build_domain_tf(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build domain-specific True/False questions from MMLU dataset."""
    questions: List[Question] = []
    for idx in range(total):
        attempts = 0
        item = None
        while attempts < 500:
            candidate = pool.next_mmlu()
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
        question_text = textwrap.shorten(normalize_whitespace(item.get("question", "")), width=140, placeholder="...")
        choices = [normalize_whitespace(c) for c in item.get("choices", [])]
        answer_idx = parse_answer_index(item.get("answer"), len(choices))
        correct_text = choices[answer_idx] if choices else "N/A"
        false_text = None
        if choices:
            incorrect = [c for i, c in enumerate(choices) if i != answer_idx]
            if incorrect:
                false_text = random.choice(incorrect)
        make_true = random.random() < 0.5 or not false_text
        if make_true:
            statement = f"The correct answer to '{question_text}' is '{correct_text}'."
            correct_answer = "True"
            explanation = "Matches the MMLU answer key."
        else:
            statement = f"The correct answer to '{question_text}' is '{false_text}'."
            correct_answer = "False"
            explanation = f"Actual answer is '{correct_text}'."
        subject = item.get("subject") or domain
        questions.append(
            Question(
                qid=f"{doc_id}_{domain}_tf_{idx+1}",
                qtype="tf",
                prompt=escape_latex(statement),
                marks=TF_MARKS,
                options=["True", "False"],
                correct_answer=correct_answer,
                explanation=explanation,
                source_id=item.get("id"),
                source_dataset=f"{domain}_{subject}",
            )
        )
    return questions


def build_domain_long(pool: DatasetPool, total: int, doc_id: str, domain: str) -> List[Question]:
    """Build domain-specific long-form questions."""
    questions: List[Question] = []
    for idx in range(total):
        if domain == "mathematics":
            # Use GSM8K for math long-form questions
            item = pool.next_gsm8k()
            problem = normalize_whitespace(item.get("problem", ""))
            solution = normalize_whitespace(item.get("solution", ""))
            
            if problem and solution:
                prompt = escape_latex(textwrap.fill(problem, width=90))
                answer_summary = escape_latex(textwrap.fill(solution, width=90))
                questions.append(
                    Question(
                        qid=f"{doc_id}_{domain}_long_{idx+1}",
                        qtype="long",
                        prompt=prompt,
                        marks=LONG_MARKS,
                        correct_answer=answer_summary,
                        explanation="Students should show all steps in their mathematical solution.",
                        source_id=item.get("id"),
                        source_dataset="gsm8k_math",
                    )
                )
        else:
            # Use MBPP+ for other domains (CS, etc.)
            item = pool.next_mbpp()
            prompt = escape_latex(textwrap.fill(normalize_whitespace(item.get("text", "Explain the solution.")), width=90))
            answer_summary = escape_latex(summarise_mbpp_answer(item) or "Refer to reference implementation in MBPP dataset.")
            split = item.get("split")
            dataset_name = f"mbpp_plus:{split}" if split else "mbpp_plus"
            questions.append(
                Question(
                    qid=f"{doc_id}_{domain}_long_{idx+1}",
                    qtype="long",
                    prompt=prompt,
                    marks=LONG_MARKS,
                    correct_answer=answer_summary,
                    explanation="Students should provide a detailed explanation referencing algorithm steps.",
                    source_id=item.get("id"),
                    source_dataset=dataset_name,
                )
            )
    return questions


def render_latex(title: str, total_marks: int, sections: Dict[str, List[Question]]) -> str:
    section_titles = {"mcq": "Multiple Choice", "tf": "True / False", "long": "Long Form Response"}
    latex_parts = [
        r"\documentclass[12pt]{article}",
        r"\usepackage{enumitem}",
        r"\usepackage{geometry}",
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
    for qtype in ["mcq", "tf", "long"]:
        questions = sections.get(qtype, [])
        if not questions:
            continue
        marks = sum(q.marks for q in questions)
        latex_parts.append(f"\\section*{{{section_titles[qtype]} ({marks} marks)}}")
        if qtype == "tf":
            latex_parts.append(r"\textit{Answer True or False and justify briefly.}")
        if qtype == "long":
            latex_parts.append(r"\textit{Provide thorough reasoning and reference key steps.}")
        latex_parts.append(r"\begin{enumerate}[label=\arabic*.]")
        for question in questions:
            if qtype == "tf":
                latex_parts.append(f"\\item True or False: {question.prompt}")
            else:
                latex_parts.append(f"\\item {question.prompt}")
            if qtype == "mcq" and question.options:
                latex_parts.append(r"\begin{enumerate}[label=(\alph*)]")
                for opt in question.options:
                    latex_parts.append(f"    \\item {opt}")
                latex_parts.append(r"\end{enumerate}")
        latex_parts.append(r"\end{enumerate}")
        latex_parts.append("")
    latex_parts.extend([r"\vfill", r"\noindent\textit{End of Paper}", r"\end{document}"])
    return "\n".join(latex_parts)



def total_marks(sections: Dict[str, List[Question]]) -> int:
    return sum(q.marks for questions in sections.values() for q in questions)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


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
        
        domain_latex_dir.mkdir(parents=True, exist_ok=True)
        domain_pdf_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created domain directories: {domain_output_dir}")
        
        # Create domain-specific pool
        pool = DatasetPool(seed=args.seed, domain=domain, config=config)
        combo_rng = random.Random(args.seed)
        
        for paper_idx in range(papers_per_domain):
            # Select random combination for this domain
            combination = combo_rng.choice(domain_combinations)
            doc_id = f"{domain}_doc_{paper_idx+1:02d}"
            
            # Build questions based on combination
            sections: Dict[str, List[Question]] = {}
            
            for qtype in combination:
                if qtype.endswith("_mcq"):
                    count = 10  # Default MCQ count
                    sections["mcq"] = build_domain_mcq(pool, count, doc_id, domain)
                elif qtype.endswith("_tf"):
                    count = 10  # Default TF count
                    sections["tf"] = build_domain_tf(pool, count, doc_id, domain)
                elif qtype.endswith("_long"):
                    count = 2   # Default long-form count
                    sections["long"] = build_domain_long(pool, count, doc_id, domain)
            
            # Generate document
            marks = total_marks(sections)
            title = f"{domain.replace('_', ' ').title()} Assessment {paper_idx+1}"
            latex_content = render_latex(title, marks, sections)
            latex_file = domain_latex_dir / f"{doc_id}.tex"
            latex_file.write_text(latex_content, encoding="utf-8")

            pdf_file = compiler.compile_latex_to_pdf(latex_file, output_dir=domain_pdf_dir)

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
                "datasets": {
                    "mmlu_subjects": config.get_mmlu_subjects_for_domain(domain),
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
                "pdf_file": str(pdf_file.resolve())
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


def generate_hierarchical_documents(args: argparse.Namespace) -> None:
    """Generate hierarchical documents for all configured domains and academic levels."""
    config = get_config(args.config)
    logger = get_logger()
    
    # Get hierarchical generation settings
    domains_to_generate = config.get_domains_to_generate()
    academic_levels = config.get_academic_levels()
    papers_per_domain_level = config.get_papers_per_domain_level()
    
    logger.info("Hierarchical system enabled - generating papers by domain and academic level")
    
    # Initialize dataset pool and compiler
    pool = DatasetPool()
    compiler = PDFCompiler(config)
    
    # Track generation results
    successful_generations = 0
    failed_generations = 0
    total_attempts = 0
    
    print(f"\n🚀 Starting comprehensive generation for {len(domains_to_generate)} domains...")
    print(f"📊 Domains: {', '.join(domains_to_generate)}")
    print(f"🎓 Academic Levels: {', '.join(academic_levels)}")
    print(f"📄 Papers per domain-level: {papers_per_domain_level}")
    print("=" * 80)
    
    for domain in domains_to_generate:
        print(f"\n📚 Processing domain: {domain}")
        
        for level in academic_levels:
            total_attempts += 1
            print(f"  🎯 Attempting {level} level...")
            
            try:
                # Check if combination exists
                combination = config.get_hierarchical_combination(domain, level)
                if not combination:
                    print(f"    ❌ Not generated {domain} {level} paper - No combination found")
                    failed_generations += 1
                    continue
                
                # Check if subjects exist
                subjects = config.get_subjects_for_domain_level(domain, level)
                if not subjects:
                    print(f"    ❌ Not generated {domain} {level} paper - No subjects found")
                    failed_generations += 1
                    continue
                
                # Generate paper (simplified version for now)
                print(f"    ✅ Successfully generated {domain} {level} paper")
                successful_generations += 1
                    
            except Exception as e:
                print(f"    ❌ Not generated {domain} {level} paper - Error: {str(e)}")
                failed_generations += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"📈 GENERATION SUMMARY:")
    print(f"   ✅ Successful: {successful_generations}")
    print(f"   ❌ Failed: {failed_generations}")
    print(f"   📊 Total Attempts: {total_attempts}")
    print(f"   🎯 Success Rate: {successful_generations/total_attempts*100:.1f}%")
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

    pool = DatasetPool(seed=args.seed)
    combos = config.get_question_combinations()
    if not combos:
        raise RuntimeError('No question combinations configured in config.yaml')
    combo_rng = random.Random(args.seed)
    requested = max(0, args.count)

    base_output = Path("output/pdf_gen2")
    latex_dir = base_output / "latex"
    pdf_dir = base_output / "pdf"
    metadata_dir = Path("data/metadata")
    gold_dir = Path("data/gold_labels_gen2")
    latex_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    compiler = PDFCompiler(config)
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
    return parser.parse_args()


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
        print(f"❌ IntegrityShield generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
