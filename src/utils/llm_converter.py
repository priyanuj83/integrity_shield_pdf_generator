"""
LLM-based converter for transforming MCQ questions to long-form and True/False questions.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError

from src.utils.logger import get_logger


class LLMConverter:
    """Converts MCQ questions to long-form and True/False questions using OpenAI API."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM converter with configuration.
        
        Args:
            config: LLM configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = get_logger()
        
        # Load API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or export it as an environment variable."
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 200)
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30)
        
        # Caching
        self.enable_caching = config.get("enable_caching", True)
        self.cache_file = Path(config.get("cache_file", "data/cache/llm_conversions.json"))
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, str]:
        """Load conversion cache from file."""
        if not self.enable_caching:
            return {}
        
        if not self.cache_file.exists():
            # Create cache directory if it doesn't exist
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                self.logger.debug(f"Loaded {len(cache)} entries from cache")
                return cache
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning(f"Error loading cache: {e}. Starting with empty cache.")
            return {}
    
    def _save_cache(self):
        """Save conversion cache to file."""
        if not self.enable_caching:
            return
        
        try:
            # Ensure cache directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except IOError as e:
            self.logger.warning(f"Error saving cache: {e}")
    
    def _generate_cache_key(self, mcq_stem: str, domain: str, academic_level: str) -> str:
        """Generate cache key from MCQ stem, domain, and academic level."""
        key_string = f"{mcq_stem}|{domain}|{academic_level}"
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()
    
    def _call_openai_api(self, prompt: str) -> Optional[str]:
        """
        Call OpenAI API with retry logic and error handling.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            Response text or None if all retries fail
        """
        for attempt in range(self.max_retries):
            self.logger.info(f"Making OpenAI API call (model: {self.model}, attempt: {attempt+1}/{self.max_retries})")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert educator converting multiple-choice questions into long-form essay questions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                
                # Extract response text
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content.strip()
                    if content:
                        self.logger.info(f"OpenAI API call successful (model: {self.model})")
                        return content
                    else:
                        self.logger.warning("Empty response from API")
                        return None
                else:
                    self.logger.warning("No choices in API response")
                    return None
                    
            except RateLimitError as e:
                wait_time = (2 ** attempt) * 2  # Exponential backoff
                self.logger.warning(f"Rate limit exceeded (model: {self.model}, attempt: {attempt+1}/{self.max_retries}). Waiting {wait_time} seconds before retry")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Rate limit error after {self.max_retries} attempts (model: {self.model}): {e}")
                    return None
                    
            except APITimeoutError as e:
                self.logger.warning(f"API timeout on attempt {attempt + 1}/{self.max_retries} (model: {self.model}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"API timeout after {self.max_retries} attempts (model: {self.model})")
                    return None
                    
            except APIError as e:
                self.logger.error(f"OpenAI API error (model: {self.model}, attempt: {attempt+1}/{self.max_retries}): {e}")
                return None
                
            except Exception as e:
                self.logger.error(f"Unexpected error calling OpenAI API (model: {self.model}, attempt: {attempt+1}/{self.max_retries}): {e}")
                return None
        
        return None
    
    def _generate_prompt(self, mcq_stem: str, correct_answer: str, domain: str, academic_level: str) -> str:
        """
        Generate prompt for converting MCQ to long-form question.
        
        Args:
            mcq_stem: The MCQ question stem
            correct_answer: The correct answer option
            domain: The domain/subject area
            academic_level: The academic level (K-12, Undergraduate, Graduate)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Convert this multiple-choice question into a well-framed long-form question.

MCQ Question: {mcq_stem}
Domain: {domain}
Academic Level: {academic_level}
Correct Answer: {correct_answer}

Convert this MCQ question into a well-framed long-form question that:
1. Is appropriate for {domain} domain at {academic_level} level
2. Asks students to explain, analyze, or discuss (not just recall)
3. Is clear, concise (under 200 characters), and academically appropriate
4. Does not include the multiple-choice options
5. Encourages critical thinking and detailed explanation
6. Maintains the core topic and learning objective

Return ONLY the converted long-form question text. Do not include explanations, markdown formatting, or any other text."""
        
        return prompt
    
    def convert_mcq_to_long_form(
        self, 
        mcq_stem: str, 
        correct_answer: str, 
        domain: str, 
        academic_level: str = "K-12"
    ) -> Optional[str]:
        """
        Convert MCQ question to long-form question using LLM.
        
        Args:
            mcq_stem: The MCQ question stem
            correct_answer: The correct answer option
            domain: The domain/subject area
            academic_level: The academic level (default: "K-12")
            
        Returns:
            Converted long-form question or None if conversion fails
        """
        self.logger.info(f"Converting MCQ to long-form for {domain} {academic_level}")
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._generate_cache_key(mcq_stem, domain, academic_level)
            if cache_key in self.cache:
                self.logger.info(f"Cache hit for long-form conversion (domain: {domain})")
                return self.cache[cache_key]
        
        # Generate prompt
        prompt = self._generate_prompt(mcq_stem, correct_answer, domain, academic_level)
        
        # Call API
        self.logger.info(f"Calling OpenAI API for long-form conversion (domain: {domain}, model: {self.model})")
        converted_question = self._call_openai_api(prompt)
        
        if converted_question:
            # Clean up the response (remove any markdown formatting, extra whitespace)
            converted_question = converted_question.strip()
            # Remove markdown code blocks if present
            if converted_question.startswith("```"):
                lines = converted_question.split("\n")
                converted_question = "\n".join(lines[1:-1]).strip()
            
            # Save to cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(mcq_stem, domain, academic_level)
                self.cache[cache_key] = converted_question
                self._save_cache()
            
            self.logger.info(f"Successfully converted MCQ to long-form (domain: {domain})")
            return converted_question
        else:
            self.logger.info(f"LLM long-form conversion failed (domain: {domain})")
            return None
    
    def _generate_long_answer_prompt(
        self, 
        long_form_question: str, 
        correct_answer_text: str, 
        domain: str, 
        academic_level: str
    ) -> str:
        """
        Generate prompt for creating a long-form answer.
        
        Args:
            long_form_question: The long-form question text
            correct_answer_text: The correct answer (from original MCQ option)
            domain: The domain/subject area
            academic_level: The academic level (K-12, Undergraduate, Graduate)
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Generate a concise but complete answer to this long-form question.

Question: {long_form_question}
Domain: {domain}
Academic Level: {academic_level}
Key Answer/Concept: {correct_answer_text}

Write a clear, well-structured answer that:
1. Directly addresses the question
2. Incorporates the key answer/concept provided
3. Provides brief supporting explanation or reasoning
4. Is appropriate for {academic_level} level in {domain}
5. Is 4-6 sentences long (thorough but focused)

Return ONLY the answer text. Do not include any prefixes like "Answer:" or markdown formatting."""
        
        return prompt

    def generate_long_form_answer(
        self,
        long_form_question: str,
        correct_answer_text: str,
        domain: str,
        academic_level: str = "K-12"
    ) -> Optional[str]:
        """
        Generate a meaningful long-form answer using LLM.
        
        Args:
            long_form_question: The long-form question text
            correct_answer_text: The correct answer (from original MCQ option)
            domain: The domain/subject area
            academic_level: The academic level (default: "K-12")
            
        Returns:
            Generated long-form answer or None if generation fails
        """
        self.logger.info(f"Generating long-form answer for {domain} {academic_level}")
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._generate_cache_key(
                f"long_answer::{long_form_question}::{correct_answer_text}", 
                domain, 
                academic_level
            )
            if cache_key in self.cache:
                self.logger.info(f"Cache hit for long-form answer (domain: {domain})")
                return self.cache[cache_key]
        
        # Generate prompt
        prompt = self._generate_long_answer_prompt(
            long_form_question, correct_answer_text, domain, academic_level
        )
        
        # Call API
        self.logger.info(f"Calling OpenAI API for long-form answer (domain: {domain}, model: {self.model})")
        generated_answer = self._call_openai_api(prompt)
        
        if generated_answer:
            # Clean up the response
            generated_answer = generated_answer.strip()
            # Remove markdown code blocks if present
            if generated_answer.startswith("```"):
                lines = generated_answer.split("\n")
                generated_answer = "\n".join(lines[1:-1]).strip()
            # Remove "Answer:" prefix if present
            if generated_answer.lower().startswith("answer:"):
                generated_answer = generated_answer[7:].strip()
            
            # Save to cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(
                    f"long_answer::{long_form_question}::{correct_answer_text}", 
                    domain, 
                    academic_level
                )
                self.cache[cache_key] = generated_answer
                self._save_cache()
            
            self.logger.info(f"Successfully generated long-form answer (domain: {domain})")
            return generated_answer
        else:
            self.logger.warning(f"LLM long-form answer generation failed (domain: {domain})")
            return None
    
    def _generate_explanation_prompt(
        self,
        question_text: str,
        answer: str,
        question_type: str,
        domain: str,
        academic_level: str
    ) -> str:
        """
        Generate prompt for creating an answer explanation.
        
        Args:
            question_text: The question text
            answer: The correct answer
            question_type: Type of question (MCQ, TF, LONG)
            domain: The domain/subject area
            academic_level: The academic level (K-12, Undergraduate, Graduate)
            
        Returns:
            Formatted prompt string
        """
        if question_type == "TF":
            prompt = f"""Explain why the following True/False statement has the given answer.

Statement: {question_text}
Correct Answer: {answer}
Domain: {domain}
Academic Level: {academic_level}

Write a concise 1 sentence explanation of why this statement is {answer}. Focus on the key concept or fact that makes it true or false.

Return ONLY the explanation. Do not include prefixes like "Explanation:" or markdown formatting."""
        elif question_type == "LONG":
            prompt = f"""Explain why the following answer is correct for the given long-form question.

Question: {question_text}
Answer: {answer}
Domain: {domain}
Academic Level: {academic_level}

Write a concise 1 sentence explanation highlighting the key concepts or reasoning that make this answer correct.

Return ONLY the explanation. Do not include prefixes like "Explanation:" or markdown formatting."""
        else:  # MCQ
            prompt = f"""Explain why the following answer is correct for the given multiple choice question.

Question: {question_text}
Correct Answer: {answer}
Domain: {domain}
Academic Level: {academic_level}

Write a concise 1 sentence explanation of why this is the correct answer. Include the key concept, fact, or reasoning that supports this choice.

Return ONLY the explanation. Do not include prefixes like "Explanation:" or markdown formatting."""
        
        return prompt

    def generate_answer_explanation(
        self,
        question_text: str,
        answer: str,
        question_type: str = "MCQ",
        domain: str = "general",
        academic_level: str = "K-12"
    ) -> Optional[str]:
        """
        Generate a concise explanation for why an answer is correct.
        
        Args:
            question_text: The question text
            answer: The correct answer
            question_type: Type of question (MCQ, TF, LONG)
            domain: The domain/subject area
            academic_level: The academic level (default: "K-12")
            
        Returns:
            Generated explanation (1-2 sentences) or None if generation fails
        """
        self.logger.info(f"Generating answer explanation for {question_type} in {domain} {academic_level}")
        
        # Check cache first
        if self.enable_caching:
            cache_key = self._generate_cache_key(
                f"explanation::{question_type}::{question_text}::{answer}", 
                domain, 
                academic_level
            )
            if cache_key in self.cache:
                self.logger.info(f"Cache hit for answer explanation (domain: {domain})")
                return self.cache[cache_key]
        
        # Generate prompt
        prompt = self._generate_explanation_prompt(
            question_text, answer, question_type, domain, academic_level
        )
        
        # Call API
        self.logger.info(f"Calling OpenAI API for answer explanation (domain: {domain}, type: {question_type}, model: {self.model})")
        generated_explanation = self._call_openai_api(prompt)
        
        if generated_explanation:
            # Clean up the response
            generated_explanation = generated_explanation.strip()
            # Remove markdown code blocks if present
            if generated_explanation.startswith("```"):
                lines = generated_explanation.split("\n")
                generated_explanation = "\n".join(lines[1:-1]).strip()
            # Remove common prefixes if present
            for prefix in ["Explanation:", "Answer:", "The answer is"]:
                if generated_explanation.lower().startswith(prefix.lower()):
                    generated_explanation = generated_explanation[len(prefix):].strip()
            
            # Save to cache
            if self.enable_caching:
                cache_key = self._generate_cache_key(
                    f"explanation::{question_type}::{question_text}::{answer}", 
                    domain, 
                    academic_level
                )
                self.cache[cache_key] = generated_explanation
                self._save_cache()
            
            self.logger.info(f"Successfully generated answer explanation (domain: {domain}, type: {question_type})")
            return generated_explanation
        else:
            self.logger.warning(f"LLM answer explanation generation failed (domain: {domain}, type: {question_type})")
            return None
    
    def _generate_tf_prompt(
        self,
        mcq_stem: str,
        choices: List[str],
        correct_answer: str,
        domain: str,
        academic_level: str,
    ) -> str:
        """
        Generate prompt for converting MCQ to True/False statements.
        
        Args:
            mcq_stem: The MCQ question stem
            choices: List of all answer choices
            correct_answer: The correct answer option
            domain: The domain/subject area
            academic_level: The academic level (K-12, Undergraduate, Graduate)
            
        Returns:
            Formatted prompt string
        """
        choices_text = "\n".join([f"- {choice}" for choice in choices])
        prompt = f"""You are an expert {domain} educator. Convert this multiple-choice question into two well-framed True/False statements.

MCQ Question: {mcq_stem}
Answer Choices:
{choices_text}
Correct Answer: {correct_answer}
Domain: {domain}
Academic Level: {academic_level}

Create two declarative statements:
1. A TRUE statement that accurately reflects the concept being tested
2. A FALSE statement that is plausible but incorrect (not obviously wrong)

Requirements:
- Each statement should be a clear, standalone declarative sentence
- Statements should be appropriate for {academic_level} level in {domain}
- Keep each statement under 160 characters
- The TRUE statement should accurately represent the correct answer
- The FALSE statement should be a plausible misconception or incorrect fact
- Statements should make sense independently (not reference "the question above")
- Use clear, academic language appropriate for the level

Return your response as valid JSON with exactly these keys:
{{
    "true_statement": "Your true statement here",
    "false_statement": "Your false statement here"
}}

Return ONLY the JSON object, no other text or markdown formatting."""
        
        return prompt
    
    def convert_mcq_to_tf(
        self,
        mcq_stem: str,
        choices: List[str],
        correct_answer: str,
        domain: str,
        academic_level: str = "K-12",
    ) -> Optional[Dict[str, str]]:
        """
        Convert MCQ question to True/False statements using LLM.
        
        Args:
            mcq_stem: The MCQ question stem
            choices: List of all answer choices
            correct_answer: The correct answer option
            domain: The domain/subject area
            academic_level: The academic level (default: "K-12")
            
        Returns:
            Dictionary with "true_statement" and "false_statement" keys, or None if conversion fails
        """
        self.logger.info(f"Converting MCQ to TF for {domain} {academic_level}")
        
        # Generate cache key (include choices to ensure uniqueness)
        if self.enable_caching:
            cache_key = self._generate_cache_key(
                f"tf::{mcq_stem}::{','.join(choices)}", domain, academic_level
            )
            if cache_key in self.cache:
                self.logger.info(f"Cache hit for TF conversion (domain: {domain}, level: {academic_level})")
                cached_result = self.cache[cache_key]
                # Handle both string (old format) and dict (new format) cache entries
                if isinstance(cached_result, dict):
                    return cached_result
                elif isinstance(cached_result, str):
                    # Legacy cache entry, try to parse as JSON
                    try:
                        return json.loads(cached_result)
                    except:
                        pass
        
        # Generate prompt
        prompt = self._generate_tf_prompt(mcq_stem, choices, correct_answer, domain, academic_level)
        
        self.logger.info(f"Calling OpenAI API for TF conversion (domain: {domain}, level: {academic_level}, model: {self.model})")
        
        # Call API with updated system message for TF conversion
        original_system_message = "You are an expert educator converting multiple-choice questions into long-form essay questions."
        tf_system_message = f"You are an expert {domain} educator converting multiple-choice questions into well-framed True/False statements."
        
        # Temporarily override system message for TF conversion
        for attempt in range(self.max_retries):
            self.logger.info(f"Making OpenAI API call (model: {self.model}, attempt: {attempt+1}/{self.max_retries})")
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": tf_system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout
                )
                
                if response.choices and len(response.choices) > 0:
                    content = response.choices[0].message.content.strip()
                    if content:
                        self.logger.info(f"OpenAI API responded successfully for TF conversion (domain: {domain})")
                        # Try to extract JSON from response
                        # Remove markdown code blocks if present
                        if content.startswith("```"):
                            lines = content.split("\n")
                            content = "\n".join(lines[1:-1]).strip()
                        if content.startswith("```json"):
                            lines = content.split("\n")
                            content = "\n".join(lines[1:-1]).strip()
                        
                        # Parse JSON response
                        try:
                            statements = json.loads(content)
                            if isinstance(statements, dict) and "true_statement" in statements and "false_statement" in statements:
                                true_stmt = statements["true_statement"].strip()
                                false_stmt = statements["false_statement"].strip()
                                
                                if true_stmt and false_stmt:
                                    # Validate length
                                    if len(true_stmt) > 160:
                                        self.logger.warning(f"True statement too long ({len(true_stmt)} chars), truncating")
                                        true_stmt = true_stmt[:157] + "..."
                                    if len(false_stmt) > 160:
                                        self.logger.warning(f"False statement too long ({len(false_stmt)} chars), truncating")
                                        false_stmt = false_stmt[:157] + "..."
                                    
                                    result = {
                                        "true_statement": true_stmt,
                                        "false_statement": false_stmt
                                    }
                                    
                                    self.logger.info(f"Successfully parsed TF statements from API response (domain: {domain})")
                                    
                                    # Save to cache
                                    if self.enable_caching:
                                        cache_key = self._generate_cache_key(
                                            f"tf::{mcq_stem}::{','.join(choices)}", domain, academic_level
                                        )
                                        self.cache[cache_key] = result
                                        self._save_cache()
                                        self.logger.info(f"Saved TF conversion to cache (domain: {domain})")
                                    
                                    return result
                                else:
                                    self.logger.warning("Empty statements in TF conversion response")
                                    self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
                                    return None
                            else:
                                self.logger.warning(f"Invalid TF conversion response structure: {statements}")
                                self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
                                return None
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse TF conversion JSON: {e}. Raw response: {content}")
                            self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
                            return None
                    else:
                        self.logger.warning("Empty response from API for TF conversion")
                        self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
                        return None
                else:
                    self.logger.warning("No choices in API response for TF conversion")
                    self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
                    return None
                    
            except RateLimitError as e:
                wait_time = (2 ** attempt) * 2
                self.logger.warning(f"Rate limit exceeded for TF conversion. Waiting {wait_time} seconds before retry {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Rate limit error after {self.max_retries} attempts for TF conversion: {e}")
                    self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
                    return None
                    
            except APITimeoutError as e:
                self.logger.warning(f"API timeout on attempt {attempt + 1}/{self.max_retries} for TF conversion: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.logger.error(f"API timeout after {self.max_retries} attempts for TF conversion")
                    self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
                    return None
                    
            except APIError as e:
                self.logger.error(f"OpenAI API error for TF conversion: {e}")
                self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
                return None
                
            except Exception as e:
                self.logger.error(f"Unexpected error calling OpenAI API for TF conversion: {e}")
                self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
                return None
        
        self.logger.info(f"LLM TF conversion failed, returning None (domain: {domain})")
        return None

