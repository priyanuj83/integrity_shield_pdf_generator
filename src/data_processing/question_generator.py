"""
Question generator for IntegrityShield system.
Generates different types of questions from downloaded datasets.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger, log_data_processing, log_success, log_error
from ..utils.config import get_config


class QuestionType(Enum):
    """Enumeration of question types."""
    MCQ = "mcq"
    TRUE_FALSE = "tf"
    LONG_FORM = "long"


@dataclass
class Question:
    """Represents a single question."""
    id: str
    type: QuestionType
    text: str
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    marks: int = 1
    difficulty: str = "medium"
    subject: str = "general"


@dataclass
class Document:
    """Represents a complete assessment document."""
    id: str
    title: str
    questions: List[Question]
    total_marks: int
    question_types: List[QuestionType]
    metadata: Dict[str, Any]


class QuestionGenerator:
    """Generates questions from processed datasets."""
    
    def __init__(self, config_manager=None):
        """Initialize question generator."""
        self.config = config_manager or get_config()
        self.logger = get_logger()
        self.data_dir = Path(self.config.get_output_dirs()['raw_data_dir'])
        self.processed_dir = Path(self.config.get_output_dirs()['processed_data_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.datasets = self._load_datasets()
    
    def _load_datasets(self) -> Dict[str, Any]:
        """Load all downloaded datasets."""
        datasets = {}
        
        for dataset_file in self.data_dir.glob("*.json"):
            try:
                with open(dataset_file, 'r') as f:
                    data = json.load(f)
                datasets[dataset_file.stem] = data
                self.logger.info(f"Loaded dataset: {dataset_file.stem} ({data.get('total_items', 0)} items)")
            except Exception as e:
                self.logger.error(f"Error loading dataset {dataset_file}: {e}")
        
        return datasets
    
    def generate_mcq_questions(self, count: int, subject: str = "math") -> List[Question]:
        """Generate multiple choice questions."""
        questions = []
        
        if subject == "math" and "gsm8k_mcq" in self.datasets:
            questions.extend(self._generate_math_mcq(count))
        elif subject == "social_science" and "mmlu" in self.datasets:
            questions.extend(self._generate_social_science_mcq(count))
        else:
            # Fallback to generic MCQ generation
            questions.extend(self._generate_generic_mcq(count, subject))
        
        return questions[:count]
    
    def _generate_math_mcq(self, count: int) -> List[Question]:
        """Generate math MCQ questions from GSM8K data."""
        questions = []
        math_data = self.datasets.get("gsm8k_mcq", {}).get("items", [])
        
        for i, item in enumerate(math_data[:count]):
            # Create question text
            question_text = item.get("question", "")
            
            # Generate options (correct answer + 3 distractors)
            correct_answer = item.get("answer", "")
            options = self._generate_math_distractors(question_text, correct_answer)
            
            question = Question(
                id=f"math_mcq_{i+1}",
                type=QuestionType.MCQ,
                text=question_text,
                options=options,
                correct_answer=correct_answer,
                marks=2,
                difficulty="medium",
                subject="math"
            )
            questions.append(question)
        
        return questions
    
    def _generate_math_distractors(self, question: str, correct_answer: str) -> List[str]:
        """Generate distractors for math questions."""
        # Simple distractor generation - in practice, this would use more sophisticated methods
        try:
            correct_num = float(correct_answer.split()[-1]) if correct_answer.split()[-1].replace('.', '').isdigit() else 0
        except:
            correct_num = 0
        
        distractors = [
            str(correct_num + 1),
            str(correct_num - 1),
            str(correct_num * 2),
            str(correct_num / 2) if correct_num != 0 else "0"
        ]
        
        # Add correct answer
        distractors.append(correct_answer)
        
        # Shuffle and return first 4
        random.shuffle(distractors)
        return distractors[:4]
    
    def _generate_social_science_mcq(self, count: int) -> List[Question]:
        """Generate social science MCQ questions from MMLU data."""
        questions = []
        mmlu_data = self.datasets.get("mmlu", {}).get("items", [])
        
        for i, item in enumerate(mmlu_data[:count]):
            question_text = item.get("question", "")
            choices = item.get("choices", [])
            correct_answer = item.get("answer", "")
            
            if len(choices) >= 4:
                question = Question(
                    id=f"social_mcq_{i+1}",
                    type=QuestionType.MCQ,
                    text=question_text,
                    options=choices,
                    correct_answer=correct_answer,
                    marks=2,
                    difficulty="medium",
                    subject="social_science"
                )
                questions.append(question)
        
        return questions
    
    def _generate_generic_mcq(self, count: int, subject: str) -> List[Question]:
        """Generate generic MCQ questions."""
        questions = []
        
        for i in range(count):
            question = Question(
                id=f"generic_mcq_{i+1}",
                type=QuestionType.MCQ,
                text=f"Sample {subject} question {i+1}?",
                options=["Option A", "Option B", "Option C", "Option D"],
                correct_answer="Option A",
                marks=2,
                difficulty="medium",
                subject=subject
            )
            questions.append(question)
        
        return questions
    
    def generate_true_false_questions(self, count: int, subject: str = "general") -> List[Question]:
        """Generate true/false questions."""
        questions = []
        
        # Generate T/F questions from various sources
        for i in range(count):
            # Create statement and determine if it's true or false
            statement = f"Sample {subject} statement {i+1}."
            is_true = random.choice([True, False])
            
            question = Question(
                id=f"tf_{i+1}",
                type=QuestionType.TRUE_FALSE,
                text=statement,
                correct_answer="True" if is_true else "False",
                explanation=f"This statement is {'correct' if is_true else 'incorrect'}.",
                marks=2,
                difficulty="easy",
                subject=subject
            )
            questions.append(question)
        
        return questions
    
    def generate_long_form_questions(self, count: int, subject: str = "general") -> List[Question]:
        """Generate long-form questions."""
        questions = []
        
        for i in range(count):
            question = Question(
                id=f"long_{i+1}",
                type=QuestionType.LONG_FORM,
                text=f"Explain in detail: {subject} concept {i+1}. Provide examples and reasoning.",
                correct_answer="Sample detailed explanation with examples.",
                marks=5,
                difficulty="hard",
                subject=subject
            )
            questions.append(question)
        
        return questions
    
    def generate_document(self, doc_id: str, question_types: List[QuestionType], 
                         total_marks: int = 40) -> Document:
        """Generate a complete assessment document."""
        self.logger.info(f"Generating document {doc_id} with types: {[t.value for t in question_types]}")
        
        questions = []
        marks_used = 0
        
        # Generate questions for each type
        for q_type in question_types:
            if q_type == QuestionType.MCQ:
                q_count = min(5, (total_marks - marks_used) // 2)
                if q_count > 0:
                    new_questions = self.generate_mcq_questions(q_count)
                    questions.extend(new_questions)
                    marks_used += sum(q.marks for q in new_questions)
            
            elif q_type == QuestionType.TRUE_FALSE:
                q_count = min(5, (total_marks - marks_used) // 2)
                if q_count > 0:
                    new_questions = self.generate_true_false_questions(q_count)
                    questions.extend(new_questions)
                    marks_used += sum(q.marks for q in new_questions)
            
            elif q_type == QuestionType.LONG_FORM:
                q_count = min(3, (total_marks - marks_used) // 5)
                if q_count > 0:
                    new_questions = self.generate_long_form_questions(q_count)
                    questions.extend(new_questions)
                    marks_used += sum(q.marks for q in new_questions)
        
        # Create document
        document = Document(
            id=doc_id,
            title=f"Assessment - {doc_id}",
            questions=questions,
            total_marks=marks_used,
            question_types=question_types,
            metadata={
                "generated_at": "2023-01-01",
                "version": "1.0",
                "subject": "Basic Mathematics"
            }
        )
        
        log_success(f"Document generation: {doc_id}", f"Generated {len(questions)} questions, {marks_used} marks")
        return document
    
    def generate_initial_batch(self) -> List[Document]:
        """Generate the initial batch of 5 documents."""
        self.logger.info("Generating initial batch of 5 documents")
        
        documents = []
        combinations = self.config.get_question_combinations()[:5]  # First 5 combinations
        
        for i, combination in enumerate(combinations):
            doc_id = f"doc_{i+1:02d}"
            question_types = [QuestionType(qt) for qt in combination]
            
            document = self.generate_document(doc_id, question_types)
            documents.append(document)
        
        # Save documents
        self._save_documents(documents)
        
        log_success("Initial batch generation", f"Generated {len(documents)} documents")
        return documents
    
    def _save_documents(self, documents: List[Document]):
        """Save generated documents to files."""
        for document in documents:
            doc_file = self.processed_dir / f"{document.id}.json"
            
            # Convert to serializable format
            doc_data = {
                "id": document.id,
                "title": document.title,
                "total_marks": document.total_marks,
                "question_types": [qt.value for qt in document.question_types],
                "questions": [
                    {
                        "id": q.id,
                        "type": q.type.value,
                        "text": q.text,
                        "options": q.options,
                        "correct_answer": q.correct_answer,
                        "explanation": q.explanation,
                        "marks": q.marks,
                        "difficulty": q.difficulty,
                        "subject": q.subject
                    }
                    for q in document.questions
                ],
                "metadata": document.metadata
            }
            
            with open(doc_file, 'w') as f:
                json.dump(doc_data, f, indent=2)
            
            self.logger.info(f"Saved document: {doc_file}")


def main():
    """Main function for testing question generator."""
    from ..utils.logger import setup_logging
    
    # Setup logging
    config = get_config()
    setup_logging(config.get_logging_config())
    
    # Initialize generator
    generator = QuestionGenerator()
    
    # Generate initial batch
    documents = generator.generate_initial_batch()
    
    # Print results
    print(f"\nGenerated {len(documents)} documents:")
    for doc in documents:
        print(f"- {doc.id}: {len(doc.questions)} questions, {doc.total_marks} marks, types: {[t.value for t in doc.question_types]}")


if __name__ == "__main__":
    main()
