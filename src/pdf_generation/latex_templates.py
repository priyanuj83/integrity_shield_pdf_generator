"""
LaTeX template system for IntegrityShield PDF generation.
Creates LaTeX documents from question data using templates.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from string import Template

from ..utils.logger import get_logger, log_success, log_error
from ..utils.config import get_config


class LaTeXTemplates:
    """Manages LaTeX templates for document generation."""
    
    def __init__(self, config_manager=None):
        """Initialize LaTeX templates."""
        self.config = config_manager or get_config()
        self.logger = get_logger()
        self.template_dir = Path("templates")
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        # Create base template
        self._create_base_template()
    
    def _create_base_template(self):
        """Create the base LaTeX template."""
        base_template = r"""
\documentclass[12pt]{article}
\usepackage{enumitem}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{amsmath, amssymb}
\geometry{margin=1in}

\begin{document}

\begin{center}
\Large \textbf{Assessment - $assessment_number} \\[0.5cm]
\large Term: $term \quad Subject: $subject \quad Course Number: $course_number \\[0.3cm]
\large Total Marks: $total_marks \\
\textit{Answer all questions with reasoning.}
\end{center}

\vspace{0.25cm}

$sections

\vfill
\noindent\textit{End of Paper}

\end{document}
"""
        
        template_file = self.template_dir / "base_template.tex"
        with open(template_file, 'w') as f:
            f.write(base_template)
        
        self.logger.info(f"Created base template: {template_file}")
    
    def generate_mcq_section(self, questions: List[Dict], section_title: str = "MCQ with Reason") -> str:
        """Generate LaTeX for MCQ section."""
        latex = f"\\section*{{{section_title} ({sum(q.get('marks', 1) for q in questions)} marks)}}\n\n"
        latex += "\\begin{enumerate}[label=\\arabic*.]\n\n"
        
        for question in questions:
            latex += f"\\item {question['text']}:\n"
            latex += "\\begin{enumerate}[label=(\\alph*)]\n"
            
            for option in question.get('options', []):
                latex += f"    \\item {option} \\qquad "
            
            latex += "\n\\end{enumerate}\n\n"
        
        latex += "\\end{enumerate}\n"
        return latex
    
    def generate_tf_section(self, questions: List[Dict], section_title: str = "True/False with Reason") -> str:
        """Generate LaTeX for True/False section."""
        latex = f"\\section*{{{section_title} ({sum(q.get('marks', 1) for q in questions)} marks)}}\n"
        latex += "\\textit{Write True or False in your answer file and include a brief reason.}\n\n"
        latex += "\\begin{enumerate}[label=\\arabic*.]\n\n"
        
        for question in questions:
            latex += f"\\item {question['text']}\n\n"
        
        latex += "\\end{enumerate}\n"
        return latex
    
    def generate_long_section(self, questions: List[Dict], section_title: str = "Solve") -> str:
        """Generate LaTeX for long-form section."""
        latex = f"\\section*{{{section_title} ({sum(q.get('marks', 1) for q in questions)} marks)}}\n"
        latex += "\\textit{Show all steps clearly in your solution file. Each question carries equal marks unless indicated.}\n\n"
        latex += "\\begin{enumerate}[label=\\arabic*.]\n\n"
        
        for question in questions:
            marks = question.get('marks', 1)
            latex += f"\\item {question['text']}  \\hfill ({marks} marks)\n\n"
        
        latex += "\\end{enumerate}\n"
        return latex
    
    def generate_document_latex(self, document_data: Dict) -> str:
        """Generate complete LaTeX document from document data."""
        self.logger.info(f"Generating LaTeX for document: {document_data['id']}")
        
        # Load base template
        template_file = self.template_dir / "base_template.tex"
        with open(template_file, 'r') as f:
            template = Template(f.read())
        
        # Generate sections
        sections = []
        questions = document_data.get('questions', [])
        
        # Group questions by type
        mcq_questions = [q for q in questions if q.get('type') == 'mcq']
        tf_questions = [q for q in questions if q.get('type') == 'tf']
        long_questions = [q for q in questions if q.get('type') == 'long']
        
        # Generate sections
        if mcq_questions:
            sections.append(self.generate_mcq_section(mcq_questions))
        
        if tf_questions:
            sections.append(self.generate_tf_section(tf_questions))
        
        if long_questions:
            sections.append(self.generate_long_section(long_questions))
        
        # Get LaTeX configuration
        latex_config = self.config.get_latex_config()
        
        # Fill template
        latex_content = template.substitute(
            assessment_number=document_data['id'].replace('doc_', ''),
            term=latex_config.term,
            subject=latex_config.subject,
            course_number=latex_config.course_number,
            total_marks=document_data.get('total_marks', 40),
            sections='\n'.join(sections)
        )
        
        log_success(f"LaTeX generation: {document_data['id']}", f"Generated {len(sections)} sections")
        return latex_content
    
    def save_latex_document(self, document_data: Dict, output_dir: Path) -> Path:
        """Save LaTeX document to file."""
        latex_content = self.generate_document_latex(document_data)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LaTeX file
        latex_file = output_dir / f"{document_data['id']}.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_content)
        
        self.logger.info(f"Saved LaTeX document: {latex_file}")
        return latex_file
    
    def generate_gold_labels(self, document_data: Dict) -> Dict[str, Any]:
        """Generate gold labels (answer key) for a document."""
        self.logger.info(f"Generating gold labels for document: {document_data['id']}")
        
        gold_labels = {
            "document_id": document_data['id'],
            "title": f"Answer Key - {document_data['id']}",
            "total_marks": document_data.get('total_marks', 40),
            "answers": []
        }
        
        for question in document_data.get('questions', []):
            answer_entry = {
                "question_id": question['id'],
                "type": question['type'],
                "correct_answer": question.get('correct_answer', ''),
                "marks": question.get('marks', 1),
                "explanation": question.get('explanation', '')
            }
            gold_labels["answers"].append(answer_entry)
        
        log_success(f"Gold labels generation: {document_data['id']}", f"Generated {len(gold_labels['answers'])} answers")
        return gold_labels
    
    def save_gold_labels(self, gold_labels: Dict, output_dir: Path) -> Path:
        """Save gold labels to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gold_file = output_dir / f"{gold_labels['document_id']}_gold.json"
        with open(gold_file, 'w') as f:
            json.dump(gold_labels, f, indent=2)
        
        self.logger.info(f"Saved gold labels: {gold_file}")
        return gold_file
    
    def process_document(self, document_data, latex_output_dir: Path, gold_output_dir: Path) -> Dict[str, Path]:
        """Process a complete document (LaTeX + gold labels)."""
        # Handle both Document objects and dictionaries
        if hasattr(document_data, 'id'):
            doc_id = document_data.id
            doc_dict = {
                'id': document_data.id,
                'title': document_data.title,
                'questions': [
                    {
                        'id': q.id,
                        'type': q.type.value if hasattr(q.type, 'value') else str(q.type),
                        'text': q.text,
                        'options': getattr(q, 'options', []),
                        'correct_answer': getattr(q, 'correct_answer', ''),
                        'explanation': getattr(q, 'explanation', ''),
                        'marks': getattr(q, 'marks', 1)
                    } for q in document_data.questions
                ]
            }
        else:
            doc_id = document_data['id']
            doc_dict = document_data
        
        self.logger.info(f"Processing document: {doc_id}")
        
        # Generate and save LaTeX
        latex_file = self.save_latex_document(doc_dict, latex_output_dir)
        
        # Generate and save gold labels
        gold_labels = self.generate_gold_labels(doc_dict)
        gold_file = self.save_gold_labels(gold_labels, gold_output_dir)
        
        return {
            "latex_file": latex_file,
            "gold_file": gold_file
        }
    
    def process_multiple_documents(self, documents, latex_output_dir: Path, gold_output_dir: Path) -> List[Dict[str, Path]]:
        """Process multiple documents."""
        self.logger.info(f"Processing {len(documents)} documents")
        
        results = []
        for document_data in documents:
            try:
                result = self.process_document(document_data, latex_output_dir, gold_output_dir)
                results.append(result)
            except Exception as e:
                doc_id = getattr(document_data, 'id', 'unknown')
                log_error(e, f"Processing document {doc_id}")
        
        log_success(f"Document processing", f"Processed {len(results)} documents successfully")
        return results


def main():
    """Main function for testing LaTeX templates."""
    from ..utils.logger import setup_logging
    from ..data_processing.question_generator import QuestionGenerator
    
    # Setup logging
    config = get_config()
    setup_logging(config.get_logging_config())
    
    # Initialize components
    question_generator = QuestionGenerator()
    latex_templates = LaTeXTemplates()
    
    # Generate sample document
    sample_doc = question_generator.generate_document(
        "test_doc", 
        [QuestionType.MCQ, QuestionType.TRUE_FALSE], 
        20
    )
    
    # Convert to dict format
    doc_data = {
        "id": sample_doc.id,
        "title": sample_doc.title,
        "total_marks": sample_doc.total_marks,
        "questions": [
            {
                "id": q.id,
                "type": q.type.value,
                "text": q.text,
                "options": q.options,
                "correct_answer": q.correct_answer,
                "marks": q.marks,
                "explanation": q.explanation
            }
            for q in sample_doc.questions
        ]
    }
    
    # Process document
    latex_output = Path("output/latex_documents")
    gold_output = Path("output/gold_labels")
    
    result = latex_templates.process_document(doc_data, latex_output, gold_output)
    
    print(f"Generated LaTeX: {result['latex_file']}")
    print(f"Generated gold labels: {result['gold_file']}")


if __name__ == "__main__":
    main()
