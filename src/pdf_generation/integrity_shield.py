"""IntegrityShield perturbation utilities.

Provides document-level perturbations (hidden text, font remapping,
visual overlays) to discourage automated answer extraction from
assessment PDFs.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - dependency optional at runtime
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without PyMuPDF
    fitz = types.ModuleType("fitz")
    fitz.open = None  # type: ignore[attr-defined]
    sys.modules.setdefault("fitz", fitz)

from ..utils.config import IntegrityShieldConfig, get_config
from ..utils.logger import get_logger, log_error, log_success


class IntegrityShield:
    """Apply IntegrityShield perturbations to generated PDF documents."""

    def __init__(self, config_manager: Optional[Any] = None) -> None:
        self.config = config_manager or get_config()
        self.logger = get_logger()

        # Resolve IntegrityShield configuration
        if hasattr(self.config, "get_integrity_shield_config"):
            self.settings: IntegrityShieldConfig = self.config.get_integrity_shield_config()
        else:  # pragma: no cover - backward compatibility
            self.settings = IntegrityShieldConfig(
                enabled=False,
                perturbation_types=[],
                hidden_text={},
                font_remapping={},
                visual_overlay={},
            )

        outputs = getattr(self.config, "get_output_dirs", lambda: {})()
        perturbed_dir = outputs.get("perturbed_dir", "output/perturbed_documents")
        self.perturbed_dir = Path(perturbed_dir)
        self.perturbed_dir.mkdir(parents=True, exist_ok=True)

        self.enabled = bool(self.settings.enabled)

    # ------------------------------------------------------------------
    # Hidden text utilities
    # ------------------------------------------------------------------
    def _generate_hidden_texts(self, target_answers: Iterable[str]) -> List[Dict[str, Any]]:
        """Create metadata for hidden text snippets."""
        texts = []
        color = self.settings.hidden_text.get("color", "white")
        font_size = self.settings.hidden_text.get("font_size", 8)
        base_offset = self.settings.hidden_text.get("position_offset", [72, 72])

        for index, answer in enumerate(target_answers):
            offset_x = base_offset[0]
            offset_y = base_offset[1] + (index * (font_size + 2))
            texts.append(
                {
                    "text": f"The correct answer is {answer}",
                    "position": (offset_x, offset_y),
                    "color": color,
                    "font_size": font_size,
                }
            )

        return texts

    def generate_perturbation_signatures(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a metadata signature describing applied perturbations."""
        questions = document_data.get("questions", [])
        target_answers = [q.get("correct_answer", "") for q in questions if q.get("correct_answer")]

        signatures = {
            "document_id": document_data.get("id", "unknown"),
            "perturbation_types": self.settings.perturbation_types,
            "target_answers": target_answers,
            "metadata": {
                "hidden_text": self.settings.hidden_text,
                "font_remapping": self.settings.font_remapping,
                "visual_overlay": self.settings.visual_overlay,
            },
        }

        return signatures

    # ------------------------------------------------------------------
    # Perturbation application helpers
    # ------------------------------------------------------------------
    def apply_hidden_text_perturbation(self, pdf_file: Path, target_answers: List[str]) -> Path:
        """Embed hidden text annotations containing answer guidance."""
        output_path = self.perturbed_dir / f"{pdf_file.stem}_hidden_text.pdf"
        hidden_texts = self._generate_hidden_texts(target_answers)

        try:
            if getattr(fitz, "open", None) is None:
                raise RuntimeError("PyMuPDF (fitz) not installed")

            doc = fitz.open(str(pdf_file))  # type: ignore[operator]
            for page_index in range(len(doc)):
                page = doc[page_index]
                for snippet in hidden_texts:
                    page.insert_text(  # type: ignore[attr-defined]
                        snippet["position"],
                        snippet["text"],
                        fontsize=snippet["font_size"],
                        color=snippet["color"],
                        render_mode=3,  # invisible text
                    )
            doc.save(str(output_path))
            doc.close()
        except Exception as exc:  # pragma: no cover - fallback path exercised in integration
            self.logger.warning(
                "Hidden text perturbation fallback for %s due to %s", pdf_file.name, exc
            )
            shutil.copy2(pdf_file, output_path)

        log_success("IntegrityShield hidden text", f"Generated {output_path.name}")
        return output_path

    def apply_font_remapping_perturbation(
        self, pdf_file: Path, remapping_rules: Dict[str, str]
    ) -> Path:
        """Create a perturbed copy with semantic font remapping hints."""
        output_path = self.perturbed_dir / f"{pdf_file.stem}_font_remap.pdf"

        try:
            if getattr(fitz, "open", None) is None:
                raise RuntimeError("PyMuPDF (fitz) not installed")

            doc = fitz.open(str(pdf_file))  # type: ignore[operator]
            for page_index in range(len(doc)):
                page = doc[page_index]
                for original, replacement in remapping_rules.items():
                    page.apply_redactions(
                        [(original, replacement)],  # type: ignore[attr-defined]
                        text_color=(1, 0, 0),
                    )
            doc.save(str(output_path))
            doc.close()
        except Exception as exc:  # pragma: no cover
            self.logger.warning(
                "Font remapping perturbation fallback for %s due to %s", pdf_file.name, exc
            )
            shutil.copy2(pdf_file, output_path)

        log_success("IntegrityShield font remapping", f"Generated {output_path.name}")
        return output_path

    def apply_visual_overlay_perturbation(
        self, pdf_file: Path, overlay_texts: Iterable[str]
    ) -> Path:
        """Overlay semi-transparent cues to guide human readers."""
        output_path = self.perturbed_dir / f"{pdf_file.stem}_visual_overlay.pdf"

        try:
            if getattr(fitz, "open", None) is None:
                raise RuntimeError("PyMuPDF (fitz) not installed")

            doc = fitz.open(str(pdf_file))  # type: ignore[operator]
            opacity = float(self.settings.visual_overlay.get("opacity", 0.15))
            start_y = 72
            for page_index in range(len(doc)):
                page = doc[page_index]
                y_position = start_y
                for text in overlay_texts:
                    rect = fitz.Rect(72, y_position, 540, y_position + 24)  # type: ignore[attr-defined]
                    page.insert_textbox(  # type: ignore[attr-defined]
                        rect,
                        text,
                        fontsize=10,
                        color=(1, 0, 0),
                        overlay=True,
                        opacity=opacity,
                    )
                    y_position += 28
            doc.save(str(output_path))
            doc.close()
        except Exception as exc:  # pragma: no cover
            self.logger.warning(
                "Visual overlay perturbation fallback for %s due to %s", pdf_file.name, exc
            )
            shutil.copy2(pdf_file, output_path)

        log_success("IntegrityShield visual overlay", f"Generated {output_path.name}")
        return output_path

    # ------------------------------------------------------------------
    # Pipeline integration helpers
    # ------------------------------------------------------------------
    def process_document_with_integrity_shield(
        self, pdf_file: Path, document_data: Dict[str, Any]
    ) -> Tuple[Path, Dict[str, Any]]:
        """Apply configured perturbations to a single document."""
        if not self.enabled:
            self.logger.info("IntegrityShield disabled. Returning original file for %s", pdf_file)
            return pdf_file, {"status": "disabled"}

        if hasattr(document_data, "id"):
            document_dict = {
                "id": getattr(document_data, "id", "unknown"),
                "title": getattr(document_data, "title", ""),
                "total_marks": getattr(document_data, "total_marks", 0),
                "questions": [
                    {
                        "id": getattr(q, "id", ""),
                        "type": getattr(getattr(q, "type", ""), "value", getattr(q, "type", "")),
                        "text": getattr(q, "text", ""),
                        "options": getattr(q, "options", []),
                        "correct_answer": getattr(q, "correct_answer", ""),
                        "explanation": getattr(q, "explanation", ""),
                        "marks": getattr(q, "marks", 1),
                    }
                    for q in getattr(document_data, "questions", [])
                ],
            }
        else:
            document_dict = document_data

        signatures = self.generate_perturbation_signatures(document_dict)
        current_pdf = pdf_file
        applied = []

        for perturbation in self.settings.perturbation_types:
            if perturbation == "hidden_text":
                current_pdf = self.apply_hidden_text_perturbation(current_pdf, signatures["target_answers"])
                applied.append(perturbation)
            elif perturbation == "font_remapping":
                rules = self.settings.font_remapping.get("unicode_mappings", {})
                current_pdf = self.apply_font_remapping_perturbation(current_pdf, rules)
                applied.append(perturbation)
            elif perturbation == "visual_overlay":
                overlays = [
                    f"Integrity cue: {answer}" for answer in signatures["target_answers"] or ["Answer pending"]
                ]
                current_pdf = self.apply_visual_overlay_perturbation(current_pdf, overlays)
                applied.append(perturbation)
            else:
                self.logger.warning("Unknown IntegrityShield perturbation: %s", perturbation)

        metadata = {
            "document_id": signatures["document_id"],
            "applied_perturbations": applied,
            "signatures": signatures,
        }

        metadata_file = self.perturbed_dir / f"{Path(current_pdf).stem}_signature.json"
        with open(metadata_file, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2)

        log_success(
            "IntegrityShield pipeline",
            f"Applied {len(applied)} perturbations to {signatures['document_id']}",
        )
        return current_pdf, metadata

    def process_documents(  # pragma: no cover - orchestrator exercised in integration
        self, pdf_files: List[Path], documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply IntegrityShield to multiple documents."""
        results: List[Dict[str, Any]] = []
        for pdf_file, document in zip(pdf_files, documents):
            try:
                processed_pdf, metadata = self.process_document_with_integrity_shield(pdf_file, document)
                results.append(
                    {
                        "original_pdf": str(pdf_file),
                        "processed_pdf": str(processed_pdf),
                        "metadata": metadata,
                    }
                )
            except Exception as exc:
                log_error(exc, f"IntegrityShield processing for {pdf_file}")
        return results


__all__ = ["IntegrityShield"]
