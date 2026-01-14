"""
Compute dataset statistics for the IntegrityShield generated paper corpus.

This script aggregates:
- number of generated papers (JSON outputs, PDFs, metadata)
- domain-wise and education-level-wise paper distribution
- per-dataset counts: (# papers that used dataset, # questions sourced from dataset)

It writes a citation-ready markdown report and several CSVs into `docs/`.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def esc(x: object) -> str:
        s = "" if x is None else str(x)
        return s.replace("\n", " ").replace("|", "\\|")

    head = "| " + " | ".join(esc(h) for h in headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(esc(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body])


@dataclass(frozen=True)
class DocRecord:
    docid: str
    domain: str
    academic_level: str
    question_count: int
    datasets_used: Tuple[str, ...]  # unique, sorted


def main() -> None:
    # ---------------------------------------------------------------------
    # Input discovery
    # ---------------------------------------------------------------------
    meta_files = glob.glob("data/metadata_hierarchical/*_metadata.json")
    json_files = glob.glob("output/**/JSON_output/*.json", recursive=True)
    pdf_files = glob.glob("output/**/pdf_documents/*.pdf", recursive=True)

    # ---------------------------------------------------------------------
    # Load metadata IDs (for consistency checks)
    # ---------------------------------------------------------------------
    meta_ids: Set[str] = set()
    for p in meta_files:
        d = _read_json(p)
        if d.get("document_id"):
            meta_ids.add(d["document_id"])

    # ---------------------------------------------------------------------
    # Load structured JSON outputs (authoritative for questions/provenance)
    # ---------------------------------------------------------------------
    docs: List[DocRecord] = []
    questions_total = 0
    questions_by_dataset: Counter[str] = Counter()
    papers_by_dataset: Counter[str] = Counter()

    json_ids: Set[str] = set()
    for p in json_files:
        d = _read_json(p)
        docid = d.get("docid") or d.get("document_id") or d.get("id")
        domain = d.get("domain")
        level = d.get("academic_level")

        if not docid or not domain or not level:
            # If this ever triggers, the file is malformed relative to the schema we observed.
            # We skip it rather than crash to keep the report generation robust.
            continue

        qs = d.get("questions", []) or []
        qcount = len(qs)
        questions_total += qcount

        per_doc_datasets: Set[str] = set()
        for q in qs:
            src = q.get("source") or {}
            ds = src.get("dataset")
            if not ds:
                continue
            questions_by_dataset[ds] += 1
            per_doc_datasets.add(ds)

        for ds in per_doc_datasets:
            papers_by_dataset[ds] += 1

        json_ids.add(docid)
        docs.append(
            DocRecord(
                docid=str(docid),
                domain=str(domain),
                academic_level=str(level),
                question_count=qcount,
                datasets_used=tuple(sorted(per_doc_datasets)),
            )
        )

    # ---------------------------------------------------------------------
    # PDFs (deliverables) IDs for cross-checks
    # ---------------------------------------------------------------------
    pdf_ids: Set[str] = set(os.path.splitext(os.path.basename(p))[0] for p in pdf_files)

    # ---------------------------------------------------------------------
    # Aggregations (papers)
    # ---------------------------------------------------------------------
    papers_total = len(docs)
    papers_by_domain = Counter(d.domain for d in docs)
    papers_by_level = Counter(d.academic_level for d in docs)
    papers_by_domain_level = Counter((d.domain, d.academic_level) for d in docs)

    avg_questions_per_paper = (questions_total / papers_total) if papers_total else 0.0

    # ---------------------------------------------------------------------
    # Consistency / missing artefacts
    # ---------------------------------------------------------------------
    meta_no_json = sorted(meta_ids - json_ids)
    pdf_no_json = sorted(pdf_ids - json_ids)

    # ---------------------------------------------------------------------
    # Output files
    # ---------------------------------------------------------------------
    _ensure_dir("docs")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # CSV: domains
    with open("docs/dataset_statistics_domains.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "paper_count", "paper_percent_of_json_outputs"])
        for dom, cnt in papers_by_domain.most_common():
            w.writerow([dom, cnt, f"{_pct(cnt, papers_total):.2f}"])

    # CSV: levels
    with open("docs/dataset_statistics_levels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["academic_level", "paper_count", "paper_percent_of_json_outputs"])
        for lvl, cnt in papers_by_level.most_common():
            w.writerow([lvl, cnt, f"{_pct(cnt, papers_total):.2f}"])

    # CSV: domain x level
    with open("docs/dataset_statistics_domain_x_level.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "academic_level", "paper_count"])
        for (dom, lvl), cnt in sorted(papers_by_domain_level.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            w.writerow([dom, lvl, cnt])

    # CSV: datasets
    with open("docs/dataset_statistics_datasets.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset_label",
                "question_count",
                "question_percent_of_all_questions",
                "paper_count_using_dataset",
                "paper_percent_of_all_papers",
            ]
        )
        for ds, qcnt in questions_by_dataset.most_common():
            pcnt = papers_by_dataset.get(ds, 0)
            w.writerow(
                [
                    ds,
                    qcnt,
                    f"{_pct(qcnt, questions_total):.2f}",
                    pcnt,
                    f"{_pct(pcnt, papers_total):.2f}",
                ]
            )

    # Markdown report
    # Summary block
    summary_rows = [
        ["Generated at (UTC)", generated_at],
        ["Papers (JSON_output, includes questions/provenance)", papers_total],
        ["Total questions (from JSON_output)", questions_total],
        ["Avg questions per paper", f"{avg_questions_per_paper:.2f}"],
        ["PDF papers present (pdf_documents)", len(pdf_ids)],
        ["Metadata entries present (data/metadata_hierarchical)", len(meta_ids)],
        ["Metadata-but-missing-JSON_output (count)", len(meta_no_json)],
        ["PDF-but-missing-JSON_output (count)", len(pdf_no_json)],
        ["Unique dataset labels in provenance (source.dataset)", len(questions_by_dataset)],
    ]

    domain_rows = [
        [dom, cnt, f"{_pct(cnt, papers_total):.2f}%"] for dom, cnt in papers_by_domain.most_common()
    ]
    level_rows = [
        [lvl, cnt, f"{_pct(cnt, papers_total):.2f}%"] for lvl, cnt in papers_by_level.most_common()
    ]

    # Domain x level table
    levels_order = ["K-12", "Undergraduate", "Graduate"]
    doms_order = [dom for dom, _ in papers_by_domain.most_common()]
    dxl_rows: List[List[object]] = []
    for dom in doms_order:
        row = [dom]
        total = 0
        for lvl in levels_order:
            c = papers_by_domain_level.get((dom, lvl), 0)
            row.append(c)
            total += c
        row.append(total)
        dxl_rows.append(row)

    # Top datasets table (keep readable in MD, full list in CSV)
    top_datasets = questions_by_dataset.most_common(25)
    top_ds_rows = []
    for ds, qcnt in top_datasets:
        pcnt = papers_by_dataset.get(ds, 0)
        top_ds_rows.append(
            [ds, qcnt, f"{_pct(qcnt, questions_total):.2f}%", pcnt, f"{_pct(pcnt, papers_total):.2f}%"]
        )

    md = []
    md.append("# Dataset statistics (generated artefacts)\n")
    md.append(
        "This report was generated by `scripts/compute_dataset_statistics.py` by aggregating the on-disk artefacts:\n"
        "- `output/**/JSON_output/*.json` (paper structure + per-question `source.dataset` provenance)\n"
        "- `output/**/pdf_documents/*.pdf` (deliverable PDFs)\n"
        "- `data/metadata_hierarchical/*_metadata.json` (per-paper metadata)\n"
    )

    md.append("## Summary\n")
    md.append(_md_table(["Metric", "Value"], summary_rows))
    md.append("\n")

    md.append("## Paper distribution by domain (JSON_output papers)\n")
    md.append(_md_table(["Domain", "Paper count", "Share of papers"], domain_rows))
    md.append("\n")

    md.append("## Paper distribution by education level (JSON_output papers)\n")
    md.append(_md_table(["Education level", "Paper count", "Share of papers"], level_rows))
    md.append("\n")

    md.append("## Paper distribution by domain × education level (JSON_output papers)\n")
    md.append(_md_table(["Domain", *levels_order, "Total"], dxl_rows))
    md.append("\n")

    md.append("## Per-dataset usage (from per-question provenance)\n")
    md.append(
        "Counts below treat each unique value of `question.source.dataset` as a dataset label.\n"
        "- **Paper count using dataset**: number of papers that contain ≥1 question with that `source.dataset`.\n"
        "- **Question count**: total number of questions whose `source.dataset` equals that label.\n"
        "\n"
        "The full dataset table is saved as `docs/dataset_statistics_datasets.csv`; the top 25 are shown here.\n"
    )
    md.append(
        _md_table(
            ["Dataset label (`source.dataset`)", "Questions", "Share of questions", "Papers", "Share of papers"],
            top_ds_rows,
        )
    )
    md.append("\n")

    md.append("## Notes on missing structured JSON outputs\n")
    md.append(
        f"- Metadata present but missing `output/**/JSON_output/*.json`: **{len(meta_no_json)}** papers\n"
        f"- PDF present but missing `output/**/JSON_output/*.json`: **{len(pdf_no_json)}** papers\n"
        "\n"
        "These can happen if a run produced metadata/PDFs but did not serialize the structured JSON output (or files were removed).\n"
    )
    if meta_no_json:
        md.append("\n### Metadata present but JSON_output missing (docids)\n")
        md.append("\n".join([f"- `{x}`" for x in meta_no_json]))
        md.append("\n")
    if pdf_no_json:
        md.append("\n### PDFs present but JSON_output missing (docids)\n")
        md.append("\n".join([f"- `{x}`" for x in pdf_no_json]))
        md.append("\n")

    with open("docs/dataset_statistics.md", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md).strip() + "\n")

    print("Wrote:")
    print(" - docs/dataset_statistics.md")
    print(" - docs/dataset_statistics_domains.csv")
    print(" - docs/dataset_statistics_levels.csv")
    print(" - docs/dataset_statistics_domain_x_level.csv")
    print(" - docs/dataset_statistics_datasets.csv")


if __name__ == "__main__":
    main()


