import pandas as pd
import plotly.express as px

# -------------------------
# Your numbers (from tables)
# -------------------------
docs_clean = 1826
docs_attacked = 8130

# Variants (counts not provided per type -> equal split)
pdf_variants = ["ICW", "Font-based", "Dual-layer", "IGShield-1", "IGShield-2"]
html_variants = ["CSS hidden text", "Inline span overlay", "Image-canvas overlay"]

# Assessment structure (percentages)
qtypes = {"MCQ": 41.7, "T/F": 41.7, "Long-form": 16.7}
marks = {"2-mark": 83, "10-mark": 17}
sources = {"Benchmark": 65, "OCW": 35}

# Academic profile (percentages)
disciplines = {"STEM": 48, "Humanities": 27, "Social sciences": 15, "Other": 10}
levels = {"K-12": 18.1, "Undergraduate": 39.8, "Graduate": 42.1}

# -------------------------
# Helper to create rows
# -------------------------
rows = []

ROOT = "IntegrityShield Exam Corpus"

def add_path(path, value):
    # path is list like [ROOT, L1, L2, ...]
    rows.append({"path": path, "value": value})

# -------------------------
# Level 1: corpus split
# -------------------------
add_path([ROOT, "Clean base corpus"], docs_clean)
add_path([ROOT, "Watermarked / adversarial corpus"], docs_attacked)

# -------------------------
# Clean branch: attach "properties" rings (not doc-format specific in your tables)
# (We split by percentages just to visualize structure)
# -------------------------
for name, pct in qtypes.items():
    add_path([ROOT, "Clean base corpus", "Question types", name], docs_clean * pct / 100.0)

for name, pct in marks.items():
    add_path([ROOT, "Clean base corpus", "Marks per question", name], docs_clean * pct / 100.0)

for name, pct in sources.items():
    add_path([ROOT, "Clean base corpus", "Sources", name], docs_clean * pct / 100.0)

for name, pct in disciplines.items():
    add_path([ROOT, "Clean base corpus", "Discipline", name], docs_clean * pct / 100.0)

for name, pct in levels.items():
    add_path([ROOT, "Clean base corpus", "Education level", name], docs_clean * pct / 100.0)

# -------------------------
# Attacked branch: show PDF/HTML -> perturbations
# We don't know PDF vs HTML split, so split attacked docs evenly between them.
# -------------------------
docs_attacked_pdf = docs_attacked / 2
docs_attacked_html = docs_attacked / 2

# PDF perturbations (equal split)
for p in pdf_variants:
    add_path([ROOT, "Watermarked / adversarial corpus", "PDF (5 variants)", "PDF perturbations", p],
             docs_attacked_pdf / len(pdf_variants))

# HTML perturbations (equal split)
for p in html_variants:
    add_path([ROOT, "Watermarked / adversarial corpus", "HTML (3 variants)", "HTML perturbations", p],
             docs_attacked_html / len(html_variants))

# Also attach global properties to attacked branch (same table stats)
for name, pct in qtypes.items():
    add_path([ROOT, "Watermarked / adversarial corpus", "Question types", name], docs_attacked * pct / 100.0)

for name, pct in marks.items():
    add_path([ROOT, "Watermarked / adversarial corpus", "Marks per question", name], docs_attacked * pct / 100.0)

for name, pct in sources.items():
    add_path([ROOT, "Watermarked / adversarial corpus", "Sources", name], docs_attacked * pct / 100.0)

for name, pct in disciplines.items():
    add_path([ROOT, "Watermarked / adversarial corpus", "Discipline", name], docs_attacked * pct / 100.0)

for name, pct in levels.items():
    add_path([ROOT, "Watermarked / adversarial corpus", "Education level", name], docs_attacked * pct / 100.0)

df = pd.DataFrame(rows)

# Plotly sunburst expects either explicit parent/label columns or "path" columns.
# We'll expand path into separate columns.
max_depth = df["path"].map(len).max()
path_cols = [f"level_{i}" for i in range(max_depth)]
expanded = pd.DataFrame(df["path"].tolist(), columns=path_cols)
df_plot = pd.concat([expanded, df[["value"]]], axis=1)

fig = px.sunburst(
    df_plot,
    path=path_cols,
    values="value",
)

# Paper-friendly formatting
fig.update_traces(
    insidetextorientation="radial",
    textinfo="label",
)
fig.update_layout(
    margin=dict(t=10, l=10, r=10, b=10),
    uniformtext=dict(minsize=10, mode="hide"),
)

fig.show()

# Export (best for papers: PDF/SVG). Requires: pip install -U kaleido
# fig.write_image("integrityshield_sunburst.pdf")
# fig.write_image("integrityshield_sunburst.svg")
