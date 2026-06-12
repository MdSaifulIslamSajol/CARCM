"""Render memory.md content into a publication-styled DOCX file
saved as paper_plan.docx in this directory."""

import os
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT_DIR = "/home/saiful/CARCM/Ablation study"
OUT_FILE = os.path.join(OUT_DIR, "paper_plan.docx")


def set_cell_shading(cell, fill_hex):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tc_pr.append(shd)


def add_table(doc, headers, rows, header_fill="1F4E79",
              col_widths_cm=None, group_rows=None,
              bold_row_indices=None):
    """group_rows: dict {row_idx: 'Group label'} — those rows are merged & bold.
       bold_row_indices: list of row indices whose label cell should be bold."""
    n_cols = len(headers)
    table = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.style = "Light Grid Accent 1"

    if col_widths_cm:
        for j, w in enumerate(col_widths_cm):
            for cell in table.columns[j].cells:
                cell.width = Cm(w)

    # header
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(cell, header_fill)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # body
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
            if bold_row_indices and (i - 1) in bold_row_indices:
                run.bold = True
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    if group_rows:
        for ridx, label in group_rows.items():
            target = ridx + 1  # +1 for header row
            cell0 = table.rows[target].cells[0]
            for k in range(1, n_cols):
                cell0.merge(table.rows[target].cells[k])
            cell0.text = ""
            run = cell0.paragraphs[0].add_run(label)
            run.italic = True
            run.bold = True
            run.font.size = Pt(9)
            set_cell_shading(cell0, "EEEEEE")


def add_caption(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)


def main():
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # Title
    t = doc.add_heading("Ablation Study — Paper Presentation Plan", level=0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run(
        "Reference document for which of the 20 trained ablation "
        "configurations should appear in the paper, in which form, and where."
    )
    sr.italic = True
    sr.font.size = Pt(11)

    # ---- 1. What's in this directory ----
    doc.add_heading("1. What's in this directory", level=1)
    doc.add_paragraph(
        "20 trained configurations × 10 seeds, split into 4 logical "
        "sub-studies."
    )
    add_table(
        doc,
        headers=["Sub-study", "Configs", "Source"],
        rows=[
            ["A. Context ablation", "8",
             "results/{full, sequence_only, seq_variety, seq_tissue, seq_stage, "
             "permuted_labels, context_only, param_matched_bias}/"],
            ["B. Feature LOO under full context", "4",
             "results/{no_binned, no_positional, no_count, no_cooc}/"],
            ["C. Feature LOO under no context", "4",
             "results/{seq_only_no_binned, seq_only_no_positional, "
             "seq_only_no_count, seq_only_no_cooc}/"],
            ["D. Feature LOI under full context", "4",
             "feature_loi/results/{only_binned, only_positional, only_count, "
             "only_cooc}/"],
        ],
        col_widths_cm=[5.0, 1.5, 9.5],
    )

    # ---- 2. Plan ----
    doc.add_heading("2. Paper plan: 2 tables, 3 figures", level=1)
    doc.add_paragraph(
        "20 rows in one table will overwhelm reviewers. 20 rows split into 4 "
        "separate small tables looks like padding. The clean middle ground is:"
    )
    for b in [
        "1 main-text table + 1 supplementary table",
        "1 main-text figure + 2 supplementary figures",
    ]:
        doc.add_paragraph(b, style="List Bullet")

    # ---- 3. Main paper Table 2 ----
    doc.add_heading("3. Main paper — Table 2 (architecture ablation, 8 rows)",
                    level=1)
    doc.add_paragraph(
        "Pick the rows that each answer a distinct reviewer question. Eight "
        "rows is the sweet spot."
    )
    table2_rows = [
        ["1", "Full CARCM",
         "0.8531", "+0.232", "reference", "the model"],
        ["2", "Sequence-only",
         "0.6211", "0 (ref)", "<1e-300 ****", "defines the gain to explain"],
        ["3", "Context-only",
         "0.6811", "+0.060", "<1e-300 ****",
         "upper bound on label-prior memorisation"],
        ["4", "Permuted-label control",
         "0.5977", "−0.023", "<1e-300 ****",
         "proves context is real signal"],
        ["5", "Param-matched bias control",
         "0.6205", "−0.001", "<1e-300 ****",
         "rules out the parameter-count confound"],
        ["6", "Sequence + Tissue only",
         "0.7614", "+0.140", "<1e-300 ****",
         "strongest single context dimension"],
        ["7", "only_binned + context",
         "0.8484", "+0.227", "0.105 n.s.",
         "binned alone matches Full"],
        ["8", "only_cooc + context",
         "0.7569", "+0.136", "<1e-300 ****",
         "weakest single feature (range of LOI)"],
    ]
    add_table(
        doc,
        headers=["#", "Configuration", "Ensemble AUC",
                 "ΔAUC vs Seq-only", "DeLong p vs Full",
                 "Why this row is in the table"],
        rows=table2_rows,
        col_widths_cm=[0.8, 4.6, 1.8, 2.0, 2.0, 4.8],
        bold_row_indices=[0, 3, 4, 6],   # Full, perm, bias, only_binned
    )
    doc.add_paragraph(
        "Why this set: rows 1–5 settle the “context is real / not just "
        "parameters” debate; row 6 names the strongest context dimension; "
        "rows 7–8 frame the LOI ceiling range (best to worst single-feature). "
        "Eight rows answer every objection a reviewer is likely to raise."
    )

    # ---- 4. Main paper figure ----
    doc.add_heading("4. Main paper — Figure 1 (one figure)", level=1)
    doc.add_paragraph(
        "Use the existing figures/ablation_delta_auc.png — it already shows "
        "context ablation rows 1–7 with paired-seed 95% CIs and DeLong "
        "asterisks at 400 dpi. No re-rendering needed."
    )
    fig1 = os.path.join(OUT_DIR, "figures", "ablation_delta_auc.png")
    if os.path.exists(fig1):
        doc.add_picture(fig1, width=Inches(6.5))
        add_caption(
            doc,
            "Figure 1 (preview). Context ablation ΔAUC vs sequence-only baseline; "
            "bars are paired-seed mean ± 95% CI across 10 seeds; markers are "
            "DeLong p vs Full CARCM."
        )

    # ---- 5. Supplementary Table S1 ----
    doc.add_heading(
        "5. Supplementary Table S1 — Full feature decomposition (16 rows)",
        level=1
    )
    doc.add_paragraph(
        "The single supplementary table that bundles all 12 feature-level "
        "rows from sub-studies B + C + D, with three references at the top."
    )
    s1_rows = [
        ["Reference rows", "", "", "", "", ""],
        ["1", "Full CARCM", "0.8531", "—", "—", "—"],
        ["2", "Sequence-only", "0.6211", "—", "—", "—"],
        ["7", "Context-only", "0.6811", "—", "—", "—"],
        ["Group B — LOO with context (vs Full)", "", "", "", "", ""],
        ["9",  "− Binned",         "0.8405", "−0.013", "Full", "DeLong p=1.1e-4 ***"],
        ["10", "− Positional",     "0.8503", "−0.003", "Full", "n.s."],
        ["11", "− Global Counts",  "0.8548", "+0.002", "Full", "n.s."],
        ["12", "− Co-occurrence",  "0.8539", "+0.001", "Full", "n.s."],
        ["Group C — LOO without context (vs Sequence-only)", "", "", "", "", ""],
        ["13", "− Binned (no ctx)",        "0.6252", "+0.004", "Seq-only", "n.s."],
        ["14", "− Positional (no ctx)",    "0.6206", "−0.001", "Seq-only", "n.s."],
        ["15", "− Global Counts (no ctx)", "0.6216", "+0.001", "Seq-only", "n.s."],
        ["16", "− Co-occurrence (no ctx)", "0.6199", "−0.001", "Seq-only", "n.s."],
        ["Group D — LOI with context (vs Full)", "", "", "", "", ""],
        ["17", "only_binned + ctx",     "0.8484", "−0.005", "Full", "n.s."],
        ["19", "only_count + ctx",      "0.8389", "−0.014", "Full", "DeLong p=7.4e-4 ***"],
        ["18", "only_positional + ctx", "0.8135", "−0.040", "Full", "<1e-300 ****"],
        ["20", "only_cooc + ctx",       "0.7569", "−0.096", "Full", "<1e-300 ****"],
    ]
    # Map of group-label rows for merging:
    group_rows = {
        0: "Reference rows",
        4: "Group B — LOO with context (vs Full)",
        9: "Group C — LOO without context (vs Sequence-only)",
        14: "Group D — LOI with context (vs Full)",
    }
    add_table(
        doc,
        headers=["#", "Configuration", "Ensemble AUC",
                 "ΔAUC vs reference", "Reference", "Significance"],
        rows=s1_rows,
        col_widths_cm=[1.0, 5.5, 2.0, 2.5, 2.0, 4.0],
        group_rows=group_rows,
    )
    doc.add_paragraph(
        "This single table tells the complete feature-level story. "
        "Crucially, both LOO and LOI must be present — LOO alone makes 3 of 4 "
        "features look like noise; LOI alone hides the redundancy. The pair "
        "is what's defensible."
    )

    # ---- 6. Supplementary figures ----
    doc.add_heading("6. Supplementary Figures", level=1)
    add_table(
        doc,
        headers=["#", "Figure", "Path", "Why"],
        rows=[
            ["S-F1", "LOI two-panel",
             "feature_loi/figures/loi_delta_auc.png",
             "Cleanest single visual for the feature-level story"],
            ["S-F2", "LOO with-vs-without context",
             "figures/feature_loo_compare.png",
             "Shows redundancy persists even with the gating fallback removed"],
        ],
        col_widths_cm=[1.4, 4.5, 6.0, 6.0],
    )
    s_f1 = os.path.join(OUT_DIR, "feature_loi", "figures", "loi_delta_auc.png")
    s_f2 = os.path.join(OUT_DIR, "figures", "feature_loo_compare.png")
    if os.path.exists(s_f1):
        doc.add_picture(s_f1, width=Inches(6.5))
        add_caption(doc, "Supplementary Figure S1 (preview). LOI two-panel.")
    if os.path.exists(s_f2):
        doc.add_picture(s_f2, width=Inches(6.5))
        add_caption(doc,
                    "Supplementary Figure S2 (preview). LOO with-vs-without context.")

    # ---- 7. What to omit ----
    doc.add_heading("7. What to omit from the paper", level=1)
    add_table(
        doc,
        headers=["Configs / Artifacts", "Why omit"],
        rows=[
            ["Sequence + Variety only (Config 3)",
             "Variety contribution (+0.07) is significant but redundant with "
             "the Tissue row in the main paper. Belongs in supplement S1 if "
             "expanded."],
            ["Sequence + Stage only (Config 5)", "Same logic."],
            ["ablation_metric_grid.png (8-config 4-metric grid)",
             "Metric grids are noisy and largely redundant with the ΔAUC bar chart."],
            ["feature_ablation_metric_grid.png, feature_seqonly_delta_auc.png, loi_metric_grid.png",
             "Bundle metrics into table cells; keep figures focused on ΔAUC."],
            ["All console_outputs/ logs",
             "Supplementary data deposit, not paper content."],
        ],
        col_widths_cm=[7.5, 9.5],
    )

    # ---- 8. Summary ----
    doc.add_heading("8. Summary — what to put where", level=1)
    add_table(
        doc,
        headers=["Paper location", "Item", "Source"],
        rows=[
            ["Main text — Table 2",
             "Architecture ablation (8 rows)",
             "aggregate_metrics.csv + feature_loi/results/loi_aggregate_metrics.csv"],
            ["Main text — Figure 1",
             "Context ablation ΔAUC bar chart",
             "figures/ablation_delta_auc.png"],
            ["Supplement — Table S1",
             "Full feature decomposition (16 rows)",
             "combined CSV"],
            ["Supplement — Figure S1",
             "LOI two-panel",
             "feature_loi/figures/loi_delta_auc.png"],
            ["Supplement — Figure S2",
             "LOO with-vs-without context",
             "figures/feature_loo_compare.png"],
        ],
        col_widths_cm=[5.0, 5.5, 7.0],
    )
    doc.add_paragraph(
        "That's it: two tables, three figures, drawn entirely from this "
        "directory. The 8-row main-text table answers the high-stakes "
        "reviewer questions; the 16-row supplementary table backs up every "
        "claim; both figures are pre-rendered at Q1-acceptable 400 dpi."
    )

    doc.save(OUT_FILE)
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
