#!/usr/bin/env python3
"""Generate a formatted .docx report from statistical test results."""

import csv
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

OUT_DIR = "/home/saiful/CARCM/benchmarks/statistical_test_results"

def read_csv(filename):
    with open(os.path.join(OUT_DIR, filename)) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows

def fmt_p(val):
    v = float(val)
    if v == 0:
        return "< 1e-300"
    elif v < 1e-15:
        return f"{v:.2e}"
    elif v < 0.001:
        return f"{v:.2e}"
    else:
        return f"{v:.4f}"

def fmt_float(val, decimals=4):
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)

def set_cell_text(cell, text, bold=False, size=9, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    run = p.add_run(str(text))
    run.font.size = Pt(size)
    run.bold = bold

def add_styled_table(doc, header, rows, col_formatters=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, h in enumerate(header):
        set_cell_text(table.rows[0].cells[i], h, bold=True, size=9)

    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            if col_formatters and c_idx in col_formatters:
                val = col_formatters[c_idx](val)
            set_cell_text(table.rows[r_idx + 1].cells[c_idx], val, size=9)

    return table


def main():
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    # ── Title ──
    title = doc.add_heading("Statistical Significance Analysis", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("CARCM vs. Benchmark Models")
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    doc.add_paragraph()

    # ── 1. Overview ──
    doc.add_heading("1. Overview", level=1)
    doc.add_paragraph(
        "This report presents the results of comprehensive statistical "
        "significance testing comparing CARCM (Context-Aware Rice Classification "
        "Model) against 12 benchmark models. All tests were performed on the same "
        "held-out test set to ensure paired comparisons. Multiple-comparison "
        "correction was applied using the Benjamini\u2013Hochberg (BH) procedure "
        "to control the false discovery rate (FDR)."
    )
    doc.add_paragraph(
        "The four statistical components are:\n"
        "  (a) DeLong\u2019s test \u2013 compares paired ROC AUCs "
        "(Z-statistic, p-value, 95% CI on \u0394AUC)\n"
        "  (b) Stratified bootstrap 95% CIs \u2013 for AUC, PR-AUC, F1, and MCC "
        "(2,000 resamples preserving class prevalence)\n"
        "  (c) McNemar\u2019s exact test \u2013 paired classification-level "
        "comparison at each model\u2019s optimal threshold\n"
        "  (d) Benjamini\u2013Hochberg FDR correction \u2013 across the family "
        "of 24 pairwise tests (12 DeLong + 12 McNemar)"
    )

    # ── 2. DeLong's Test ──
    doc.add_heading("2. DeLong\u2019s Test (CARCM vs. Each Benchmark)", level=1)
    doc.add_paragraph(
        "DeLong\u2019s test (Sun & Xu, 2014 fast implementation) evaluates whether "
        "the difference in AUC between CARCM and each benchmark is statistically "
        "significant. A two-sided test is used with the null hypothesis that both "
        "AUCs are equal. The BH-adjusted q-values are reported alongside raw p-values."
    )

    header, rows = read_csv("delong_vs_CARCM.csv")
    display_header = ["Model", "AUC\n(CARCM)", "AUC\n(Bench)", "\u0394AUC",
                      "Z-stat", "p-value", "95% CI\nLow", "95% CI\nHigh",
                      "BH q-value"]
    formatters = {
        1: lambda v: fmt_float(v, 4),
        2: lambda v: fmt_float(v, 4),
        3: lambda v: fmt_float(v, 4),
        4: lambda v: fmt_float(v, 2),
        5: fmt_p,
        6: lambda v: fmt_float(v, 4),
        7: lambda v: fmt_float(v, 4),
        8: fmt_p,
    }
    add_styled_table(doc, display_header, rows, formatters)

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Interpretation: ")
    run.bold = True
    p.add_run(
        "All DeLong p-values are effectively zero (< 1e-300), with Z-statistics "
        "ranging from 10.98 (vs. FTTransformer) to 27.83 (vs. GaussianNB). "
        "After BH correction, all q-values remain < 1e-15, confirming that "
        "CARCM\u2019s AUC advantage over every benchmark is statistically "
        "significant at any conventional threshold."
    )

    # ── 3. McNemar's Test ──
    doc.add_heading("3. McNemar\u2019s Test (Paired Classification Decisions)", level=1)
    doc.add_paragraph(
        "McNemar\u2019s test examines whether the disagreements between CARCM and "
        "each benchmark are symmetric. It counts cases where one model is correct "
        "and the other is wrong, testing whether CARCM systematically outperforms. "
        "BH-adjusted q-values are included."
    )

    header, rows = read_csv("mcnemar_vs_CARCM.csv")
    display_header = ["Model", "CARCM Right,\nBench Wrong",
                      "CARCM Wrong,\nBench Right", "\u03c7\u00b2 Stat",
                      "p-value", "BH q-value"]
    formatters = {
        3: lambda v: fmt_float(v, 2),
        4: fmt_p,
        5: fmt_p,
    }
    add_styled_table(doc, display_header, rows, formatters)

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Interpretation: ")
    run.bold = True
    p.add_run(
        "In every comparison, CARCM correctly classifies substantially more "
        "samples that the benchmark gets wrong than vice versa. All McNemar "
        "p-values and BH-adjusted q-values are < 1e-15, confirming statistically "
        "significant superiority at the classification-decision level."
    )

    # ── 4. Bootstrap CIs ──
    doc.add_heading(
        "4. Stratified Bootstrap 95% Confidence Intervals", level=1)
    doc.add_paragraph(
        "Stratified bootstrap resampling (2,000 iterations, preserving "
        "positive/negative class proportions in each draw) was used to compute "
        "95% confidence intervals for AUC, PR-AUC (Average Precision), "
        "F1-score, and Matthews Correlation Coefficient (MCC) for each model."
    )

    header, rows = read_csv("bootstrap_ci.csv")
    display_header = ["Model",
                      "AUC", "AUC Lo", "AUC Hi",
                      "PR-AUC", "PR-AUC Lo", "PR-AUC Hi",
                      "F1", "F1 Lo", "F1 Hi",
                      "MCC", "MCC Lo", "MCC Hi"]
    formatters = {i: lambda v: fmt_float(v, 4) for i in range(1, 13)}
    add_styled_table(doc, display_header, rows, formatters)

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Interpretation: ")
    run.bold = True
    p.add_run(
        "CARCM achieves AUC = 0.8522 [0.8411, 0.8628], PR-AUC = 0.8731 "
        "[0.8623, 0.8836], F1 = 0.7910 [0.7815, 0.7999], and "
        "MCC = 0.5051 [0.4794, 0.5284]. The CARCM AUC lower bound (0.8411) "
        "exceeds the upper bound of the best competitor FTTransformer (0.8019). "
        "Similarly, CARCM\u2019s PR-AUC lower bound (0.8623) exceeds "
        "FTTransformer\u2019s upper bound (0.8319). No confidence intervals overlap."
    )

    # ── 5. Supplementary: Pairwise p-values with BH q-values ──
    doc.add_heading(
        "5. Supplementary Table: Pairwise p-values with BH-adjusted q-values",
        level=1)
    doc.add_paragraph(
        "Benjamini\u2013Hochberg FDR correction was applied jointly across all "
        "24 pairwise tests (12 DeLong + 12 McNemar). The table below reports "
        "raw p-values and BH-adjusted q-values for each comparison."
    )

    header, rows = read_csv("pairwise_pvalues_bh.csv")
    display_header = ["Model", "DeLong p", "DeLong q (BH)",
                      "McNemar p", "McNemar q (BH)"]
    formatters = {1: fmt_p, 2: fmt_p, 3: fmt_p, 4: fmt_p}
    add_styled_table(doc, display_header, rows, formatters)

    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("Interpretation: ")
    run.bold = True
    p.add_run(
        "All 24 BH-adjusted q-values are < 4.44e-16, confirming that all "
        "comparisons survive multiple-testing correction at any practical "
        "significance level. The false discovery rate is controlled well below "
        "0.05 across the entire family of tests."
    )

    # ── 6. Conclusion ──
    doc.add_heading("6. Conclusion", level=1)
    doc.add_paragraph(
        "All four statistical components \u2014 DeLong\u2019s test, "
        "McNemar\u2019s test, stratified bootstrap confidence intervals, and "
        "Benjamini\u2013Hochberg FDR correction \u2014 unanimously confirm that "
        "CARCM significantly outperforms all 12 benchmark models. After "
        "BH correction, all q-values remain < 1e-15. The improvements are not "
        "only statistically significant but also practically meaningful:"
    )

    items = [
        "AUC improvement ranges from +0.063 (vs. FTTransformer) to +0.262 "
        "(vs. GaussianNB/BernoulliNB).",
        "PR-AUC improvement ranges from +0.054 (vs. FTTransformer) to +0.250 "
        "(vs. GaussianNB).",
        "F1 improvement ranges from +0.042 (vs. FTTransformer) to +0.294 "
        "(vs. GaussianNB).",
        "MCC improvement ranges from +0.116 (vs. FTTransformer) to +0.473 "
        "(vs. SVM_RBF).",
        "All BH-adjusted q-values < 4.44e-16, surviving stringent FDR control.",
    ]
    for item in items:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_paragraph(
        "These results provide strong statistical evidence that the CARCM "
        "architecture yields superior classification performance compared to "
        "all evaluated traditional machine learning and deep learning baselines, "
        "meeting the standards expected by reviewers at Bioinformatics and "
        "Briefings in Bioinformatics."
    )

    # ── 7. Figures ──
    forest_path = os.path.join(OUT_DIR, "forest_auc_ci.png")
    heatmap_path = os.path.join(OUT_DIR, "pvalue_heatmap.png")

    if os.path.exists(forest_path) or os.path.exists(heatmap_path):
        doc.add_heading("7. Figures", level=1)

    if os.path.exists(forest_path):
        doc.add_heading(
            "Figure 1: AUC Forest Plot with 95% Stratified Bootstrap CIs",
            level=2)
        doc.add_picture(forest_path, width=Inches(5.8))
        last_p = doc.paragraphs[-1]
        last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()

    if os.path.exists(heatmap_path):
        doc.add_heading(
            "Figure 2: p-value Heatmap (CARCM vs. Benchmarks)", level=2)
        doc.add_picture(heatmap_path, width=Inches(5.8))
        last_p = doc.paragraphs[-1]
        last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # ── 8. Output Files ──
    doc.add_heading("8. Output Files", level=1)
    doc.add_paragraph(
        "All result files are saved in the directory: "
        "benchmarks/statistical_test_results/"
    )

    file_descriptions = [
        ("delong_vs_CARCM.csv",
         "DeLong\u2019s test results with AUC differences, Z-statistics, "
         "raw p-values, 95% CIs, and BH-adjusted q-values."),
        ("mcnemar_vs_CARCM.csv",
         "McNemar\u2019s test results with discordant counts, \u03c7\u00b2 "
         "statistics, raw p-values, and BH-adjusted q-values."),
        ("bootstrap_ci.csv",
         "Stratified bootstrap 95% confidence intervals for AUC, PR-AUC, "
         "F1, and MCC (2,000 resamples) for all 13 models."),
        ("pairwise_pvalues_bh.csv",
         "Supplementary table of all pairwise raw p-values with "
         "Benjamini\u2013Hochberg FDR-adjusted q-values."),
        ("summary.csv",
         "Paper-ready merged table: every metric as \u201cX.XXXX [lo, hi]\u201d "
         "with DeLong/McNemar p-values and BH q-values."),
        ("forest_auc_ci.png / .pdf",
         "Forest plot of AUC point estimates and 95% stratified bootstrap CIs."),
        ("pvalue_heatmap.png / .pdf",
         "Heatmap of \u2212log10(p) for DeLong and McNemar tests."),
        ("statistical_significance_report.docx",
         "This document."),
    ]

    file_table = doc.add_table(rows=1 + len(file_descriptions), cols=2)
    file_table.style = "Light Grid Accent 1"
    file_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_cell_text(file_table.rows[0].cells[0], "File", bold=True, size=9)
    set_cell_text(file_table.rows[0].cells[1], "Description", bold=True,
                  size=9, align=WD_ALIGN_PARAGRAPH.LEFT)
    for i, (fname, desc) in enumerate(file_descriptions):
        set_cell_text(file_table.rows[i + 1].cells[0], fname, size=9)
        set_cell_text(file_table.rows[i + 1].cells[1], desc, size=9,
                      align=WD_ALIGN_PARAGRAPH.LEFT)

    out_path = os.path.join(OUT_DIR, "statistical_significance_report.docx")
    doc.save(out_path)
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
