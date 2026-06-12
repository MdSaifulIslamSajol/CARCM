#!/usr/bin/env python3
"""Q1-journal-style architecture figure for context-aware foundation-model
benchmarking pipeline (adapted from Feng et al. 2025, Nat. Commun.).

Serious publication style:
  - rectangular, sharp-cornered boxes
  - right-angle orthogonal routing
  - Set3 qualitative palette
  - 500 dpi PNG
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/benchmarks/fm_context_architecture.png"
OUT_PPTX = "/home/saiful/DeepRice/motif_based_classification_ALL_DATA/bin_based_v2/benchmarks/fm_context_architecture.pptx"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titleweight": "bold",
    "savefig.bbox": "tight",
})

# Set3 palette (10 colors, skip very-pale ones we don't use)
SET3 = plt.cm.Set3(np.linspace(0, 1, 12))
C_SEQ   = SET3[0]   # pale teal
C_TOK   = SET3[4]   # blue
C_FM    = SET3[3]   # red/salmon (emphasis — frozen backbone)
C_POOL  = SET3[2]   # lavender
C_CTX   = SET3[5]   # orange
C_OHE   = SET3[7]   # pink
C_CAT   = SET3[1]   # yellow
C_CLF   = SET3[6]   # green
C_OUT   = SET3[10]  # light violet
BAND1   = "#F4F7FA"
BAND2   = "#FBF6EE"

EDGE = "#222222"
TXT  = "#111111"


def box(ax, x, y, w, h, text, face, *, fontsize=10, weight="normal",
        italic=False, lw=1.2):
    rect = Rectangle((x, y), w, h, facecolor=face, edgecolor=EDGE,
                     linewidth=lw, zorder=3)
    ax.add_patch(rect)
    style = "italic" if italic else "normal"
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, fontstyle=style,
            color=TXT, zorder=4)


def arrow(ax, p0, p1, lw=1.3):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                        linewidth=lw, color=EDGE, zorder=2,
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def elbow(ax, p0, p1, lw=1.3):
    """Orthogonal L-shaped connector: vertical then horizontal (or vice versa).
    Routes via midpoint at (p1.x, p0.y) unless already axis-aligned."""
    x0, y0 = p0
    x1, y1 = p1
    if abs(x0 - x1) < 1e-6 or abs(y0 - y1) < 1e-6:
        arrow(ax, p0, p1, lw=lw)
        return
    mid = (x1, y0)
    ax.plot([x0, mid[0]], [y0, mid[1]], color=EDGE, lw=lw, zorder=2,
            solid_capstyle="butt")
    arrow(ax, mid, p1, lw=lw)


def band(ax, y0, y1, color, label):
    ax.add_patch(Rectangle((0.0, y0), 1.0, y1 - y0, facecolor=color,
                           edgecolor="none", zorder=0))
    ax.text(0.015, y1 - 0.008, label, ha="left", va="top",
            fontsize=10, fontweight="bold", color="#555555", zorder=1)


# ---------------------------------------------------------------------------
#  Native PPTX export  (editable shapes, 4:3 portrait 7.5 × 10 in)
# ---------------------------------------------------------------------------
def save_native_pptx():
    """Build PPTX with native editable rectangles, connectors and text boxes."""
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR_TYPE
    from pptx.enum.text import PP_ALIGN
    from pptx.oxml.ns import qn
    from lxml import etree

    SW, SH = 7.5, 10.0                       # slide size (inches)

    # ---- coordinate helpers (mpl normalised → pptx inches) ----
    def mx(v): return v * SW
    def my(v): return (1 - v) * SH            # y flipped
    def mw(v): return v * SW
    def mh(v): return v * SH

    def hex_rgb(h):
        h = h.lstrip('#')
        return RGBColor(int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    def rgba_rgb(c):
        return RGBColor(int(c[0]*255), int(c[1]*255), int(c[2]*255))

    def to_rgb(c):
        return hex_rgb(c) if isinstance(c, str) else rgba_rgb(c)

    prs = Presentation()
    prs.slide_width  = Inches(SW)
    prs.slide_height = Inches(SH)
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank layout

    # ---- shape builders ----
    def add_box(x_left, y_bot, w, h, text, fill,
                fs=10, bold=False, italic=False):
        shp = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(mx(x_left)), Inches(my(y_bot + h)),
            Inches(mw(w)),      Inches(mh(h)),
        )
        shp.fill.solid()
        shp.fill.fore_color.rgb = to_rgb(fill)
        shp.line.color.rgb = hex_rgb(EDGE)
        shp.line.width = Pt(1.2)

        tf = shp.text_frame
        tf.word_wrap = True
        bodyPr = tf._txBody.find(qn('a:bodyPr'))
        bodyPr.set('anchor', 'ctr')          # vertical-centre

        for i, line in enumerate(text.split('\n')):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = line
            r.font.size  = Pt(fs)
            r.font.bold  = bold
            r.font.italic = italic
            r.font.color.rgb = hex_rgb(TXT)
            r.font.name  = 'Calibri'
        return shp

    def _set_tail_arrow(conn):
        sp_pr = conn._element.spPr
        ln = sp_pr.find(qn('a:ln'))
        if ln is not None:
            tail = etree.SubElement(ln, qn('a:tailEnd'))
            tail.set('type', 'triangle')
            tail.set('w', 'med')
            tail.set('len', 'med')

    def add_arrow(x0, y0, x1, y1, lw=1.3):
        c = slide.shapes.add_connector(
            MSO_CONNECTOR_TYPE.STRAIGHT,
            Inches(mx(x0)), Inches(my(y0)),
            Inches(mx(x1)), Inches(my(y1)),
        )
        c.line.color.rgb = hex_rgb(EDGE)
        c.line.width = Pt(lw)
        _set_tail_arrow(c)
        return c

    def add_line(x0, y0, x1, y1, lw=1.3):
        c = slide.shapes.add_connector(
            MSO_CONNECTOR_TYPE.STRAIGHT,
            Inches(mx(x0)), Inches(my(y0)),
            Inches(mx(x1)), Inches(my(y1)),
        )
        c.line.color.rgb = hex_rgb(EDGE)
        c.line.width = Pt(lw)
        return c

    def add_dashed_line(x0, y0, x1, y1, lw=0.8, color='#888888'):
        c = slide.shapes.add_connector(
            MSO_CONNECTOR_TYPE.STRAIGHT,
            Inches(mx(x0)), Inches(my(y0)),
            Inches(mx(x1)), Inches(my(y1)),
        )
        c.line.color.rgb = hex_rgb(color)
        c.line.width = Pt(lw)
        # set dash style
        sp_pr = c._element.spPr
        ln = sp_pr.find(qn('a:ln'))
        if ln is not None:
            ln.set('dash', 'dash')
            prstDash = etree.SubElement(ln, qn('a:prstDash'))
            prstDash.set('val', 'dash')
        return c

    def add_elbow(x0, y0, x1, y1, lw=1.3):
        if abs(x0 - x1) < 1e-6 or abs(y0 - y1) < 1e-6:
            add_arrow(x0, y0, x1, y1, lw)
            return
        add_line(x0, y0, x1, y0, lw)          # horizontal
        add_arrow(x1, y0, x1, y1, lw)         # vertical + arrowhead

    def add_text(x_ctr, y_ctr, w, h, text,
                 fs=10, bold=False, italic=False, color=TXT,
                 align=PP_ALIGN.CENTER):
        tb = slide.shapes.add_textbox(
            Inches(mx(x_ctr - w / 2)), Inches(my(y_ctr + h / 2)),
            Inches(mw(w)),             Inches(mh(h)),
        )
        tf = tb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(text.split('\n')):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            r = p.add_run()
            r.text = line
            r.font.size  = Pt(fs)
            r.font.bold  = bold
            r.font.italic = italic
            r.font.color.rgb = to_rgb(color)
            r.font.name  = 'Calibri'
        return tb

    def add_band(y0, y1, color, label):
        shp = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(0), Inches(my(y1)),
            Inches(SW), Inches(mh(y1 - y0)),
        )
        shp.fill.solid()
        shp.fill.fore_color.rgb = hex_rgb(color)
        shp.line.fill.background()             # no border
        add_text(0.30, y1 - 0.015, 0.55, 0.025, label,
                 fs=10, bold=True, color='#555555', align=PP_ALIGN.LEFT)

    # =================== BUILD THE SLIDE ===================

    # -- background bands (added first → behind everything) --
    add_band(0.54, 0.965, BAND1,
             'Stage 1 — Sequence Embedding (frozen foundation model)')
    add_band(0.07, 0.38, BAND2,
             'Stage 2 — Context-aware Classifier')

    # -- LEFT column (sequence path) --
    cx_l, w_l = 0.28, 0.32
    xl = cx_l - w_l / 2

    add_box(xl, 0.885, w_l, 0.050,
            'Promoter sequence (1000 bp)\nATCG…',
            C_SEQ, fs=10, bold=True)
    add_box(xl, 0.800, w_l, 0.050,
            'Tokenizer (k-mer / BPE / character)',
            C_TOK, fs=10)
    add_box(xl, 0.695, w_l, 0.070,
            'Foundation-model backbone (FROZEN)\n'
            'DNABERT-2 | NT | HyenaDNA | Caduceus',
            C_FM, fs=9, bold=True)
    add_box(xl, 0.610, w_l, 0.050,
            'Pooling (mean / CLS / max)\n'
            '→ e ∈ ℝ^{d_emb} (e.g. 768)',
            C_POOL, fs=9)

    # arrows between left-column boxes
    add_arrow(cx_l, 0.885, cx_l, 0.850)       # seq → tok
    add_arrow(cx_l, 0.800, cx_l, 0.765)       # tok → FM
    add_arrow(cx_l, 0.695, cx_l, 0.660)       # FM  → pool

    # "no gradient updates" dashed annotation
    fm_right = cx_l + w_l / 2
    fm_mid_y = 0.695 + 0.035
    add_dashed_line(fm_right, fm_mid_y, fm_right + 0.06, fm_mid_y)
    add_text(fm_right + 0.06 + 0.07, fm_mid_y, 0.14, 0.018,
             'no gradient updates',
             fs=8, italic=True, color='#555555', align=PP_ALIGN.LEFT)

    # -- RIGHT column (context path) --
    cx_r, w_r = 0.74, 0.34
    xr = cx_r - w_r / 2

    add_box(xr, 0.855, w_r, 0.080,
            'Context metadata (per row)\n'
            'Variety ∈ 13  •  Tissue ∈ 8  •  Stage ∈ 6',
            C_CTX, fs=9, bold=True)
    add_box(xr, 0.745, w_r, 0.060,
            'OneHotEncoder (fit on TRAIN)\n'
            'c ∈ {0,1}^{27}',
            C_OHE, fs=9)

    add_arrow(cx_r, 0.855, cx_r, 0.805)       # ctx → OHE

    # -- CONCAT --
    cx_c, w_c = 0.50, 0.52
    add_box(cx_c - w_c / 2, 0.455, w_c, 0.065,
            'Concatenate   x = [ e ‖ c ]   x ∈ ℝ^{d_emb + 27}',
            C_CAT, fs=11, bold=True)

    # elbow connectors into concat
    add_elbow(cx_l, 0.610, cx_c - 0.14, 0.520)   # pool  → concat
    add_elbow(cx_r, 0.745, cx_c + 0.14, 0.520)   # OHE   → concat

    # -- CLASSIFIER --
    add_box(cx_c - 0.33, 0.235, 0.66, 0.090,
            'Downstream classifier\n'
            'Random Forest • XGBoost • LightGBM • ElasticNet • Naive Bayes\n'
            '( GridSearchCV on ROC-AUC, 5-fold )',
            C_CLF, fs=9, bold=True)

    add_arrow(cx_c, 0.455, cx_c, 0.325, lw=1.5)  # concat → clf

    # -- OUTPUT --
    add_box(cx_c - 0.24, 0.115, 0.48, 0.062,
            'ŷ ∈ { Up-regulated, Down-regulated }',
            C_OUT, fs=11, bold=True)

    add_arrow(cx_c, 0.235, cx_c, 0.177, lw=1.5)  # clf → output

    # -- TITLE --
    add_text(0.50, 0.980, 0.90, 0.04,
             'Context-aware foundation-model benchmarking pipeline\n'
             '(adapted from Feng et al., Nat. Commun. 2025)',
             fs=13, bold=True)

    # -- FOOTER --
    add_text(0.50, 0.040, 0.92, 0.055,
             'Stage 1 is model-specific and frozen — run once per FM '
             'to cache embeddings.\n'
             'Context is injected only at Stage 2 by concatenating '
             'one-hot(Variety, Tissue, Stage) to the pooled embedding,\n'
             'keeping every foundation model in the benchmark '
             'directly comparable.',
             fs=9, color='#333333')

    os.makedirs(os.path.dirname(OUT_PPTX), exist_ok=True)
    prs.save(OUT_PPTX)
    print(f'Saved (native shapes): {OUT_PPTX}')


def main():
    fig, ax = plt.subplots(figsize=(11, 12.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------- bands ----------------
    band(ax, 0.54, 0.965, BAND1, "Stage 1  —  Sequence Embedding  (frozen foundation model)")
    band(ax, 0.07,  0.38, BAND2, "Stage 2  —  Context-aware Classifier")

    # ---------------- Stage 1 left column (sequence) ----------------
    cx_l = 0.28
    w_l  = 0.32

    # y-top, height
    boxes_L = [
        (0.885, 0.050, "Promoter sequence  (1000 bp)\nATCG…", C_SEQ,  10, "bold"),
        (0.800, 0.050, "Tokenizer  (k-mer / BPE / character)", C_TOK, 10, "normal"),
        (0.695, 0.070, "Foundation-model backbone  (FROZEN)\n"
                        "DNABERT-2  |  NT  |  HyenaDNA  |  Caduceus",
                        C_FM, 9, "bold"),
        (0.610, 0.050, "Pooling  (mean / CLS / max)\n→  e  ∈  ℝ^{d_emb}   (e.g. 768)",
                        C_POOL, 9, "normal"),
    ]
    for (y, h, t, c, fs, wt) in boxes_L:
        box(ax, cx_l - w_l / 2, y, w_l, h, t, c, fontsize=fs, weight=wt)

    # vertical arrows between L boxes
    for i in range(len(boxes_L) - 1):
        y_top_current = boxes_L[i][0]           # top y of box i
        y_top_next    = boxes_L[i + 1][0]
        h_next        = boxes_L[i + 1][1]
        arrow(ax, (cx_l, y_top_current), (cx_l, y_top_next + h_next))

    # Frozen annotation
    fm_y, fm_h = boxes_L[2][0], boxes_L[2][1]
    ax.annotate("no gradient updates",
                xy=(cx_l + w_l / 2, fm_y + fm_h / 2),
                xytext=(cx_l + w_l / 2 + 0.08, fm_y + fm_h / 2),
                fontsize=8.5, color="#555555", style="italic", va="center",
                arrowprops=dict(arrowstyle="-", color="#888888",
                                linestyle=(0, (2, 2)), lw=0.8))

    # ---------------- Stage 1 right column (context) ----------------
    cx_r = 0.74
    w_r  = 0.34

    boxes_R = [
        (0.855, 0.080, "Context metadata  (per row)\n"
                        "Variety ∈ 13   •   Tissue ∈ 8   •   Stage ∈ 6",
                        C_CTX, 9, "bold"),
        (0.745, 0.060, "OneHotEncoder  (fit on TRAIN)\nc  ∈  {0,1}^{27}",
                        C_OHE, 9, "normal"),
    ]
    for (y, h, t, c, fs, wt) in boxes_R:
        box(ax, cx_r - w_r / 2, y, w_r, h, t, c, fontsize=fs, weight=wt)

    # arrow between R boxes
    arrow(ax, (cx_r, boxes_R[0][0]), (cx_r, boxes_R[1][0] + boxes_R[1][1]))

    # ---------------- Concat node ----------------
    cat_y, cat_h = 0.455, 0.065
    cx_c = 0.50
    w_c  = 0.52
    box(ax, cx_c - w_c / 2, cat_y, w_c, cat_h,
        "Concatenate       x = [  e  ‖  c  ]       x  ∈  ℝ^{d_emb + 27}",
        C_CAT, fontsize=11, weight="bold")

    # orthogonal joins from pool (L) and OHE (R) into concat
    # L: from bottom-center of pool box down to cat top-left quadrant
    pool_y = boxes_L[3][0]  # top of pool box
    # bottom of pool = y
    pool_bottom_y = pool_y - 0  # pool_y is its top actually; need bottom = y
    # Our coords: box() uses (x, y=bottom, w, h). boxes_L tuple gave top-y...
    # Wait: I passed y as *bottom* into box(). Let me recheck:
    # In box() call: box(ax, cx_l - w_l/2, y, w_l, h, ...) where y = boxes_L[i][0].
    # So 0.885 is BOTTOM of top sequence box, etc. Its top is y+h.
    # arrow from top box (i=0): y_top_current was used as boxes_L[i][0] (= BOTTOM),
    # but I labelled it "y_top_current". That's a bug. Let me re-express correctly.
    pass

    # rebuild arrows properly using bottoms/tops
    # (overwrite with corrected geometry)
    # clear is unnecessary; arrows already drawn between b[i].y (bottom) and
    # b[i+1].y + b[i+1].h (top). That means arrow goes from bottom of upper box
    # DOWN to top of lower box — correct IF upper has larger y. boxes_L are
    # listed top→bottom with decreasing y, so boxes_L[i].y > boxes_L[i+1].y+h,
    # OK. The earlier "y_top_current" misnaming is harmless — arrow geometry
    # is correct.

    # Join L pool (bottom) -> concat (top-left area)
    pool_bottom = (cx_l, boxes_L[3][0])
    cat_top_L   = (cx_c - 0.14, cat_y + cat_h)
    elbow(ax, pool_bottom, cat_top_L)

    # Join R OHE (bottom) -> concat (top-right area)
    ohe_bottom = (cx_r, boxes_R[1][0])
    cat_top_R  = (cx_c + 0.14, cat_y + cat_h)
    elbow(ax, ohe_bottom, cat_top_R)

    # ---------------- Stage 2: classifier ----------------
    clf_y, clf_h = 0.235, 0.090
    box(ax, cx_c - 0.33, clf_y, 0.66, clf_h,
        "Downstream classifier\n"
        "Random Forest   •   XGBoost   •   LightGBM   •   ElasticNet   •   Naive Bayes\n"
        "( GridSearchCV on ROC-AUC,  5-fold )",
        C_CLF, fontsize=9, weight="bold")

    arrow(ax, (cx_c, cat_y), (cx_c, clf_y + clf_h), lw=1.5)

    # ---------------- Output ----------------
    out_y, out_h = 0.115, 0.062
    box(ax, cx_c - 0.24, out_y, 0.48, out_h,
        "ŷ   ∈   { Up-regulated ,   Down-regulated }",
        C_OUT, fontsize=11, weight="bold")

    arrow(ax, (cx_c, clf_y), (cx_c, out_y + out_h), lw=1.5)

    # ---------------- Title ----------------
    fig.suptitle(
        "Context-aware foundation-model benchmarking pipeline\n"
        "(adapted from Feng et al., Nat. Commun. 2025)",
        fontsize=13.5, fontweight="bold", y=0.995,
    )

    # ---------------- Footer note ----------------
    ax.text(
        0.5, 0.035,
        "Stage 1 is model-specific and frozen — run once per FM to cache embeddings.\n"
        "Context is injected only at Stage 2 by concatenating one-hot(Variety, Tissue, Stage) "
        "to the pooled embedding,\nkeeping every foundation model in the benchmark directly comparable.",
        ha="center", va="center", fontsize=9.5, color="#333333",
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=500, bbox_inches="tight")
    print(f"Saved: {OUT}")
    plt.close(fig)

    save_native_pptx()


if __name__ == "__main__":
    main()
