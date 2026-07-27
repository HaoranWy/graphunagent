#!/usr/bin/env python3
"""Build the editable, final-size PWSC-CI overview figure.

The source figure was authored on the final-size canvas with 6 pt text and
shrink-to-fit enabled.  Rebuilding the first slide with short, native text
boxes makes the 9 pt minimum measurable in the exported artifact while
retaining the original five-part story and color language.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt


W = 6.86  # 17.42 cm, the full-width final canvas used by the paper
H = 2.60  # 6.60 cm
MIN_PT = 9

WHITE = RGBColor(255, 255, 255)
INK = RGBColor(24, 37, 45)
MUTED = RGBColor(83, 102, 111)
RED = RGBColor(220, 93, 83)
RED_PALE = RGBColor(255, 244, 241)
TEAL = RGBColor(0, 135, 132)
TEAL_PALE = RGBColor(239, 250, 248)
GOLD = RGBColor(199, 145, 22)
GOLD_PALE = RGBColor(255, 250, 236)
SLATE = RGBColor(41, 59, 68)
GRID = RGBColor(207, 220, 221)


def set_cell_fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def set_line(shape, color: RGBColor, width: float = 0.8) -> None:
    shape.line.color.rgb = color
    shape.line.width = Pt(width)


def add_box(slide, x, y, w, h, fill, line, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    set_cell_fill(shape, fill)
    set_line(shape, line, 0.75)
    return shape


def add_text(
    slide,
    x,
    y,
    w,
    h,
    text,
    *,
    size=MIN_PT,
    color=INK,
    bold=False,
    font="Arial",
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.MIDDLE,
    margin=0.0,
):
    if size < MIN_PT:
        raise ValueError(f"figure text below {MIN_PT} pt: {text!r}")
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = f"txt_{text[:24]}"
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    paragraphs = text.split("\n")
    for idx, value in enumerate(paragraphs):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = value
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        p.line_spacing = 1.0
        for run in p.runs:
            run.font.name = font
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
    return shape


def add_formula(
    slide,
    x,
    y,
    w,
    h,
    lines,
    *,
    size=MIN_PT,
    color=INK,
    align=PP_ALIGN.CENTER,
):
    """Add editable math-like text with true PowerPoint baselines.

    Each segment is ``(text, baseline)`` where baseline is ``sub``, ``sup``,
    or ``None``.  Subscripts remain 9 pt; only their baseline moves.
    """
    if size < MIN_PT:
        raise ValueError("formula run below the 9 pt figure minimum")
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = "formula_" + "".join(seg[0] for seg in lines[0])[:24]
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = False
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for line_idx, segments in enumerate(lines):
        p = tf.paragraphs[0] if line_idx == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        p.line_spacing = 1.0
        for value, baseline in segments:
            run = p.add_run()
            run.text = value
            run.font.name = "Cambria Math"
            run.font.size = Pt(size)
            run.font.color.rgb = color
            rpr = run._r.get_or_add_rPr()
            if baseline == "sub":
                rpr.set("baseline", "-25000")
            elif baseline == "sup":
                rpr.set("baseline", "30000")
    return shape


def add_rule(slide, x1, y1, x2, y2, color=GRID, width=0.8):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1),
        Inches(y1),
        Inches(x2),
        Inches(y2),
    )
    set_line(line, color, width)
    return line


def add_arrow(slide, x, y, w=0.12, color=SLATE):
    arrow = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(x), Inches(y), Inches(w), Inches(w))
    arrow.rotation = 90
    set_cell_fill(arrow, color)
    arrow.line.fill.background()
    return arrow


def add_dot(slide, x, y, d, fill, line=WHITE):
    dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    set_cell_fill(dot, fill)
    set_line(dot, line, 0.6)
    return dot


def add_matrix(slide, x, y, rows, cols, color, pale):
    size = 0.10
    gap = 0.012
    for r in range(rows):
        for c in range(cols):
            cell = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(x + c * (size + gap)),
                Inches(y + r * (size + gap)),
                Inches(size),
                Inches(size),
            )
            set_cell_fill(cell, color if (r + 2 * c) % 3 == 0 else pale)
            cell.line.fill.background()


def add_stage_header(slide, x, w, label, color):
    add_text(
        slide,
        x,
        0.10,
        w,
        0.18,
        label,
        size=9,
        color=color,
        bold=True,
        align=PP_ALIGN.CENTER,
    )
    add_rule(slide, x + 0.08, 0.31, x + w - 0.08, 0.31, color, 1.1)


def build(path: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE

    # The top row is reserved for stage labels; the method identity sits in
    # the motivation panel so no label competes for the same vertical space.
    add_text(slide, 0.08, 0.08, 1.30, 0.19, "WHY TWO SIGNALS?", size=9, bold=True)
    add_rule(slide, 0.16, 0.31, 1.22, 0.31, RED, 1.1)

    panel_y, panel_h = 0.37, 2.10
    left = (0.08, 1.22)
    s1 = (1.40, 1.04)
    s2 = (2.53, 1.38)
    s3 = (4.02, 1.61)
    s4 = (5.74, 1.04)

    # Motivation panel: two compact visual cues, no prose paragraph.
    add_box(slide, left[0], panel_y, left[1], panel_h, RED_PALE, RGBColor(241, 173, 163))
    add_text(slide, 0.16, 0.48, 1.06, 0.19, "weak token", size=9, color=RED, bold=True, align=PP_ALIGN.CENTER)
    for i in range(6):
        add_box(slide, 0.18 + i * 0.12, 0.73, 0.085, 0.085, RED if i == 2 else RGBColor(249, 206, 198), RED, False)
    add_text(slide, 0.16, 0.86, 1.06, 0.18, "local quality", size=9, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, 0.16, 1.11, 1.06, 0.18, "confidence ≠", size=9, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, 0.16, 1.30, 1.06, 0.18, "agreement", size=9, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_dot(slide, 0.28, 1.64, 0.16, GOLD)
    add_dot(slide, 0.52, 1.64, 0.16, TEAL)
    add_dot(slide, 0.76, 1.64, 0.16, RED)
    add_rule(slide, 0.44, 1.72, 0.52, 1.72, GRID, 1.0)
    add_rule(slide, 0.68, 1.72, 0.76, 1.72, GRID, 1.0)
    add_text(slide, 0.16, 1.88, 1.06, 0.18, "different states", size=9, color=MUTED, align=PP_ALIGN.CENTER)
    add_text(slide, 0.14, 2.12, 1.10, 0.28, "PWSC-CI\nkeeps both", size=9, color=SLATE, bold=True, align=PP_ALIGN.CENTER)

    # Stage 1: fixed target and confidence-weighted peers.
    add_stage_header(slide, s1[0], s1[1], "01  SAMPLES", GOLD)
    add_box(slide, 1.47, 0.52, 0.90, 0.42, GOLD_PALE, RGBColor(225, 193, 102))
    add_text(slide, 1.52, 0.57, 0.80, 0.16, "0  target", size=9, color=GOLD, bold=True, align=PP_ALIGN.CENTER)
    for i in range(4):
        add_box(slide, 1.55 + i * 0.12, 0.78, 0.08, 0.08, RGBColor(247, 215, 137), GOLD, False)
    add_box(slide, 1.47, 1.14, 0.90, 0.42, TEAL_PALE, RGBColor(153, 206, 201))
    add_text(slide, 1.50, 1.19, 0.84, 0.16, "1…N−1 peers", size=9, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    for i in range(4):
        add_box(slide, 1.55 + i * 0.12, 1.40, 0.08, 0.08, RGBColor(169, 220, 213), TEAL, False)
    add_text(slide, 1.46, 1.79, 0.88, 0.18, "N sample paths", size=9, color=MUTED, align=PP_ALIGN.CENTER)
    add_arrow(slide, 2.43, 1.24, 0.12, SLATE)

    # Stage 2: one quality equation and one task-kernel equation.
    add_stage_header(slide, s2[0], s2[1], "02  SIGNALS", RED)
    add_box(slide, 2.60, 0.47, 1.24, 0.72, RED_PALE, RGBColor(241, 173, 163))
    add_text(slide, 2.67, 0.52, 1.10, 0.16, "local quality", size=9, color=RED, bold=True, align=PP_ALIGN.CENTER)
    add_formula(
        slide,
        2.65,
        0.72,
        1.14,
        0.22,
        [[("q", None), ("i", "sub"), ("=v", None), ("i", "sub"), (" exp(−s", None), ("i", "sub"), (")", None)]],
        size=10,
    )
    add_text(slide, 2.67, 0.98, 1.10, 0.15, "validity × evidence", size=9, color=MUTED, align=PP_ALIGN.CENTER)
    add_box(slide, 2.60, 1.30, 1.24, 0.72, TEAL_PALE, RGBColor(153, 206, 201))
    add_text(slide, 2.67, 1.35, 1.10, 0.16, "task geometry", size=9, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_formula(slide, 2.65, 1.55, 1.14, 0.22, [[("K = ΦΦ", None), ("T", "sup")]], size=10)
    add_text(slide, 2.62, 1.82, 1.20, 0.15, "Math · Logic · Code", size=9, color=MUTED, align=PP_ALIGN.CENTER)
    add_arrow(slide, 3.88, 1.24, 0.12, SLATE)

    # Stage 3: normalization, paired kernels, and spectral contrast.
    add_stage_header(slide, s3[0], s3[1], "03  CONTRAST", GOLD)
    add_box(slide, 4.09, 0.47, 1.47, 0.55, GOLD_PALE, RGBColor(225, 193, 102))
    add_text(slide, 4.15, 0.52, 1.35, 0.16, "target α  ·  peers β", size=9, color=GOLD, bold=True, align=PP_ALIGN.CENTER)
    add_formula(
        slide,
        4.12,
        0.72,
        1.41,
        0.20,
        [[("w", None), ("i", "sub"), ("=q", None), ("i", "sub"), ("/Q   ·   π", None), ("j", "sub"), ("=q", None), ("j", "sub"), ("/Q", None), ("−0", "sub")]],
    )
    add_box(slide, 4.09, 1.12, 1.47, 0.60, TEAL_PALE, RGBColor(153, 206, 201))
    add_text(slide, 4.14, 1.17, 0.35, 0.16, "mix", size=9, color=GOLD, bold=True)
    add_matrix(slide, 4.52, 1.18, 2, 3, GOLD, RGBColor(246, 232, 183))
    add_text(slide, 4.14, 1.47, 0.35, 0.16, "peers", size=9, color=TEAL, bold=True)
    add_matrix(slide, 4.52, 1.48, 2, 3, TEAL, RGBColor(190, 228, 224))
    add_text(slide, 4.96, 1.28, 0.54, 0.16, "weighted", size=9, color=MUTED, align=PP_ALIGN.CENTER)
    add_text(slide, 4.96, 1.58, 0.54, 0.16, "target−0", size=9, color=MUTED, align=PP_ALIGN.CENTER)
    add_box(slide, 4.09, 1.88, 1.47, 0.43, TEAL_PALE, RGBColor(153, 206, 201))
    add_formula(
        slide,
        4.14,
        1.99,
        1.36,
        0.20,
        [[("χ", None), ("0", "sub"), (" = S", None), ("mix", "sub"), (" − β S", None), ("peer", "sub")]],
    )
    # One input fans out to both risk estimation and answer selection.
    add_rule(slide, 5.56, 1.28, 5.67, 1.28, SLATE, 0.9)
    add_rule(slide, 5.67, 0.86, 5.67, 2.11, SLATE, 0.9)
    add_arrow(slide, 5.67, 0.80, 0.12, SLATE)
    add_arrow(slide, 5.67, 2.05, 0.12, SLATE)

    # Stage 4: risk, the N=1 guard, and the self-consistent answer branch.
    add_stage_header(slide, s4[0], s4[1], "04  DECISION", TEAL)
    add_box(slide, 5.78, 0.47, 0.96, 0.78, GOLD_PALE, RGBColor(225, 193, 102))
    add_text(slide, 5.82, 0.52, 0.88, 0.15, "risk", size=9, color=GOLD, bold=True, align=PP_ALIGN.CENTER)
    add_formula(
        slide,
        5.82,
        0.70,
        0.88,
        0.17,
        [[("R", None), ("CI", "sub"), ("=clip", None), ("[0,1]", "sub")]],
    )
    add_formula(
        slide,
        5.82,
        0.89,
        0.88,
        0.17,
        [[("(χ", None), ("0", "sub"), ("/H", None), ("2", "sub"), ("(α))", None)]],
    )
    add_rule(slide, 5.98, 1.15, 6.53, 1.15, RGBColor(225, 193, 102), 1.2)
    add_dot(slide, 5.94, 1.11, 0.08, GOLD, GOLD)
    add_dot(slide, 6.49, 1.11, 0.08, TEAL, TEAL)
    add_box(slide, 5.78, 1.34, 0.96, 0.40, WHITE, RGBColor(225, 193, 102))
    add_formula(
        slide,
        5.82,
        1.39,
        0.88,
        0.30,
        [
            [("N=1: 1−q", None), ("0", "sub")],
            [("q", None), ("0", "sub"), ("Q", None), ("−0", "sub"), ("=0: 1", None)],
        ],
    )
    add_box(slide, 5.78, 1.82, 0.96, 0.62, TEAL_PALE, RGBColor(153, 206, 201))
    add_text(slide, 5.82, 1.85, 0.88, 0.15, "self-consistent", size=9, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_formula(
        slide,
        5.82,
        2.03,
        0.88,
        0.16,
        [[("i", None), ("*", "sup"), (" = argmax", None), ("i", "sub")]],
    )
    add_formula(
        slide,
        5.82,
        2.19,
        0.88,
        0.16,
        [[("q", None), ("i", "sub"), ("(Kw)", None), ("i", "sub")]],
    )

    # A compact second slide retains production facts that were previously in
    # the source deck's notes slide.  It is not included when exporting slide 1.
    notes = prs.slides.add_slide(blank)
    notes.background.fill.solid()
    notes.background.fill.fore_color.rgb = WHITE
    add_text(notes, 0.35, 0.35, 6.1, 0.35, "AAAI-27 production notes", size=14, bold=True)
    add_text(
        notes,
        0.40,
        0.95,
        6.0,
        1.1,
        "Final canvas: 17.42 × 6.60 cm\nFigure labels and formulas: 9 pt minimum at final size\nKeep slide 1 at 100% when exporting; use vector PDF or ≥300 dpi raster",
        size=10,
        color=MUTED,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.output)


if __name__ == "__main__":
    main()
