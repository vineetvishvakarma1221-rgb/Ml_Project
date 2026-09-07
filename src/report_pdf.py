from __future__ import annotations

import os
from io import BytesIO
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (getSampleStyleSheet,ParagraphStyle,)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    Image as RLImage,
    HRFlowable,
)

from report_generator import (
    prepare_report_dataframe,
    calculate_report_summ,
)

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"

# ============================================================
# REPORT CONSTANTS
# ============================================================

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT = 18 * mm
MARGIN_RIGHT = 18 * mm
MARGIN_TOP = 18 * mm
MARGIN_BOTTOM = 18 * mm

# --- role: panel / header backgrounds (deep electric navy) ---
PRIMARY = colors.HexColor("#0E1B3D")

# --- role: general / muted body text (soft slate-blue, readable on dark bg) ---
SECONDARY = colors.HexColor("#B8C6EC")

# --- role: headings, highlights, glowing borders (neon cyan) ---
ACCENT = colors.HexColor("#22E5FF")

# --- role: "Low risk" / "Normal" indicator (neon green) ---
SUCCESS = colors.HexColor("#39FF9E")

# --- role: "Medium risk" indicator (neon amber) ---
WARNING = colors.HexColor("#FFC94D")

# --- role: "High risk" / "Fraud" indicator (neon red/pink) ---
DANGER = colors.HexColor("#FF4C6A")

# --- role: secondary panel background used for zebra striping ---
LIGHT_BG = colors.HexColor("#11214A")

# --- role: table grid lines / card borders (soft glowing blue) ---
BORDER = colors.HexColor("#2C4C8C")

WHITE = colors.white

PAGE_BACKGROUND = colors.HexColor("#050B18")     # near-black page canvas
PANEL_BACKGROUND = colors.HexColor("#0A1430")    # slightly lighter card panel
ROW_DARK = colors.HexColor("#0A1430")            # zebra stripe - darker row
ROW_ALT = colors.HexColor("#101E46")             # zebra stripe - lighter row
GRID_LINE = colors.Color(0.13, 0.83, 1, alpha=0.06)   # faint circuit-grid lines
GLOW_LINE = colors.Color(0.13, 0.9, 1, alpha=0.35)    # brighter accent glow line
TEXT_MUTED = colors.HexColor("#7C93C4")          # dim captions / footnotes

# ============================================================
# MATPLOTLIB THEME
# ------------------------------------------------------------
# Hex twins of the ReportLab colours above, expressed as plain
# strings because Matplotlib does not understand ReportLab's
# colour objects. Keeping the charts colour-matched to the PDF
# theme is what makes them feel native to the report rather than
# "pasted in".
# ============================================================

MPL_BG = "#050B18"
MPL_PANEL = "#0A1430"
MPL_GRID = "#1B3A66"
MPL_TEXT = "#E7F0FF"
MPL_MUTED = "#7C93C4"
MPL_CYAN = "#22E5FF"
MPL_GREEN = "#39FF9E"
MPL_AMBER = "#FFC94D"
MPL_RED = "#FF4C6A"

# ============================================================
# STYLES
# ============================================================

styles = getSampleStyleSheet()
TITLE_STYLE = ParagraphStyle(
    "ReportTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=25,
    leading=30,
    alignment=TA_CENTER,
    textColor=ACCENT,          # neon cyan title glow instead of flat navy text
    spaceAfter=8,
)

SUBTITLE_STYLE = ParagraphStyle(
    "ReportSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=11,
    leading=16,
    alignment=TA_CENTER,
    textColor=SECONDARY,       # light slate-blue reads clearly on dark bg
    spaceAfter=8,
)

SECTION_STYLE = ParagraphStyle(
    "SectionHeading",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=19,
    textColor=ACCENT,          # neon cyan section headers
    spaceBefore=8,
    spaceAfter=6,
)

SUBSECTION_STYLE = ParagraphStyle(
    "SubSectionHeading",
    parent=styles["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=15,
    textColor=colors.HexColor("#7DD3FC"),   # slightly softer cyan-blue
    spaceBefore=6,
    spaceAfter=6,
)

BODY_STYLE = ParagraphStyle(
    "ReportBody",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=14,
    textColor=SECONDARY,       # light slate-blue body copy
    spaceAfter=6,
)

SMALL_STYLE = ParagraphStyle(
    "Small",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7.5,
    leading=10,
    textColor=TEXT_MUTED,
)

CENTER_STYLE = ParagraphStyle(
    "Center",
    parent=BODY_STYLE,
    alignment=TA_CENTER,
)

# ------------------------------------------------------------
# NEW: styles used only by the redesigned KPI "hologram" cards.
# ------------------------------------------------------------
KPI_LABEL_STYLE = ParagraphStyle(
    "KpiLabel",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=8.5,
    leading=11,
    alignment=TA_CENTER,
    textColor=TEXT_MUTED,
)

KPI_VALUE_STYLE = ParagraphStyle(
    "KpiValue",
    parent=styles["Normal"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    alignment=TA_CENTER,
    textColor=ACCENT,
)

# ============================================================
# BASIC HELPERS
# ------------------------------------------------------------
# Unchanged logic - these purely format/sanitize values and do
# not depend on colours or layout at all.
# ============================================================

def _safe(value, default="N/A"):
    """
    Convert missing/NaN values into safe printable values.
    """
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def _format_number(value, decimals=2):
    """
    Format numeric values safely.
    """
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(_safe(value))


def _format_percent(value):
    """
    Format percentage value.
    """
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _paragraph(text, style=BODY_STYLE):
    """
    Convert text to ReportLab Paragraph.
    """
    text = str(text)
    # Basic XML-safe replacements
    text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return Paragraph(text, style)


def _rich_paragraph(markup, style=BODY_STYLE):
    return Paragraph(str(markup), style)

# ============================================================
# CHART HELPERS (NEW)
# ============================================================

def _figure_to_rl_image(fig, width_mm=140, dpi=200):
    """
    Render a Matplotlib figure to an in-memory PNG and wrap it as
    a ReportLab Image flowable, scaled to a fixed width (in mm)
    while preserving the figure's native aspect ratio so charts
    never look stretched or squashed inside the PDF.
    """
    png_buffer = BytesIO()
    fig.savefig(
        png_buffer,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.18,
    )
    plt.close(fig)  # free the figure's memory immediately after saving
    png_buffer.seek(0)
    # Inspect native pixel dimensions so we can scale height to match
    # the requested width without distorting the chart.
    probe_image = PILImage.open(png_buffer)
    native_width, native_height = probe_image.size
    aspect_ratio = native_height / native_width

    png_buffer.seek(0)  # rewind after PIL read it

    target_width = width_mm * mm
    target_height = target_width * aspect_ratio

    return RLImage(png_buffer, width=target_width, height=target_height)

def _style_dark_axes(ax):
    """
    Apply the shared "electric" dark styling to a Matplotlib Axes
    object: dark face colour, muted tick colours, hidden spines.
    Kept as a small shared helper so all three charts stay visually
    consistent with each other and with the PDF theme.
    """
    ax.set_facecolor(MPL_BG)
    ax.tick_params(colors=MPL_MUTED, labelsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

def _build_risk_distribution_chart(summary):
    """
    Render a neon donut chart of the Low / Medium / High risk
    case counts, taken directly from the summary dictionary that
    calculate_report_summ() already produced (low_risk,
    medium_risk, high_risk, total_predictions).
    """
    labels = ["Low", "Medium", "High"]
    values = [
        summary["low_risk"],
        summary["medium_risk"],
        summary["high_risk"],
    ]
    chart_colors = [MPL_GREEN, MPL_AMBER, MPL_RED]
    # Avoid rendering a broken/empty pie when there is no data yet.
    plot_values = values if sum(values) > 0 else [1, 0, 0]

    fig, ax = plt.subplots(figsize=(4.6, 3.2), facecolor=MPL_BG)
    ax.set_facecolor(MPL_BG)
    wedges, _ = ax.pie(
        plot_values,
        colors=chart_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor=MPL_BG, linewidth=2),
    )
    # Soft outer "glow" ring behind each wedge - purely decorative,
    # reinforces the neon/electric look of the report.
    for wedge, color in zip(wedges, chart_colors):
        glow = mpatches.Wedge(
            wedge.center,
            wedge.r * 1.05,
            wedge.theta1,
            wedge.theta2,
            width=wedge.r * 0.42 * 1.2,
            facecolor=color,
            alpha=0.16,
            edgecolor="none",
        )
        ax.add_patch(glow)
    # Centre readout showing the total case count.
    ax.text(
        0, 0.10, f"{summary['total_predictions']:,}",
        ha="center", va="center",
        fontsize=19, fontweight="bold", color=MPL_TEXT,
    )
    ax.text(
        0, -0.16, "TOTAL CASES",
        ha="center", va="center",
        fontsize=8, fontweight="bold", color=MPL_MUTED,
    )

    legend_labels = [f"{lbl}   ({val:,})" for lbl, val in zip(labels, values)]

    legend = ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        labelcolor=MPL_TEXT,
    )

    ax.set_aspect("equal")
    fig.tight_layout()

    return _figure_to_rl_image(fig, width_mm=150)

def _build_case_volume_chart(summary):
    """
    Render a neon bar chart comparing Normal vs Fraud-flagged case
    counts, using summary['normal_cases'] and summary['fraud_cases']
    which are already computed upstream.
    """
    categories = ["Normal", "Fraud"]
    values = [summary["normal_cases"], summary["fraud_cases"]]
    bar_colors = [MPL_CYAN, MPL_RED]
    fig, ax = plt.subplots(figsize=(4.8, 3.0), facecolor=MPL_BG)
    _style_dark_axes(ax)
    bars = ax.bar(categories, values, color=bar_colors, width=0.5, zorder=3)
    # Glow effect: a wider, translucent bar drawn just behind each
    # real bar to simulate a soft neon light bleed.
    for bar, color in zip(bars, bar_colors):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (bar.get_x() - 0.04, 0),
                bar.get_width() + 0.08,
                max(bar.get_height(), 0.01),
                boxstyle="round,pad=0,rounding_size=0.03",
                linewidth=0,
                facecolor=color,
                alpha=0.18,
                zorder=2,
            )
        )

    # Value labels above each bar.
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center", va="bottom",
            color=MPL_TEXT, fontsize=10, fontweight="bold",
        )

    ax.set_ylabel("Number of Cases", color=MPL_MUTED, fontsize=8.5)
    ax.grid(axis="y", color=MPL_GRID, linewidth=0.6, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # A little headroom above the tallest bar so labels never clip.
    ax.set_ylim(0, max(values + [1]) * 1.2)

    fig.tight_layout()

    return _figure_to_rl_image(fig, width_mm=130)


def _build_probability_gauge_chart(probability_percent, risk_label):
    """
    Render a semi-circular neon gauge for a single customer's fraud
    probability (0-100), with colour bands for Low / Medium / High
    risk and a needle pointing at the current value. Used by the
    customer-level PDF only.
    """

    fig, ax = plt.subplots(
        figsize=(4.8, 2.8),
        subplot_kw={"aspect": "equal"},
        facecolor=MPL_BG,
    )
    ax.set_facecolor(MPL_BG)

    # Risk colour bands, matching the Low(<40) / Medium(<70) / High
    # thresholds used elsewhere in the system for risk labelling.
    bands = [
        (0, 40, MPL_GREEN),
        (40, 70, MPL_AMBER),
        (70, 100, MPL_RED),
    ]

    for start, end, color in bands:
        theta_start = 180 - (end / 100) * 180
        theta_end = 180 - (start / 100) * 180

        ax.add_patch(
            mpatches.Wedge(
                (0, 0), 1.0, theta_start, theta_end,
                width=0.28,
                facecolor=color,
                alpha=0.85,
                edgecolor=MPL_BG,
                linewidth=1.5,
            )
        )

    clamped_value = max(0.0, min(100.0, probability_percent))
    needle_angle = np.deg2rad(180 - (clamped_value / 100) * 180)
    needle_length = 0.82

    ax.plot(
        [0, needle_length * np.cos(needle_angle)],
        [0, needle_length * np.sin(needle_angle)],
        color=MPL_TEXT, linewidth=2.6, zorder=5, solid_capstyle="round",
    )

    ax.add_patch(mpatches.Circle((0, 0), 0.05, facecolor=MPL_TEXT, zorder=6))

    ax.text(
        0, -0.32, f"{clamped_value:.1f}%",
        ha="center", va="center",
        fontsize=19, fontweight="bold", color=MPL_TEXT,
    )
    ax.text(
        0, -0.52, f"RISK: {str(risk_label).upper()}",
        ha="center", va="center",
        fontsize=9, fontweight="bold", color=MPL_MUTED,
    )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.65, 1.05)
    ax.axis("off")

    fig.tight_layout()

    return _figure_to_rl_image(fig, width_mm=115)


# ============================================================
# PAGE BACKGROUND / HUD DECORATION HELPERS (NEW)
# ============================================================

def _draw_circuit_grid(canvas):
    """
    Draw a very faint circuit-board style grid across the full page
    to reinforce the "electricity system" theme without competing
    with the report's text and tables.
    """
    canvas.saveState()
    canvas.setStrokeColor(GRID_LINE)
    canvas.setLineWidth(0.4)

    step = 12 * mm

    x = 0.0
    while x <= PAGE_WIDTH:
        canvas.line(x, 0, x, PAGE_HEIGHT)
        x += step

    y = 0.0
    while y <= PAGE_HEIGHT:
        canvas.line(0, y, PAGE_WIDTH, y)
        y += step

    # Small "circuit node" dots at a subset of intersections, giving
    # the grid a printed-circuit-board feel rather than plain graph
    # paper.
    canvas.setFillColor(GRID_LINE)

    row_index = 0
    y = 0.0
    while y <= PAGE_HEIGHT:
        col_index = 0
        x = 0.0
        while x <= PAGE_WIDTH:
            if (row_index + col_index) % 4 == 0:
                canvas.circle(x, y, 0.5, stroke=0, fill=1)
            x += step
            col_index += 1
        y += step
        row_index += 1

    canvas.restoreState()


def _draw_hud_corners(canvas):
    """
    Draw small sci-fi "HUD" style corner brackets near each page
    corner, inside the margins, for a futuristic dashboard feel.
    """
    canvas.saveState()
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1.1)

    bracket = 6 * mm
    inset = 6 * mm

    corners = [
        (inset, PAGE_HEIGHT - inset, 1, -1),               # top-left
        (PAGE_WIDTH - inset, PAGE_HEIGHT - inset, -1, -1),  # top-right
        (inset, inset, 1, 1),                               # bottom-left
        (PAGE_WIDTH - inset, inset, -1, 1),                 # bottom-right
    ]

    for x, y, dx, dy in corners:
        canvas.line(x, y, x + dx * bracket, y)
        canvas.line(x, y, x, y + dy * bracket)

    canvas.restoreState()


def _bolt_icon_points(scale=1.0):
    """
    Return the (x, y) vertices of a simple lightning-bolt silhouette,
    defined on a 24x24 unit grid and scaled by `scale`. Shared by the
    canvas-drawn header icon and the flowable Drawing icon so both
    versions of the bolt look identical.
    """
    raw_points = [13, 22, 3, 10, 12, 10, 11, 2, 21, 14, 12, 14]
    return [coord * scale for coord in raw_points]


def _draw_bolt_icon(canvas, x, y, size, color):
    """
    Draw a small filled lightning-bolt icon directly on the canvas at
    position (x, y), sized `size` (in points). Used in the page
    header/footer instead of a Unicode "⚡" character, because the
    core Helvetica font used throughout this report has no glyph for
    that symbol and would otherwise render a broken tofu box.
    """
    scale = size / 24.0
    points = _bolt_icon_points(scale)
    path = canvas.beginPath()
    for index in range(0, len(points), 2):
        px = x + points[index]
        py = y + points[index + 1]
        if index == 0:
            path.moveTo(px, py)
        else:
            path.lineTo(px, py)

    path.close()
    canvas.setFillColor(color)
    canvas.drawPath(path, fill=1, stroke=0)


def _lightning_bolt_drawing(size_mm=11, color=ACCENT):
    """
    Build a small centred lightning-bolt Drawing flowable for use on
    cover/title pages, replacing the Unicode "⚡" character (which the
    core Helvetica font cannot render) with a real vector icon.
    """
    from reportlab.graphics.shapes import Drawing, Polygon
    size_pt = size_mm * mm
    scale = size_pt / 24.0
    points = _bolt_icon_points(scale)
    drawing = Drawing(size_pt, size_pt)
    drawing.add(
        Polygon(points, fillColor=color, strokeColor=color, strokeWidth=0.4)
    )
    drawing.hAlign = "CENTER"
    return drawing


def _draw_gradient_bar(canvas, x, y, width, height, start_color, end_color, steps=48):
    """
    Simulate a horizontal colour gradient bar by painting many thin
    vertical strips whose colour is linearly interpolated between
    start_color and end_color. ReportLab has no native gradient
    fill for simple rects, so this approximation gives the header
    accent bar a "glowing" multi-colour look.
    """
    canvas.saveState()
    canvas.setStrokeColor(colors.transparent)
    strip_width = width / steps
    for i in range(steps):
        t = i / max(steps - 1, 1)
        r = start_color.red + (end_color.red - start_color.red) * t
        g = start_color.green + (end_color.green - start_color.green) * t
        b = start_color.blue + (end_color.blue - start_color.blue) * t
        canvas.setFillColor(colors.Color(r, g, b))
        canvas.rect(x + i * strip_width, y, strip_width + 0.4, height, stroke=0, fill=1)
    canvas.restoreState()

# ============================================================
# PAGE HEADER / FOOTER
# ============================================================

def _draw_page_header_footer(canvas, doc):
    """
    Draw the futuristic dark-theme header/footer on every page:
    a full-page dark background, a faint circuit grid, HUD corner
    brackets, a neon gradient accent bar, and the same brand/page
    text the original report used (page numbering logic unchanged).
    """
    canvas.saveState()
    # --------------------------------------------------------
    # Full-page dark background (drawn first, everything else on top)
    # --------------------------------------------------------
    canvas.setFillColor(PAGE_BACKGROUND)
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
    _draw_circuit_grid(canvas)
    _draw_hud_corners(canvas)
    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    # Slim neon gradient accent strip along the very top edge.
    _draw_gradient_bar(
        canvas,
        x=0,
        y=PAGE_HEIGHT - 2 * mm,
        width=PAGE_WIDTH,
        height=2 * mm,
        start_color=ACCENT,
        end_color=colors.HexColor("#7C3AED"),
    )
    canvas.setStrokeColor(GLOW_LINE)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN_LEFT,PAGE_HEIGHT - 12 * mm,PAGE_WIDTH - MARGIN_RIGHT,PAGE_HEIGHT - 12 * mm,)
    canvas.setFont("Helvetica-Bold",7.5,)
    canvas.setFillColor(ACCENT)
    _draw_bolt_icon(
        canvas,
        x=MARGIN_LEFT,
        y=PAGE_HEIGHT - 11.2 * mm,
        size=3.2 * mm,
        color=ACCENT,
    )
    canvas.drawString(
        MARGIN_LEFT + 4.4 * mm,
        PAGE_HEIGHT - 9 * mm,
        "ELECTRICITY FRAUD DETECTION SYSTEM",
    )
    canvas.setFont("Helvetica",7,)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawRightString(
        PAGE_WIDTH - MARGIN_RIGHT,
        PAGE_HEIGHT - 9 * mm,
        "Machine Learning Risk Assessment",
    )
    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------
    canvas.setStrokeColor(GLOW_LINE)
    canvas.line(
        MARGIN_LEFT,12 * mm,
        PAGE_WIDTH - MARGIN_RIGHT,12 * mm,
    )
    canvas.setFont("Helvetica",7,)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(
        MARGIN_LEFT,8 * mm,
        "Generated by Electricity Fraud Detection System",
    )

    canvas.setFillColor(ACCENT)
    canvas.setFont("Helvetica-Bold", 7)
    canvas.drawRightString(
        PAGE_WIDTH - MARGIN_RIGHT,
        8 * mm,
        f"Page {doc.page}",
    )
    canvas.restoreState()

# ============================================================
# COVER PAGE
# ============================================================

def _build_cover(story):
    """
    Build professional report cover.
    """
    story.append(Spacer(1,25 * mm,))
    # --------------------------------------------------------
    # Logo
    # --------------------------------------------------------
    if LOGO_PATH.exists():
        try:
            from reportlab.platypus import Image
            logo = Image(str(LOGO_PATH))
            logo.drawHeight = 30 * mm
            logo.drawWidth = 30 * mm
            logo.hAlign = "CENTER"
            story.append(logo)
            story.append(Spacer(1,8 * mm,))
        except Exception:
            pass

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------
    story.append(_lightning_bolt_drawing(size_mm=13))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "ELECTRICITY FRAUD<br/>DETECTION SYSTEM",
            TITLE_STYLE,
        )
    )
    story.append(
        Paragraph(
            "Machine Learning Risk Assessment Report",
            SUBTITLE_STYLE,
        )
    )
    # Thin neon divider under the title/subtitle block for extra polish.
    story.append(Spacer(1, 4 * mm))
    story.append(
        HRFlowable(
            width="35%",
            thickness=1.2,
            color=ACCENT,
            spaceAfter=0,
            hAlign="CENTER",
        )
    )

    story.append(Spacer(1,15 * mm,))

    # --------------------------------------------------------
    # Information card
    # --------------------------------------------------------

    generated_at = datetime.now().strftime("%d %B %Y, %H:%M")
    cover_data = [
        [
            _rich_paragraph("<b>Model</b>",CENTER_STYLE,),
            _rich_paragraph("Random Forest Classifier",CENTER_STYLE,),
        ],
        [
            _rich_paragraph("<b>Explainability</b>",CENTER_STYLE,),
            _rich_paragraph("SHAP",CENTER_STYLE,),
        ],
        [
            _rich_paragraph("<b>Report Generated</b>",CENTER_STYLE,),
            _rich_paragraph(generated_at,CENTER_STYLE,),
        ],
        [
            _rich_paragraph("<b>Developer</b>",CENTER_STYLE,),
            _rich_paragraph("Vineet Vishvakarma",CENTER_STYLE,),
        ],
    ]

    table = Table(cover_data,colWidths=[55 * mm,85 * mm,],)
    table.setStyle(
        TableStyle([
                ("BACKGROUND",(0, 0),(-1, -1),PANEL_BACKGROUND,),
                ("TEXTCOLOR",(0, 0),(-1, -1),SECONDARY,),
                ("BOX",(0, 0),(-1, -1),0.8,ACCENT,),
                ("INNERGRID",(0, 0),(-1, -1),0.4,BORDER,),
                ("VALIGN",(0, 0),(-1, -1),"MIDDLE",),
                ("TOPPADDING",(0, 0),(-1, -1),9,),
                ("BOTTOMPADDING",(0, 0),(-1, -1),9,),
            ]
        )
    )
    table.hAlign = "CENTER"
    story.append(table)
    story.append(Spacer(1,25 * mm,))
    story.append(
        Paragraph(
            "This report summarizes prediction outcomes, "
            "fraud risk distribution and model-generated "
            "risk indicators.",
            SUBTITLE_STYLE,
        )
    )
    story.append(PageBreak())

# ============================================================
# KPI TABLE
# ============================================================
def _build_kpi_table(summary):
    """
    Build KPI summary cards.
    """
    data = [
        [
            _rich_paragraph(
                "<b>Total Predictions</b><br/>"
                f"<font size='16'>{summary['total_predictions']:,}</font>",
                CENTER_STYLE,
            ),
            _rich_paragraph(
                "<b>Fraud Cases</b><br/>"
                f"<font size='16'>{summary['fraud_cases']:,}</font>",
                CENTER_STYLE,
            ),
        ],
        [
            _rich_paragraph(
                "<b>Normal Cases</b><br/>"
                f"<font size='16'>{summary['normal_cases']:,}</font>",
                CENTER_STYLE,
            ),
            _rich_paragraph(
                "<b>Fraud Rate</b><br/>"
                f"<font size='16'>{summary['fraud_rate']:.2f}%</font>",
                CENTER_STYLE,
            ),
        ],
    ]

    table = Table(data,
        colWidths=[82 * mm,82 * mm,],
        rowHeights=[25 * mm,25 * mm,],)

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND",(0, 0),(-1, -1),PANEL_BACKGROUND,),
                ("TEXTCOLOR",(0, 0),(-1, -1),SECONDARY,),
                ("BOX",(0, 0),(-1, -1),0.8,ACCENT,),
                ("INNERGRID",(0, 0),(-1, -1),0.5,BORDER,),
                ("VALIGN",(0, 0),(-1, -1),"MIDDLE",),
                ("ALIGN",(0, 0),(-1, -1),"CENTER",),
            ]
        )
    )
    return table

# ============================================================
# RISK DISTRIBUTION
# ============================================================
def _build_risk_table(summary):
    """
    Build risk distribution table.
    """
    total = summary["total_predictions"]
    def risk_percent(value):
        if total == 0:
            return 0
        return (value / total) * 100
    data = [
        [
            _rich_paragraph("<b>Risk Level</b>", CENTER_STYLE),
            _rich_paragraph("<b>Cases</b>", CENTER_STYLE),
            _rich_paragraph("<b>Share</b>", CENTER_STYLE),
        ],
        [
            _rich_paragraph(
                "<font color='#39FF9E'><b>Low</b></font>",
                CENTER_STYLE,
            ),
            _rich_paragraph(
                str(summary["low_risk"]),
                CENTER_STYLE,
            ),
            _rich_paragraph(
                _format_percent(
                    risk_percent(
                        summary["low_risk"]
                    )
                ),
                CENTER_STYLE,
            ),
        ],
        [
            _rich_paragraph(
                "<font color='#FFC94D'><b>Medium</b></font>",
                CENTER_STYLE,
            ),
            _rich_paragraph(
                str(summary["medium_risk"]),
                CENTER_STYLE,
            ),
            _rich_paragraph(
                _format_percent(
                    risk_percent(
                        summary["medium_risk"]
                    )
                ),
                CENTER_STYLE,
            ),
        ],
        [
            _rich_paragraph(
                "<font color='#FF4C6A'><b>High</b></font>",
                CENTER_STYLE,
            ),
            _rich_paragraph(
                str(summary["high_risk"]),
                CENTER_STYLE,
            ),
            _rich_paragraph(
                _format_percent(
                    risk_percent(
                        summary["high_risk"]
                    )
                ),
                CENTER_STYLE,
            ),
        ],
    ]
    table = Table(data,colWidths=[60 * mm,50 * mm,50 * mm,],)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND",(0, 0),(-1, 0),PRIMARY,),
                ("TEXTCOLOR",(0, 0),(-1, 0),WHITE,),
                ("GRID",(0, 0),(-1, -1),0.5,BORDER,),
                ("BOX",(0, 0),(-1, -1),0.8,ACCENT,),
                ("VALIGN",(0, 0),(-1, -1),"MIDDLE",),
                ("TOPPADDING",(0, 0),(-1, -1),7,),
                ("BOTTOMPADDING",(0, 0),(-1, -1),7,),
                ("TEXTCOLOR",(0, 1),(-1, -1),SECONDARY,),
                ("ROWBACKGROUNDS",(0, 1),(-1, -1),[ROW_DARK,ROW_ALT,],),
            ]
        )
    )
    return table

# ============================================================
# PREDICTION TABLE
# ============================================================
def _build_prediction_table(df, max_rows=25):
    """
    Add a compact detailed prediction table.
    """
    if df.empty:
        return Paragraph("No prediction records available.",BODY_STYLE,)
    working = df.head(max_rows).copy()
    rows = [["Prediction","Probability","Risk","Timestamp",]]
    # Track which data rows represent a fraud prediction so the table
    # style below can tint those rows for quick visual scanning. This
    # is presentation-only bookkeeping - it does not change what is
    # written into any cell.
    fraud_row_indices = []
    for row_offset, (_, row) in enumerate(working.iterrows(), start=1):
        prediction = row.get("prediction","N/A")
        if prediction == 1:
            prediction = "Fraud"
            fraud_row_indices.append(row_offset)
        elif prediction == 0:
            prediction = "Normal"

        probability = row.get("fraud_pobability",None)
        try:
            probability = (float(probability) * 100)
            probability_text = (f"{probability:.2f}%")
        except (TypeError,ValueError,):
            probability_text = "N/A"

        risk = row.get("risk","N/A")
        timestamp = row.get("prediction_time","N/A")
        if pd.notna(timestamp):
            timestamp = str(timestamp)
        rows.append(
            [str(prediction),probability_text,str(risk),str(timestamp),]
        )
    table = Table(rows,colWidths=[35 * mm,35 * mm,30 * mm,65 * mm,],repeatRows=1,)
    table_style_commands = [
        ("BACKGROUND",(0, 0),(-1, 0),PRIMARY,),
        ("TEXTCOLOR",(0, 0),(-1, 0),WHITE,),
        ("FONTNAME",(0, 0),(-1, 0),"Helvetica-Bold",),
        ("FONTSIZE",(0, 0),(-1, -1),7.5,),
        ("GRID",(0, 0),(-1, -1),0.4,BORDER,),
        ("BOX",(0, 0),(-1, -1),0.8,ACCENT,),
        ("VALIGN",(0, 0),(-1, -1),"MIDDLE",),
        ("TEXTCOLOR",(0, 1),(-1, -1),SECONDARY,),
        ("ROWBACKGROUNDS",(0, 1),(-1, -1),[ROW_DARK,ROW_ALT,],),        
        ("TOPPADDING",(0, 0),(-1, -1),5,),
        ("BOTTOMPADDING",(0, 0),(-1, -1),5,),        
    ]

    # Tint the "Prediction" cell of every fraud row in the danger
    # colour so high-risk rows are instantly scannable in a long
    # table, without altering the underlying text.
    for fraud_row in fraud_row_indices:
        table_style_commands.append(
            ("TEXTCOLOR", (0, fraud_row), (0, fraud_row), DANGER)
        )
        table_style_commands.append(
            ("FONTNAME", (0, fraud_row), (0, fraud_row), "Helvetica-Bold")
        )

    table.setStyle(TableStyle(table_style_commands))
    return table

# ============================================================
# EXECUTIVE PDF
# ============================================================
def generate_executive_pdf(
    data,
    report_title="Electricity Fraud Detection - Executive Report",
):
    """
    Generate an executive PDF report.
    Returns:bytes
    """
    df = prepare_report_dataframe(data)
    summary = calculate_report_summ(df)
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=MARGIN_RIGHT,
        leftMargin=MARGIN_LEFT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title=report_title,
        author="Vineet Vishvakarma",
    )
    story = []
    # --------------------------------------------------------
    # Cover
    # --------------------------------------------------------
    _build_cover(story)
    # --------------------------------------------------------
    # Executive Summary
    # --------------------------------------------------------
    story.append(
        Paragraph("1. Executive Summary",SECTION_STYLE,)
    )

    story.append(
        Paragraph(
            "This section provides a high-level overview "
            "of the predictions generated by the Electricity "
            "Fraud Detection System.",
            BODY_STYLE,
        )
    )

    story.append(_build_kpi_table(summary))
    story.append(Spacer(1,10 * mm,))
    # --------------------------------------------------------
    # NEW: Case volume chart - a purely visual bar-chart summary
    # of summary['normal_cases'] vs summary['fraud_cases'], the
    # same two numbers already shown in the KPI cards above.
    # --------------------------------------------------------
    story.append(
        Paragraph(
            "Case Volume Overview",
            SUBSECTION_STYLE,
        )
    )
    story.append(_build_case_volume_chart(summary))
    story.append(Spacer(1,8 * mm,))
    # --------------------------------------------------------
    # Additional metrics
    # --------------------------------------------------------
    story.append(
        Paragraph(
            "Key Indicators",
            SUBSECTION_STYLE,
        )
    )
    indicators = [
        ["Metric","Value",],
        ["Average Fraud Probability",
        _format_percent(
                summary[
                    "average_fraud_probability"
                ]
            ),
        ],
        ["Low Risk Cases",f"{summary['low_risk']:,}",],
        ["Medium Risk Cases",f"{summary['medium_risk']:,}",],
        ["High Risk Cases",f"{summary['high_risk']:,}",],        
    ]

    indicator_table = Table(indicators,
        colWidths=[95 * mm,65 * mm,],repeatRows=1,
    )

    indicator_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND",(0, 0),(-1, 0),PRIMARY,),
                ("TEXTCOLOR",(0, 0),(-1, 0),WHITE,), 
                ("GRID",(0, 0),(-1, -1),0.5,BORDER,),    
                ("BOX",(0, 0),(-1, -1),0.8,ACCENT,),
                ("VALIGN",(0, 0),(-1, -1),"MIDDLE",),
                ("TEXTCOLOR",(0, 1),(-1, -1),SECONDARY,),
                ("ROWBACKGROUNDS",(0, 1),(-1, -1),[ROW_DARK,ROW_ALT,],),        
                ("TOPPADDING",(0, 0),(-1, -1),6,),
                ("BOTTOMPADDING",(0, 0),(-1, -1),6,),
            ]
        )
    )
    story.append(indicator_table)
    story.append(PageBreak())
    # --------------------------------------------------------
    # Risk Analysis
    # --------------------------------------------------------
    story.append(
        Paragraph("2. Risk Distribution",SECTION_STYLE,))
    story.append(
        Paragraph(
            "The following distribution summarizes the "
            "number of predictions classified into each "
            "risk category.",
            BODY_STYLE,
        ))
    story.append(_build_risk_table(summary))
    story.append(Spacer(1,8 * mm,))
    # --------------------------------------------------------
    # NEW: Risk distribution donut chart - a purely visual
    # rendering of the same low_risk / medium_risk / high_risk
    # numbers shown in the table immediately above it.
    # --------------------------------------------------------

    story.append(_build_risk_distribution_chart(summary))
    story.append(Spacer(1,10 * mm,))
    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------
    story.append(
        Paragraph("3. Model Information",SECTION_STYLE,)
    )
    model_info = [
        ["Component","Details",],
        ["Machine Learning Model","Random Forest Classifier",],
        ["Explainability","SHAP",],
        ["Prediction Type","Binary Classification",],  
        ["Risk Levels","Low / Medium / High",],
        ["Application","Streamlit",],
        ["Database","MySQL",],     
    ]

    model_table = Table(model_info,colWidths=[65 * mm,95 * mm,],repeatRows=1,)
    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND",(0, 0),(-1, 0),PRIMARY,),
                ("TEXTCOLOR",(0, 0),(-1, 0),WHITE,), 
                ("GRID",(0, 0),(-1, -1),0.5,BORDER,),    
                ("BOX",(0, 0),(-1, -1),0.8,ACCENT,),
                ("VALIGN",(0, 0),(-1, -1),"MIDDLE",),
                ("TEXTCOLOR",(0, 1),(-1, -1),SECONDARY,),
                ("ROWBACKGROUNDS",(0, 1),(-1, -1),[ROW_DARK,ROW_ALT,],),        
                ("TOPPADDING",(0, 0),(-1, -1),6,),
                ("BOTTOMPADDING",(0, 0),(-1, -1),6,),
            ]
        )
    )

    story.append(model_table)
    story.append(Spacer(1,10 * mm,))
    # --------------------------------------------------------
    # Feature information
    # --------------------------------------------------------
    story.append(
        Paragraph("4. Model Features",SECTION_STYLE,)
    )
    features = [
        "Standard Deviation",
        "Stability",
        "Coefficient of Variance",
        "Longest Missing Streak",
        "High Consumption Ratio",
        "Peak-to-Average Ratio",
        "Zero Consumption Ratio",
        "Longest Zero Streak",
    ]
    feature_rows = [["No.","Feature",]]
    for index, feature in enumerate(features,start=1,):
        feature_rows.append([str(index),feature,])
    feature_table = Table(feature_rows,colWidths=[20 * mm,140 * mm,],repeatRows=1,)
    feature_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND",(0, 0),(-1, 0),PRIMARY,),
                ("TEXTCOLOR",(0, 0),(-1, 0),WHITE,), 
                ("GRID",(0, 0),(-1, -1),0.5,BORDER,),    
                ("BOX",(0, 0),(-1, -1),0.8,ACCENT,),
                ("VALIGN",(0, 0),(-1, -1),"MIDDLE",),
                ("TEXTCOLOR",(0, 1),(-1, -1),SECONDARY,),
                ("ROWBACKGROUNDS",(0, 1),(-1, -1),[ROW_DARK,ROW_ALT,],),        
                ("TOPPADDING",(0, 0),(-1, -1),6,),
                ("BOTTOMPADDING",(0, 0),(-1, -1),6,),
            ]
        )
    )
    story.append(feature_table)
    story.append(PageBreak())
    # --------------------------------------------------------
    # Prediction records
    # --------------------------------------------------------
    story.append(
        Paragraph(
            "5. Prediction Records",
            SECTION_STYLE,
        )
    )

    story.append(
        Paragraph(
            "A sample of the latest prediction records "
            "is included below. The complete dataset remains "
            "available through the CSV export.",
            BODY_STYLE,
        )
    )

    story.append(
        _build_prediction_table(
            df,
            max_rows=25,
        )
    )
    story.append(Spacer(1,10 * mm,))
    # --------------------------------------------------------
    # Conclusion
    # --------------------------------------------------------
    story.append(
        Paragraph("6. Conclusion",SECTION_STYLE,)
    )
    story.append(
        Paragraph(
            "The Electricity Fraud Detection System uses "
            "machine learning to identify potentially "
            "abnormal electricity consumption patterns. "
            "The generated risk categories provide a "
            "structured way to prioritize potentially "
            "suspicious cases for further investigation. "
            "Predictions should be treated as risk indicators "
            "rather than definitive proof of electricity theft.",
            BODY_STYLE,
        )
    )

    story.append(Spacer(1, 8 * mm,))
    story.append(
        Paragraph(
            "Report generated automatically by the "
            "Electricity Fraud Detection System.",
            SMALL_STYLE,
        )
    )
    # --------------------------------------------------------
    # Build PDF
    # --------------------------------------------------------
    doc.build(
        story,
        onFirstPage=_draw_page_header_footer,
        onLaterPages=_draw_page_header_footer,
    )
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# BATCH PDF
# ============================================================
def generate_batch_pdf(result_df,
    report_title="Electricity Fraud Detection - Batch Report",
):
    """
    Generate a PDF report for batch prediction results.
    Uses the same executive report engine while clearly
    identifying the report as a batch assessment.
    """
    return generate_executive_pdf(
        result_df,
        report_title=report_title,
    )

# ============================================================
# CUSTOMER PDF
# ============================================================
def generate_customer_pdf(
    customer_data,
    prediction_result,
):
    """
    Generate a single-customer PDF report.
    """
    if isinstance(customer_data, dict):
        customer_df = pd.DataFrame([customer_data])
    elif isinstance(customer_data,pd.DataFrame,):
        customer_df = customer_data.copy()
    else:
        customer_df = pd.DataFrame([customer_data])
    # --------------------------------------------------------
    # Create combined record
    # --------------------------------------------------------
    result = (
        prediction_result
        if isinstance(prediction_result,dict,)
        else {}
    )
    row = {}
    if not customer_df.empty:
        row.update(customer_df.iloc[0].to_dict())
    row.update(result)
    combined_df = pd.DataFrame([row])
    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------
    combined_df = prepare_report_dataframe(combined_df)
    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary = calculate_report_summ(combined_df)
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=MARGIN_RIGHT,
        leftMargin=MARGIN_LEFT,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title="Customer Fraud Risk Report",
        author="Vineet Vishvakarma",
    )
    story = []
    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------
    story.append(Spacer(1,12 * mm,))
    story.append(_lightning_bolt_drawing(size_mm=11))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "CUSTOMER FRAUD RISK REPORT",
            TITLE_STYLE,
        )
    )
    story.append(
        Paragraph(
            "Individual Machine Learning Assessment",
            SUBTITLE_STYLE,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        HRFlowable(
            width="35%",
            thickness=1.2,
            color=ACCENT,
            spaceAfter=0,
            hAlign="CENTER",
        )
    )
    story.append(Spacer(1,8 * mm,))
    # --------------------------------------------------------
    # Customer ID
    # --------------------------------------------------------
    consumer_id = "N/A"
    for column in ["CONS_NO","consumer_id","Consumer ID",]:
        if column in combined_df.columns:
            consumer_id = combined_df[column].iloc[0]
            break
    story.append(
        Paragraph(
            f"<b>Consumer ID:</b> "
            f"{consumer_id}",
            BODY_STYLE,
        )
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------
    prediction = result.get("prediction")
    if prediction == 1:
        prediction_label = "FRAUD SUSPECTED"
    elif prediction == 0:
        prediction_label = "NORMAL"
    else:
        prediction_label = str(prediction)
    probability = result.get(
        "fraud_pobability",
        result.get(
            "fraud_probability"
        ),
    )

    try:
        probability_percent = (float(probability) * 100 )
    except (TypeError,ValueError,):
        probability_percent = 0

    risk = result.get("risk","Unknown",)
    result_table = Table(
        [
            [
                _rich_paragraph("<b>Prediction</b>",CENTER_STYLE,),
                _rich_paragraph(prediction_label,CENTER_STYLE,),      
            ],
            [
                _rich_paragraph("<b>Fraud Probability</b>",CENTER_STYLE,),
                _rich_paragraph(f"{probability_percent:.2f}%",CENTER_STYLE,),      
            ],
            [
                _rich_paragraph("<b>Risk Level</b>",CENTER_STYLE,),
                _rich_paragraph(str(risk).upper(),CENTER_STYLE,),    
            ],
            [
                _rich_paragraph("<b>Generated At</b>",CENTER_STYLE,),
                _rich_paragraph(datetime.now().strftime("%d %b %Y, %H:%M:%S"),CENTER_STYLE,),         
            ],
        ],
        colWidths=[70 * mm,90 * mm,],     
    )

    result_table.setStyle(
        TableStyle(
            [   
                ("BACKGROUND",(0, 0),(-1, -1),PANEL_BACKGROUND,),
                ("TEXTCOLOR",(0, 0),(-1, -1),SECONDARY,),     
                ("BOX",(0, 0),(-1, -1),0.8,ACCENT,),
                ("VALIGN",(0, 0),(-1, -1),"MIDDLE",),       
                ("TOPPADDING",(0, 0),(-1, -1),9,),
                ("BOTTOMPADDING",(0, 0),(-1, -1),9,),
                ("INNERGRID",(0, 0),(-1, -1),0.4,BORDER,),            
            ]
        )
    )
    story.append(result_table)
    story.append(Spacer(1,8 * mm,))
    # --------------------------------------------------------
    # NEW: Probability gauge chart - a purely visual rendering
    # of probability_percent and risk, the same two values shown
    # in the result table immediately above it.
    # --------------------------------------------------------

    story.append(_build_probability_gauge_chart(probability_percent, risk))
    story.append(Spacer(1,8 * mm,))
    # --------------------------------------------------------
    # Feature analysis
    # --------------------------------------------------------

    story.append(
        Paragraph("Feature Assessment",SECTION_STYLE,)      
    )

    feature_rows = [["Feature","Value",]]
    feature_names = [
        "std_cons",
        "Stability",
        "cv",
        "longest_missing_streak",
        "high_cons_ratio",
        "PAR",
        "zero_ratio",
        "longest_zero_streak",
    ]

    for feature in feature_names:
        if feature in combined_df.columns:
            value = combined_df[feature].iloc[0]
            feature_rows.append(
                [feature,_format_number(value),]        
            )

    feature_table = Table(feature_rows,colWidths=[95 * mm,65 * mm,],repeatRows=1,)

    feature_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND",(0, 0),(-1, 0),PRIMARY,),
                ("TEXTCOLOR",(0, 0),(-1, 0),WHITE,),
                ("GRID",(0, 0),(-1, -1),0.5,BORDER,),
                ("BOX",(0, 0),(-1, -1),0.8,ACCENT,),
                ("TOPPADDING",(0, 0),(-1, -1),6,),
                ("BOTTOMPADDING",(0, 0),(-1, -1),6,),
                ("TEXTCOLOR",(0, 1),(-1, -1),SECONDARY,),
                ("ROWBACKGROUNDS",(0, 1),(-1, -1),[ROW_DARK,ROW_ALT,],),
            ]
        )
    )

    story.append(feature_table)
    story.append(Spacer(1,10 * mm,))
    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Assessment Interpretation",
            SECTION_STYLE,
        )
    )

    if prediction == 1:
        interpretation = (
            "The machine learning model has classified "
            "this consumption profile as potentially "
            "fraudulent. The case should be reviewed "
            "further using operational and customer-level "
            "evidence."
        )

    else:
        interpretation = (
            "The machine learning model has classified "
            "this consumption profile as normal based "
            "on the supplied engineered features. "
            "This classification is a model assessment "
            "and does not constitute a guarantee."
        )
    story.append(Paragraph(interpretation,BODY_STYLE,)) 
    story.append(Spacer(1,12 * mm,))
    story.append(
        Paragraph(
            "Important: Machine learning predictions are "
            "risk indicators and should not be treated as "
            "conclusive proof of electricity theft.",
            SMALL_STYLE,
        )
    )

    doc.build(
        story,
        onFirstPage=_draw_page_header_footer,
        onLaterPages=_draw_page_header_footer,
    )

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("PDF REPORT ENGINE TEST")
    print("=" * 60)

    sample_data = pd.DataFrame(
        [
            {
                "CONS_NO": "TEST001",
                "std_cons": 3.5,
                "Stability": 0.82,
                "cv": 0.31,
                "longest_missing_streak": 4,
                "high_cons_ratio": 0.61,
                "PAR": 3.5,
                "zero_ratio": 0.12,
                "longest_zero_streak": 2,
                "fraud_pobability": 0.82,
                "prediction": 1,
                "risk": "High",
                "prediction_time": datetime.now(),
            },
            {
                "CONS_NO": "TEST002",
                "std_cons": 1.5,
                "Stability": 0.91,
                "cv": 0.12,
                "longest_missing_streak": 1,
                "high_cons_ratio": 0.21,
                "PAR": 1.8,
                "zero_ratio": 0.03,
                "longest_zero_streak": 1,
                "fraud_pobability": 0.18,
                "prediction": 0,
                "risk": "Low",
                "prediction_time": datetime.now(),
            },
        ]
    )

    print("\nGenerating executive PDF...")

    pdf_bytes = generate_executive_pdf(
        sample_data
    )

    with open(
        "test_executive_report.pdf",
        "wb",
    ) as file:

        file.write(pdf_bytes)

    print(
        "Created: test_executive_report.pdf"
    )

    print("\nGenerating batch PDF...")

    batch_bytes = generate_batch_pdf(
        sample_data
    )

    with open(
        "test_batch_report.pdf",
        "wb",
    ) as file:

        file.write(batch_bytes)

    print(
        "Created: test_batch_report.pdf"
    )

    print("\nGenerating customer PDF...")

    customer = {
        "CONS_NO": "TEST001",
        "std_cons": 3.5,
        "Stability": 0.82,
        "cv": 0.31,
        "longest_missing_streak": 4,
        "high_cons_ratio": 0.61,
        "PAR": 3.5,
        "zero_ratio": 0.12,
        "longest_zero_streak": 2,
    }

    result = {
        "fraud_pobability": 0.82,
        "prediction": 1,
        "risk": "High",
    }

    customer_pdf = generate_customer_pdf(
        customer,
        result,
    )

    with open(
        "test_customer_report.pdf",
        "wb",
    ) as file:

        file.write(customer_pdf)

    print(
        "Created: test_customer_report.pdf"
    )

    print("\nPDF test completed successfully.")
