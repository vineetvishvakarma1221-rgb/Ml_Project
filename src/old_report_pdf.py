# ============================================================
# PDF REPORT ENGINE
# Electricity Fraud Detection System
# Step 5.3 - Professional PDF Reporting
# ============================================================

from __future__ import annotations

import os
from io import BytesIO
from datetime import datetime
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
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


# ============================================================
# COLORS
# ============================================================

PRIMARY = colors.HexColor("#0F172A")
SECONDARY = colors.HexColor("#334155")

ACCENT = colors.HexColor("#2563EB")

SUCCESS = colors.HexColor("#16A34A")
WARNING = colors.HexColor("#F59E0B")
DANGER = colors.HexColor("#DC2626")

LIGHT_BG = colors.HexColor("#F8FAFC")
BORDER = colors.HexColor("#CBD5E1")

WHITE = colors.white


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
    textColor=PRIMARY,
    spaceAfter=8,
)


SUBTITLE_STYLE = ParagraphStyle(
    "ReportSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=11,
    leading=16,
    alignment=TA_CENTER,
    textColor=SECONDARY,
    spaceAfter=8,
)


SECTION_STYLE = ParagraphStyle(
    "SectionHeading",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=19,
    textColor=PRIMARY,
    spaceBefore=8,
    spaceAfter=9,
)


SUBSECTION_STYLE = ParagraphStyle(
    "SubSectionHeading",
    parent=styles["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=15,
    textColor=SECONDARY,
    spaceBefore=6,
    spaceAfter=6,
)


BODY_STYLE = ParagraphStyle(
    "ReportBody",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.5,
    leading=14,
    textColor=SECONDARY,
    spaceAfter=6,
)


SMALL_STYLE = ParagraphStyle(
    "Small",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=7.5,
    leading=10,
    textColor=SECONDARY,
)


CENTER_STYLE = ParagraphStyle(
    "Center",
    parent=BODY_STYLE,
    alignment=TA_CENTER,
)


# ============================================================
# BASIC HELPERS
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
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return Paragraph(text, style)


# ============================================================
# PAGE HEADER / FOOTER
# ============================================================

def _draw_page_header_footer(canvas, doc):
    """
    Draw professional header/footer on every page.
    """

    canvas.saveState()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)

    canvas.line(
        MARGIN_LEFT,
        PAGE_HEIGHT - 12 * mm,
        PAGE_WIDTH - MARGIN_RIGHT,
        PAGE_HEIGHT - 12 * mm,
    )

    canvas.setFont(
        "Helvetica-Bold",
        7.5,
    )

    canvas.setFillColor(PRIMARY)

    canvas.drawString(
        MARGIN_LEFT,
        PAGE_HEIGHT - 9 * mm,
        "ELECTRICITY FRAUD DETECTION SYSTEM",
    )

    canvas.setFont(
        "Helvetica",
        7,
    )

    canvas.setFillColor(SECONDARY)

    canvas.drawRightString(
        PAGE_WIDTH - MARGIN_RIGHT,
        PAGE_HEIGHT - 9 * mm,
        "Machine Learning Risk Assessment",
    )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    canvas.setStrokeColor(BORDER)

    canvas.line(
        MARGIN_LEFT,
        12 * mm,
        PAGE_WIDTH - MARGIN_RIGHT,
        12 * mm,
    )

    canvas.setFont(
        "Helvetica",
        7,
    )

    canvas.setFillColor(SECONDARY)

    canvas.drawString(
        MARGIN_LEFT,
        8 * mm,
        "Generated by Electricity Fraud Detection System",
    )

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

    story.append(
        Spacer(
            1,
            25 * mm,
        )
    )

    # --------------------------------------------------------
    # Logo
    # --------------------------------------------------------

    if LOGO_PATH.exists():

        try:

            from reportlab.platypus import Image

            logo = Image(
                str(LOGO_PATH)
            )

            logo.drawHeight = 30 * mm
            logo.drawWidth = 30 * mm

            logo.hAlign = "CENTER"

            story.append(logo)

            story.append(
                Spacer(
                    1,
                    8 * mm,
                )
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "⚡ ELECTRICITY FRAUD<br/>DETECTION SYSTEM",
            TITLE_STYLE,
        )
    )

    story.append(
        Paragraph(
            "Machine Learning Risk Assessment Report",
            SUBTITLE_STYLE,
        )
    )

    story.append(
        Spacer(
            1,
            15 * mm,
        )
    )

    # --------------------------------------------------------
    # Information card
    # --------------------------------------------------------

    generated_at = datetime.now().strftime(
        "%d %B %Y, %H:%M"
    )

    cover_data = [
        [
            _paragraph(
                "<b>Model</b>",
                CENTER_STYLE,
            ),
            _paragraph(
                "Random Forest Classifier",
                CENTER_STYLE,
            ),
        ],
        [
            _paragraph(
                "<b>Explainability</b>",
                CENTER_STYLE,
            ),
            _paragraph(
                "SHAP",
                CENTER_STYLE,
            ),
        ],
        [
            _paragraph(
                "<b>Report Generated</b>",
                CENTER_STYLE,
            ),
            _paragraph(
                generated_at,
                CENTER_STYLE,
            ),
        ],
        [
            _paragraph(
                "<b>Developer</b>",
                CENTER_STYLE,
            ),
            _paragraph(
                "Vineet Vishvakarma",
                CENTER_STYLE,
            ),
        ],
    ]

    table = Table(
        cover_data,
        colWidths=[
            55 * mm,
            85 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_BG,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    BORDER,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
            ]
        )
    )

    table.hAlign = "CENTER"

    story.append(table)

    story.append(
        Spacer(
            1,
            25 * mm,
        )
    )

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
            _paragraph(
                "<b>Total Predictions</b><br/>"
                f"<font size='16'>{summary['total_predictions']:,}</font>",
                CENTER_STYLE,
            ),
            _paragraph(
                "<b>Fraud Cases</b><br/>"
                f"<font size='16'>{summary['fraud_cases']:,}</font>",
                CENTER_STYLE,
            ),
        ],
        [
            _paragraph(
                "<b>Normal Cases</b><br/>"
                f"<font size='16'>{summary['normal_cases']:,}</font>",
                CENTER_STYLE,
            ),
            _paragraph(
                "<b>Fraud Rate</b><br/>"
                f"<font size='16'>{summary['fraud_rate']:.2f}%</font>",
                CENTER_STYLE,
            ),
        ],
    ]

    table = Table(
        data,
        colWidths=[
            82 * mm,
            82 * mm,
        ],
        rowHeights=[
            25 * mm,
            25 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_BG,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    BORDER,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER",
                ),
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
            _paragraph("<b>Risk Level</b>", CENTER_STYLE),
            _paragraph("<b>Cases</b>", CENTER_STYLE),
            _paragraph("<b>Share</b>", CENTER_STYLE),
        ],
        [
            _paragraph(
                "<font color='#16A34A'><b>Low</b></font>",
                CENTER_STYLE,
            ),
            _paragraph(
                str(summary["low_risk"]),
                CENTER_STYLE,
            ),
            _paragraph(
                _format_percent(
                    risk_percent(
                        summary["low_risk"]
                    )
                ),
                CENTER_STYLE,
            ),
        ],
        [
            _paragraph(
                "<font color='#F59E0B'><b>Medium</b></font>",
                CENTER_STYLE,
            ),
            _paragraph(
                str(summary["medium_risk"]),
                CENTER_STYLE,
            ),
            _paragraph(
                _format_percent(
                    risk_percent(
                        summary["medium_risk"]
                    )
                ),
                CENTER_STYLE,
            ),
        ],
        [
            _paragraph(
                "<font color='#DC2626'><b>High</b></font>",
                CENTER_STYLE,
            ),
            _paragraph(
                str(summary["high_risk"]),
                CENTER_STYLE,
            ),
            _paragraph(
                _format_percent(
                    risk_percent(
                        summary["high_risk"]
                    )
                ),
                CENTER_STYLE,
            ),
        ],
    ]

    table = Table(
        data,
        colWidths=[
            60 * mm,
            50 * mm,
            50 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    PRIMARY,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    LIGHT_BG,
                ),
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
        return Paragraph(
            "No prediction records available.",
            BODY_STYLE,
        )

    working = df.head(max_rows).copy()

    rows = [
        [
            "Prediction",
            "Probability",
            "Risk",
            "Timestamp",
        ]
    ]

    for _, row in working.iterrows():

        prediction = row.get(
            "prediction",
            "N/A"
        )

        if prediction == 1:
            prediction = "Fraud"
        elif prediction == 0:
            prediction = "Normal"

        probability = row.get(
            "fraud_pobability",
            None
        )

        try:
            probability = (
                float(probability) * 100
            )

            probability_text = (
                f"{probability:.2f}%"
            )

        except (
            TypeError,
            ValueError,
        ):
            probability_text = "N/A"

        risk = row.get(
            "risk",
            "N/A"
        )

        timestamp = row.get(
            "prediction_time",
            "N/A"
        )

        if pd.notna(timestamp):
            timestamp = str(timestamp)

        rows.append(
            [
                str(prediction),
                probability_text,
                str(risk),
                str(timestamp),
            ]
        )

    table = Table(
        rows,
        colWidths=[
            35 * mm,
            35 * mm,
            30 * mm,
            65 * mm,
        ],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    PRIMARY,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7.5,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        LIGHT_BG,
                    ],
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

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

    Returns:
        bytes
    """

    df = prepare_report_dataframe(data)

    summary = calculate_report_summ(
        df
    )

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
        Paragraph(
            "1. Executive Summary",
            SECTION_STYLE,
        )
    )

    story.append(
        Paragraph(
            "This section provides a high-level overview "
            "of the predictions generated by the Electricity "
            "Fraud Detection System.",
            BODY_STYLE,
        )
    )

    story.append(
        _build_kpi_table(
            summary
        )
    )

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )

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
        [
            "Metric",
            "Value",
        ],
        [
            "Average Fraud Probability",
            _format_percent(
                summary[
                    "average_fraud_probability"
                ]
            ),
        ],
        [
            "Low Risk Cases",
            f"{summary['low_risk']:,}",
        ],
        [
            "Medium Risk Cases",
            f"{summary['medium_risk']:,}",
        ],
        [
            "High Risk Cases",
            f"{summary['high_risk']:,}",
        ],
    ]

    indicator_table = Table(
        indicators,
        colWidths=[
            95 * mm,
            65 * mm,
        ],
        repeatRows=1,
    )

    indicator_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    PRIMARY,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        LIGHT_BG,
                    ],
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        indicator_table
    )

    story.append(PageBreak())

    # --------------------------------------------------------
    # Risk Analysis
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. Risk Distribution",
            SECTION_STYLE,
        )
    )

    story.append(
        Paragraph(
            "The following distribution summarizes the "
            "number of predictions classified into each "
            "risk category.",
            BODY_STYLE,
        )
    )

    story.append(
        _build_risk_table(
            summary
        )
    )

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. Model Information",
            SECTION_STYLE,
        )
    )

    model_info = [
        [
            "Component",
            "Details",
        ],
        [
            "Machine Learning Model",
            "Random Forest Classifier",
        ],
        [
            "Explainability",
            "SHAP",
        ],
        [
            "Prediction Type",
            "Binary Classification",
        ],
        [
            "Risk Levels",
            "Low / Medium / High",
        ],
        [
            "Application",
            "Streamlit",
        ],
        [
            "Database",
            "MySQL",
        ],
    ]

    model_table = Table(
        model_info,
        colWidths=[
            65 * mm,
            95 * mm,
        ],
        repeatRows=1,
    )

    model_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    PRIMARY,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        LIGHT_BG,
                    ],
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        model_table
    )

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )

    # --------------------------------------------------------
    # Feature information
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. Model Features",
            SECTION_STYLE,
        )
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

    feature_rows = [
        [
            "No.",
            "Feature",
        ]
    ]

    for index, feature in enumerate(
        features,
        start=1,
    ):
        feature_rows.append(
            [
                str(index),
                feature,
            ]
        )

    feature_table = Table(
        feature_rows,
        colWidths=[
            20 * mm,
            140 * mm,
        ],
        repeatRows=1,
    )

    feature_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    PRIMARY,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        LIGHT_BG,
                    ],
                ),
                (
                    "ALIGN",
                    (0, 0),
                    (0, -1),
                    "CENTER",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        feature_table
    )

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

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )

    # --------------------------------------------------------
    # Conclusion
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "6. Conclusion",
            SECTION_STYLE,
        )
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

    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

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

def generate_batch_pdf(
    result_df,
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

        customer_df = pd.DataFrame(
            [customer_data]
        )

    elif isinstance(
        customer_data,
        pd.DataFrame,
    ):

        customer_df = customer_data.copy()

    else:

        customer_df = pd.DataFrame(
            [customer_data]
        )

    # --------------------------------------------------------
    # Create combined record
    # --------------------------------------------------------

    result = (
        prediction_result
        if isinstance(
            prediction_result,
            dict,
        )
        else {}
    )

    row = {}

    if not customer_df.empty:

        row.update(
            customer_df.iloc[0].to_dict()
        )

    row.update(result)

    combined_df = pd.DataFrame(
        [row]
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    combined_df = prepare_report_dataframe(
        combined_df
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = calculate_report_summ(
        combined_df
    )

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

    story.append(
        Spacer(
            1,
            12 * mm,
        )
    )

    story.append(
        Paragraph(
            "⚡ CUSTOMER FRAUD RISK REPORT",
            TITLE_STYLE,
        )
    )

    story.append(
        Paragraph(
            "Individual Machine Learning Assessment",
            SUBTITLE_STYLE,
        )
    )

    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

    # --------------------------------------------------------
    # Customer ID
    # --------------------------------------------------------

    consumer_id = "N/A"

    for column in [
        "CONS_NO",
        "consumer_id",
        "Consumer ID",
    ]:

        if column in combined_df.columns:

            consumer_id = combined_df[
                column
            ].iloc[0]

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

    prediction = result.get(
        "prediction"
    )

    if prediction == 1:
        prediction_label = "FRAUD SUSPECTED"
    elif prediction == 0:
        prediction_label = "NORMAL"
    else:
        prediction_label = str(
            prediction
        )

    probability = result.get(
        "fraud_pobability",
        result.get(
            "fraud_probability"
        ),
    )

    try:
        probability_percent = (
            float(probability) * 100
        )

    except (
        TypeError,
        ValueError,
    ):
        probability_percent = 0

    risk = result.get(
        "risk",
        "Unknown",
    )

    result_table = Table(
        [
            [
                _paragraph(
                    "<b>Prediction</b>",
                    CENTER_STYLE,
                ),
                _paragraph(
                    prediction_label,
                    CENTER_STYLE,
                ),
            ],
            [
                _paragraph(
                    "<b>Fraud Probability</b>",
                    CENTER_STYLE,
                ),
                _paragraph(
                    f"{probability_percent:.2f}%",
                    CENTER_STYLE,
                ),
            ],
            [
                _paragraph(
                    "<b>Risk Level</b>",
                    CENTER_STYLE,
                ),
                _paragraph(
                    str(risk).upper(),
                    CENTER_STYLE,
                ),
            ],
            [
                _paragraph(
                    "<b>Generated At</b>",
                    CENTER_STYLE,
                ),
                _paragraph(
                    datetime.now().strftime(
                        "%d %b %Y, %H:%M:%S"
                    ),
                    CENTER_STYLE,
                ),
            ],
        ],
        colWidths=[
            70 * mm,
            90 * mm,
        ],
    )

    result_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    LIGHT_BG,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    BORDER,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    BORDER,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
            ]
        )
    )

    story.append(
        result_table
    )

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )

    # --------------------------------------------------------
    # Feature analysis
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Feature Assessment",
            SECTION_STYLE,
        )
    )

    feature_rows = [
        [
            "Feature",
            "Value",
        ]
    ]

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

            value = combined_df[
                feature
            ].iloc[0]

            feature_rows.append(
                [
                    feature,
                    _format_number(
                        value
                    ),
                ]
            )

    feature_table = Table(
        feature_rows,
        colWidths=[
            95 * mm,
            65 * mm,
        ],
        repeatRows=1,
    )

    feature_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    PRIMARY,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER,
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        LIGHT_BG,
                    ],
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
            ]
        )
    )

    story.append(
        feature_table
    )

    story.append(
        Spacer(
            1,
            10 * mm,
        )
    )

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

    story.append(
        Paragraph(
            interpretation,
            BODY_STYLE,
        )
    )

    story.append(
        Spacer(
            1,
            12 * mm,
        )
    )

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