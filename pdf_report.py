"""Renders a report dict (produced by analysis.build_report) into a downloadable PDF."""
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

INK_PRIMARY = colors.HexColor("#0b0b0b")
INK_SECONDARY = colors.HexColor("#52514e")
INK_MUTED = colors.HexColor("#898781")
COL_BLUE = colors.HexColor("#2a78d6")
GRIDLINE = colors.HexColor("#e1e0d9")
ROW_ALT = colors.HexColor("#f2f1ec")

PAGE_SIZE = landscape(letter)
CONTENT_WIDTH = PAGE_SIZE[0] - 1.4 * inch

title_style = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20, textColor=INK_PRIMARY, spaceAfter=4)
subtitle_style = ParagraphStyle("subtitle", fontName="Helvetica", fontSize=12.5, textColor=INK_SECONDARY, spaceAfter=10)
h2_style = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13, textColor=INK_PRIMARY, spaceBefore=6, spaceAfter=6)
body_style = ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK_SECONDARY)
caption_style = ParagraphStyle("caption", fontName="Helvetica-Oblique", fontSize=8.5, textColor=INK_MUTED, spaceBefore=6)


def _table(rows, col_widths=None):
    if not rows:
        return Spacer(1, 0)
    header = list(rows[0].keys())
    data = [header] + [[str(r[k]) for k in header] for r in rows]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), COL_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, GRIDLINE),
        ("TEXTCOLOR", (0, 1), (-1, -1), INK_PRIMARY),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    t.setStyle(TableStyle(style))
    return t


def _image(png_bytes, max_width=CONTENT_WIDTH, max_height=4.6 * inch):
    img = Image(io.BytesIO(png_bytes))
    ratio = img.imageWidth / img.imageHeight
    w, h = max_width, max_width / ratio
    if h > max_height:
        h = max_height
        w = h * ratio
    img.drawWidth = w
    img.drawHeight = h
    return img


def build_pdf(report: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=PAGE_SIZE,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        title=f"{report['company_name']} vs Nifty 50 - Econometric Report",
    )
    company = report["company_name"]
    t = report["text"]
    story = []

    # ---- Title page ----
    story.append(Paragraph(f"Estimating {company}'s Market Beta Against the Nifty 50", title_style))
    story.append(Paragraph("A five-year daily-returns econometric case study", subtitle_style))
    story.append(Paragraph(t["intro"], body_style))
    story.append(Spacer(1, 14))
    story.append(_table(report["tables"]["meta"], col_widths=[2.4 * inch, 5.5 * inch]))
    story.append(PageBreak())

    # ---- Prices ----
    story.append(Paragraph("Five Years of Price History", h2_style))
    story.append(_image(report["images"]["prices"]))
    story.append(Paragraph(t["prices"], body_style))
    story.append(PageBreak())

    # ---- Indexed ----
    story.append(Paragraph("Cumulative Performance, Rebased to 100", h2_style))
    story.append(_image(report["images"]["indexed"]))
    story.append(Paragraph(t["indexed"], body_style))
    story.append(PageBreak())

    # ---- Summary stats ----
    story.append(Paragraph("Return Summary Statistics", h2_style))
    story.append(_table(report["tables"]["stats"]))
    story.append(Spacer(1, 10))
    story.append(_image(report["images"]["hist"], max_height=3.4 * inch))
    story.append(Paragraph(t["stats"], body_style))
    story.append(PageBreak())

    # ---- Scatter / market model ----
    story.append(Paragraph("The Market Model: Company Return vs. Nifty 50 Return", h2_style))
    story.append(_image(report["images"]["scatter"]))
    story.append(Paragraph(t["scatter"], body_style))
    story.append(PageBreak())

    # ---- Regression tables ----
    story.append(Paragraph("OLS Regression Results and Hypothesis Tests", h2_style))
    story.append(Paragraph(t["regression"], body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Model 1 - Simple market model", ParagraphStyle("l", parent=body_style, fontName="Helvetica-Bold", textColor=INK_PRIMARY)))
    story.append(_table(report["tables"]["reg_simple"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Model 2 - Extended model with lagged Nifty return", ParagraphStyle("l2", parent=body_style, fontName="Helvetica-Bold", textColor=INK_PRIMARY)))
    story.append(_table(report["tables"]["reg_multi"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(t["hyp1"], body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(t["hyp2"], body_style))
    story.append(PageBreak())

    # ---- Diagnostics ----
    story.append(Paragraph("Regression Diagnostics", h2_style))
    story.append(_image(report["images"]["diagnostics"], max_height=5.2 * inch))
    story.append(Paragraph(t["diagnostics"], body_style))
    story.append(PageBreak())

    # ---- Robust SE ----
    story.append(Paragraph("Heteroskedasticity-Robust Inference", h2_style))
    story.append(_table(report["tables"]["robust"]))
    story.append(Spacer(1, 8))
    story.append(_table(report["tables"]["compare"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(t["robust"], body_style))
    story.append(PageBreak())

    # ---- Prediction ----
    story.append(Paragraph("Out-of-Sample Prediction", h2_style))
    story.append(_table(report["tables"]["prediction"]))
    story.append(Spacer(1, 10))
    story.append(_image(report["images"]["prediction"], max_height=3.6 * inch))
    story.append(Paragraph(t["prediction"], body_style))
    story.append(PageBreak())

    # ---- Conclusion ----
    story.append(Paragraph("Conclusion", h2_style))
    story.append(Paragraph(t["conclusion"], body_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        f"Generated {report['generated']:%d %b %Y %H:%M}  |  Source: Yahoo Finance ({report['ticker']}, ^NSEI)",
        caption_style,
    ))

    doc.build(story)
    return buf.getvalue()
