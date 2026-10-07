"""
reports/report_generator.py
---------------------------
Generates CSV and PDF reports for completed Food Freshness Detection analyses.
Uses actual prediction results, timestamps, and Grad-CAM visual explanations.
"""

import csv
from datetime import datetime
import os
from pathlib import Path
import uuid

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def sanitize_report_data(data: dict) -> dict:
    """Ensure all required fields exist and are formatted safely."""
    now_str = data.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf_val = data.get("confidence", 0.0)
    conf_str = f"{conf_val * 100:.2f}%" if isinstance(conf_val, (int, float)) else str(conf_val)

    return {
        "timestamp": now_str,
        "filename": data.get("filename", "N/A"),
        "predicted_class": data.get("predicted_class", "N/A"),
        "produce": data.get("produce", "N/A"),
        "freshness": data.get("freshness", "N/A"),
        "confidence": conf_str,
        "confidence_raw": conf_val,
        "quality_status": data.get("quality_status", "N/A"),
        "visual_summary": data.get("visual_summary", "N/A"),
        "gradcam_image": data.get("gradcam_image"),
    }


def generate_csv_report(analysis_data: dict, output_dir: Path) -> str:
    """
    Generate a structured CSV report for the food freshness analysis.

    Returns:
        Absolute path to the generated CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = sanitize_report_data(analysis_data)
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"food_freshness_{timestamp_slug}_{unique_id}.csv"
    file_path = output_dir / filename

    fieldnames = [
        "Date/Time",
        "Image Filename",
        "Predicted Class",
        "Produce",
        "Freshness",
        "Confidence",
        "Quality Status",
        "Visual Summary",
        "Grad-CAM Image",
    ]

    row_data = {
        "Date/Time": data["timestamp"],
        "Image Filename": data["filename"],
        "Predicted Class": data["predicted_class"],
        "Produce": data["produce"],
        "Freshness": data["freshness"],
        "Confidence": data["confidence"],
        "Quality Status": data["quality_status"],
        "Visual Summary": data["visual_summary"],
        "Grad-CAM Image": data["gradcam_image"] or "N/A",
    }

    with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row_data)

    return str(file_path)


def generate_pdf_report(analysis_data: dict, output_dir: Path, project_root: Path = None) -> str:
    """
    Generate a professional PDF report for the food freshness analysis using ReportLab.

    Returns:
        Absolute path to the generated PDF file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = sanitize_report_data(analysis_data)
    timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"food_freshness_{timestamp_slug}_{unique_id}.pdf"
    file_path = output_dir / filename

    doc = SimpleDocTemplate(
        str(file_path),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        fontName="Helvetica-Bold",
        spaceBefore=10,
        spaceAfter=4,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
    )

    disclaimer_style = ParagraphStyle(
        "ReportDisclaimer",
        parent=styles["Italic"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_CENTER,
    )

    story = []

    # Title & Header
    story.append(Paragraph("FOOD FRESHNESS DETECTION REPORT", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Automated Visual Quality & Freshness Assessment", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#22c55e"), spaceAfter=15))

    # 1. Analysis Information
    story.append(Paragraph("1. Analysis Information", section_heading))
    info_table_data = [
        [Paragraph("<b>Date and Time:</b>", body_style), Paragraph(data["timestamp"], body_style)],
        [Paragraph("<b>Image Filename:</b>", body_style), Paragraph(data["filename"], body_style)],
    ]
    info_table = Table(info_table_data, colWidths=[150, 380])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 12))

    # 2. Prediction
    story.append(Paragraph("2. Prediction", section_heading))
    freshness_color = colors.HexColor("#16a34a") if data["freshness"].lower() == "fresh" else colors.HexColor("#dc2626")
    prediction_table_data = [
        [Paragraph("<b>Produce Type:</b>", body_style), Paragraph(data["produce"], body_style)],
        [
            Paragraph("<b>Freshness Status:</b>", body_style),
            Paragraph(f"<font color='{freshness_color.hexval()}'><b>{data['freshness']}</b></font>", body_style),
        ],
        [Paragraph("<b>Model Confidence:</b>", body_style), Paragraph(f"<b>{data['confidence']}</b>", body_style)],
    ]
    pred_table = Table(prediction_table_data, colWidths=[150, 380])
    pred_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(pred_table)
    story.append(Spacer(1, 12))

    # 3. Quality Inspection
    story.append(Paragraph("3. Quality Inspection", section_heading))
    inspection_table_data = [
        [Paragraph("<b>Quality Status:</b>", body_style), Paragraph(f"<b>{data['quality_status']}</b>", body_style)],
        [Paragraph("<b>Visual Summary:</b>", body_style), Paragraph(data["visual_summary"], body_style)],
    ]
    insp_table = Table(inspection_table_data, colWidths=[150, 380])
    insp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(insp_table)
    story.append(Spacer(1, 12))

    # 4. Grad-CAM Explanation
    story.append(Paragraph("4. Grad-CAM Explanation", section_heading))
    gradcam_img_path = data.get("gradcam_image")
    full_cam_path = None
    if gradcam_img_path:
        clean_path = gradcam_img_path.lstrip("/").replace("/", os.sep)
        if project_root:
            candidate = project_root / clean_path
            if candidate.is_file():
                full_cam_path = candidate
        if not full_cam_path and Path(clean_path).is_file():
            full_cam_path = Path(clean_path)

    if full_cam_path and full_cam_path.is_file():
        try:
            rl_img = RLImage(str(full_cam_path), width=200, height=200)
            story.append(rl_img)
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                "Highlighted regions indicate areas that contributed most strongly to the model's prediction.",
                body_style,
            ))
        except Exception:
            story.append(Paragraph("Grad-CAM explanation image was unavailable for this analysis.", body_style))
    else:
        story.append(Paragraph("Grad-CAM explanation image was unavailable for this analysis.", body_style))

    story.append(Spacer(1, 15))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    # 5. Disclaimer
    story.append(Paragraph("5. Disclaimer", section_heading))
    story.append(Paragraph(
        "Visual AI analysis is an image-based prediction and does not replace professional food-safety inspection or laboratory testing.",
        disclaimer_style,
    ))

    # Build document
    doc.build(story)
    return str(file_path)
