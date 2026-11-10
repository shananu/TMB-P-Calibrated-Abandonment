"""Generate paper figures from completed ToolMisuseBench release artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "out/release_runs/v0_2_large"
OUT = ROOT / "figures"
PAGE_W, PAGE_H = 800, 390

INK = HexColor("#111111")
MUTED = HexColor("#444444")
GRID = HexColor("#d0d0d0")
PARSER = HexColor("#111111")
QWEN = HexColor("#777777")
NO_RECOVERY = HexColor("#ffffff")


def stage_rows(name: str) -> list[dict]:
    return json.loads((RUN / "stages" / name / "score.json").read_text())["per_task"]


def subset_success(rows: list[dict], predicate: Callable[[dict], bool]) -> float:
    subset = [row for row in rows if predicate(row)]
    return sum(float(row["TaskSuccess"]) for row in subset) / max(1, len(subset))


def axes(canvas: Canvas, left: int, bottom: int, plot_w: int, plot_h: int, x_label: str, y_label: str) -> None:
    canvas.setFont("Times-Roman", 9)
    for tick in range(6):
        y = bottom + tick * plot_h / 5
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.35)
        canvas.line(left, y, left + plot_w, y)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(left - 9, y - 3, f"{tick / 5:.1f}")

    canvas.setStrokeColor(INK)
    canvas.setLineWidth(0.9)
    canvas.line(left, bottom, left, bottom + plot_h)
    canvas.line(left, bottom, left + plot_w, bottom)

    canvas.setFillColor(INK)
    canvas.setFont("Times-Bold", 11)
    canvas.drawCentredString(left + plot_w / 2, bottom - 38, x_label)
    canvas.saveState()
    canvas.translate(42, bottom + plot_h / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, y_label)
    canvas.restoreState()


def legend(canvas: Canvas, items: list[tuple[str, object]], center_x: float, y: int) -> None:
    widths = [74 + len(label) * 3.7 for label, _ in items]
    x = center_x - sum(widths) / 2
    for (label, color), width in zip(items, widths):
        canvas.setFillColor(color)
        canvas.setStrokeColor(INK)
        canvas.setLineWidth(0.6)
        canvas.rect(x, y, 13, 9, stroke=1, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Times-Roman", 10)
        canvas.drawString(x + 18, y - 1, label)
        x += width


def recovery_legend(canvas: Canvas, center_x: float, y: int) -> None:
    items = [("Parser", PARSER, False), ("No recovery", NO_RECOVERY, False), ("Qwen3-4B", QWEN, True)]
    widths = [74 + len(label) * 3.7 for label, _, _ in items]
    x = center_x - sum(widths) / 2
    for (label, color, hatched), width in zip(items, widths):
        canvas.setFillColor(color)
        canvas.setStrokeColor(INK)
        canvas.setLineWidth(0.6)
        canvas.rect(x, y, 13, 9, stroke=1, fill=1)
        if hatched:
            diagonal_hatch(canvas, x, y, 13, 9, spacing=5)
        canvas.setFillColor(INK)
        canvas.setFont("Times-Roman", 10)
        canvas.drawString(x + 18, y - 1, label)
        x += width


def marker_legend(canvas: Canvas, center_x: float, y: int) -> None:
    items = [("Parser", "circle"), ("Qwen3-4B", "square")]
    widths = [78 + len(label) * 3.7 for label, _ in items]
    x = center_x - sum(widths) / 2
    for (label, marker), width in zip(items, widths):
        canvas.setFillColor(PARSER if marker == "circle" else QWEN)
        canvas.setStrokeColor(INK)
        canvas.setLineWidth(0.7)
        if marker == "circle":
            canvas.circle(x + 6.5, y + 4.5, 5.0, stroke=1, fill=1)
        else:
            canvas.rect(x + 1.5, y, 10.0, 9.5, stroke=1, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Times-Roman", 10)
        canvas.drawString(x + 18, y - 1, label)
        x += width


def bar(canvas: Canvas, x: float, bottom: int, width: int, height: float, color) -> None:
    canvas.setStrokeColor(INK)
    canvas.setLineWidth(0.6)
    if height <= 0:
        canvas.setLineWidth(0.8)
        canvas.line(x, bottom + 1.5, x + width, bottom + 1.5)
        return
    canvas.setFillColor(color)
    canvas.rect(x, bottom, width, height, stroke=1, fill=1)


def diagonal_hatch(canvas: Canvas, x: float, bottom: int, width: int, height: float, spacing: int = 7) -> None:
    if height <= 0:
        return
    canvas.setStrokeColor(HexColor("#555555"))
    canvas.setLineWidth(0.35)
    start = int(x - height)
    end = int(x + width)
    pos = start
    while pos < end:
        x1 = max(x, pos)
        y1 = bottom + max(0, x - pos)
        x2 = min(x + width, pos + height)
        y2 = bottom + min(height, x + width - pos)
        canvas.line(x1, y1, x2, y2)
        pos += spacing


def value_label(canvas: Canvas, x: float, y: float, text: str) -> None:
    canvas.setFillColor(INK)
    canvas.setFont("Times-Roman", 9)
    canvas.drawCentredString(x, y, text)


def recovery_effect() -> None:
    parser = stage_rows("deterministic_instruction_parser")
    no_recovery = stage_rows("deterministic_instruction_parser_no_recovery")
    qwen = stage_rows("llm_balanced_panel")
    groups = [
        ("Clean", lambda row: row["PrimaryFault"] == "clean"),
        ("Single fault", lambda row: row["PrimaryFault"] != "clean" and len(row["FaultTypes"]) == 1),
        ("Mixed faults", lambda row: len(row["FaultTypes"]) > 1),
    ]
    series = [
        ("Parser", parser, PARSER),
        ("No recovery", no_recovery, NO_RECOVERY),
        ("Qwen3-4B", qwen, QWEN),
    ]

    canvas = Canvas(str(OUT / "v2_recovery_effect.pdf"), pagesize=(PAGE_W, PAGE_H))
    left, bottom, plot_w, plot_h = 82, 72, 630, 250
    axes(canvas, left, bottom, plot_w, plot_h, "Task slice", "Task success")
    group_w = plot_w / len(groups)
    bar_w = 38
    for i, (label, predicate) in enumerate(groups):
        center = left + group_w * (i + 0.5)
        for j, (_, rows, color) in enumerate(series):
            value = subset_success(rows, predicate)
            x = center + (j - 1) * 48 - bar_w / 2
            height = value * plot_h
            bar(canvas, x, bottom, bar_w, height, color)
            if color == QWEN:
                diagonal_hatch(canvas, x, bottom, bar_w, height)
            value_label(canvas, x + bar_w / 2, bottom + height + 6, f"{value:.2f}")
        canvas.setFillColor(INK)
        canvas.setFont("Times-Roman", 10)
        canvas.drawCentredString(center, bottom - 20, label)
    recovery_legend(canvas, PAGE_W / 2, 350)
    canvas.save()


def fault_profile() -> None:
    parser = stage_rows("deterministic_instruction_parser")
    qwen = stage_rows("llm_balanced_panel")
    faults = ["timeout", "rate_limit", "schema_drift", "authz", "adversarial_error"]
    labels = ["Timeout", "Rate limit", "Schema drift", "Authorization", "Adversarial"]
    values = {
        "Parser": [subset_success(parser, lambda row, fault=fault: row["PrimaryFault"] == fault) for fault in faults],
        "Qwen3-4B": [subset_success(qwen, lambda row, fault=fault: row["PrimaryFault"] == fault) for fault in faults],
    }
    colors = {"Parser": PARSER, "Qwen3-4B": QWEN}

    canvas = Canvas(str(OUT / "v2_fault_profile.pdf"), pagesize=(PAGE_W, PAGE_H))
    left, bottom, plot_w, plot_h = 150, 72, 555, 252
    canvas.setFont("Times-Roman", 9)
    for tick in range(6):
        x = left + tick * plot_w / 5
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.35)
        canvas.line(x, bottom, x, bottom + plot_h)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(x, bottom - 16, f"{tick / 5:.1f}")
    canvas.setStrokeColor(INK)
    canvas.setLineWidth(0.9)
    canvas.line(left, bottom, left + plot_w, bottom)
    canvas.line(left, bottom, left, bottom + plot_h)

    row_gap = plot_h / (len(faults) + 1)
    for i, label in enumerate(labels):
        y = bottom + plot_h - row_gap * (i + 1)
        parser_value = values["Parser"][i]
        qwen_value = values["Qwen3-4B"][i]
        px = left + parser_value * plot_w
        qx = left + qwen_value * plot_w
        canvas.setStrokeColor(HexColor("#9a9a9a"))
        canvas.setLineWidth(1.0)
        canvas.line(px, y, qx, y)
        canvas.setFillColor(INK)
        canvas.setFont("Times-Roman", 10)
        canvas.drawRightString(left - 12, y - 3, label)
        canvas.setFillColor(colors["Parser"])
        canvas.setStrokeColor(INK)
        canvas.circle(px, y, 5.2, stroke=1, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(px, y - 15, f"{parser_value:.2f}")

        canvas.setFillColor(colors["Qwen3-4B"])
        canvas.setStrokeColor(INK)
        canvas.rect(qx - 5.0, y - 5.0, 10.0, 10.0, stroke=1, fill=1)
        canvas.setFillColor(INK)
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(qx, y + 10, f"{qwen_value:.2f}")

    canvas.setFillColor(INK)
    canvas.setFont("Times-Bold", 11)
    canvas.drawCentredString(left + plot_w / 2, bottom - 38, "Task success")
    canvas.saveState()
    canvas.translate(48, bottom + plot_h / 2)
    canvas.rotate(90)
    canvas.drawCentredString(0, 0, "Fault family")
    canvas.restoreState()
    marker_legend(canvas, PAGE_W / 2, 350)
    canvas.save()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    recovery_effect()
    fault_profile()


if __name__ == "__main__":
    main()
