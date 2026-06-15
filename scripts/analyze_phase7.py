from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # PNG export is optional; SVG/report generation still works.
    Image = None
    ImageDraw = None
    ImageFont = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


MODEL_ORDER = ["baseline", "selective", "blind"]
MODEL_LABELS = {
    "baseline": "M1 baseline",
    "selective": "Selective",
    "blind": "Blind",
}
PREDICTION_FILES = {
    "baseline": ROOT / "data/test_qwen_predictions.jsonl",
    "selective": ROOT / "data/test_selective_predictions.jsonl",
    "blind": ROOT / "data/test_blind_predictions.jsonl",
}
SUMMARY_FILES = {
    "baseline": ROOT / "logs/qwen_test_baseline_summary.json",
    "selective": ROOT / "logs/selective_test_summary.json",
    "blind": ROOT / "logs/blind_test_summary.json",
}
LOOP_LOG_FILES = {
    "selective": ROOT / "logs/loop_log.csv",
    "blind": ROOT / "logs/blind_loop_log.csv",
}
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_id[str(row["id"])] = row
    return rows_by_id


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def float_value(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def pct(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(number(value) for value in row) + " |")
    return "\n".join(lines)


def load_evaluation_rows() -> dict[str, dict[str, Any]]:
    rows = read_csv_rows(ROOT / "results/evaluation.csv")
    by_strategy: dict[str, dict[str, Any]] = {}
    for row in rows:
        strategy = row["strategy"]
        normalized: dict[str, Any] = dict(row)
        for key in (
            "total_rows",
            "evaluated_rows",
            "scored_rows",
            "correct_rows",
            "failed_rows",
            "error_rows",
            "m1_failed_test_total",
            "m1_failed_test_correct",
            "similar_correct_rows",
        ):
            if normalized.get(key) != "":
                normalized[key] = int(float(normalized[key]))
        for key in (
            "accuracy",
            "m1_failed_test_accuracy",
            "test_generalization_score",
            "accuracy_gain_vs_m1",
            "similar_accuracy",
        ):
            normalized[key] = float_value(normalized.get(key))
        by_strategy[strategy] = normalized
    return by_strategy


def prediction_sets() -> tuple[set[str], dict[str, set[str]], dict[str, dict[str, dict[str, Any]]]]:
    predictions = {
        name: read_jsonl_by_id(path) for name, path in PREDICTION_FILES.items()
    }
    all_ids: set[str] = set()
    correct_ids: dict[str, set[str]] = {}
    for name, rows in predictions.items():
        all_ids.update(rows)
        correct_ids[name] = {
            row_id for row_id, row in rows.items() if row.get("is_correct") is True
        }
    return all_ids, correct_ids, predictions


def pairwise_overlap(
    all_ids: set[str],
    correct_ids: dict[str, set[str]],
    left: str,
    right: str,
) -> dict[str, int]:
    left_set = correct_ids[left]
    right_set = correct_ids[right]
    return {
        "both_correct": len(left_set & right_set),
        f"only_{left}": len(left_set - right_set),
        f"only_{right}": len(right_set - left_set),
        "both_wrong": len(all_ids - (left_set | right_set)),
    }


def test_transition_metrics(
    all_ids: set[str],
    correct_ids: dict[str, set[str]],
    model: str,
) -> dict[str, int]:
    baseline_correct = correct_ids["baseline"]
    model_correct = correct_ids[model]
    baseline_wrong = all_ids - baseline_correct
    return {
        "fixed_m1_mistakes": len(model_correct & baseline_wrong),
        "broke_m1_correct": len(baseline_correct - model_correct),
        "kept_m1_correct": len(model_correct & baseline_correct),
        "still_wrong": len(baseline_wrong - model_correct),
        "net_correct_delta": len(model_correct) - len(baseline_correct),
    }


def loop_metrics() -> dict[str, Any]:
    selective_rows = read_csv_rows(LOOP_LOG_FILES["selective"])
    blind_rows = read_csv_rows(LOOP_LOG_FILES["blind"])
    loop_rows = {"selective": selective_rows, "blind": blind_rows}
    metrics: dict[str, Any] = {}
    for name, rows in loop_rows.items():
        correct_rows = [row for row in rows if bool_value(row.get("is_correct"))]
        wrong_rows = [row for row in rows if not bool_value(row.get("is_correct"))]
        accepted_rows = [row for row in rows if bool_value(row.get("accepted"))]
        rejected_rows = [row for row in rows if not bool_value(row.get("accepted"))]
        losses_correct = [
            value
            for value in (float_value(row.get("train_loss")) for row in correct_rows)
            if value is not None
        ]
        losses_wrong = [
            value
            for value in (float_value(row.get("train_loss")) for row in wrong_rows)
            if value is not None
        ]
        metrics[name] = {
            "steps": len(rows),
            "similar_correct": len(correct_rows),
            "similar_wrong": len(wrong_rows),
            "similar_accuracy": len(correct_rows) / len(rows) if rows else None,
            "accepted": len(accepted_rows),
            "rejected": len(rejected_rows),
            "avg_train_loss_correct": mean(losses_correct) if losses_correct else None,
            "avg_train_loss_wrong": mean(losses_wrong) if losses_wrong else None,
        }

    selective_by_id = {row["original_id"]: row for row in selective_rows}
    blind_by_id = {row["original_id"]: row for row in blind_rows}
    common_ids = sorted(set(selective_by_id) & set(blind_by_id))
    both = selective_only = blind_only = neither = 0
    for row_id in common_ids:
        selective_correct = bool_value(selective_by_id[row_id].get("is_correct"))
        blind_correct = bool_value(blind_by_id[row_id].get("is_correct"))
        if selective_correct and blind_correct:
            both += 1
        elif selective_correct:
            selective_only += 1
        elif blind_correct:
            blind_only += 1
        else:
            neither += 1
    metrics["similar_pairwise"] = {
        "both_correct": both,
        "only_selective": selective_only,
        "only_blind": blind_only,
        "both_wrong": neither,
    }
    metrics["curves"] = {
        name: cumulative_accuracy_curve(rows) for name, rows in loop_rows.items()
    }
    return metrics


def cumulative_accuracy_curve(rows: list[dict[str, str]]) -> list[tuple[int, float]]:
    ordered_rows = sorted(rows, key=lambda row: int(row["step_index"]))
    correct = 0
    points: list[tuple[int, float]] = []
    for index, row in enumerate(ordered_rows, start=1):
        correct += int(bool_value(row.get("is_correct")))
        points.append((index, correct / index))
    return points


def sampled_curve(
    points: list[tuple[int, float]],
    every: int = 10,
) -> list[tuple[int, float]]:
    sampled = [(step, value) for step, value in points if step % every == 0]
    if points and (not sampled or sampled[-1][0] != points[-1][0]):
        sampled.append(points[-1])
    return sampled


def svg_text(x: float, y: float, text: Any, size: int = 14, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}">'
        f"{html.escape(str(text))}</text>"
    )


def write_test_accuracy_chart(evaluation: dict[str, dict[str, Any]]) -> Path:
    output = FIGURES_DIR / "test_accuracy.svg"
    width, height = 860, 500
    left, right, top, bottom = 90, 40, 70, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    labels = ["M1", "Selective", "Blind"]
    values = [
        evaluation["baseline"]["accuracy"],
        evaluation["selective"]["accuracy"],
        evaluation["blind"]["accuracy"],
    ]
    colors = ["#4b5563", "#2563eb", "#16a34a"]
    max_value = 0.70
    bar_width = 120
    gap = (plot_width - len(labels) * bar_width) / (len(labels) + 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, "Final Test Doğruluğu", 24),
    ]
    for tick in range(0, 8):
        value = tick / 10
        y = top + plot_height - (value / max_value) * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(svg_text(left - 12, y + 5, f"{value:.1f}", 12, "end"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')

    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + gap + index * (bar_width + gap)
        bar_height = (value / max_value) * plot_height
        y = top + plot_height - bar_height
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="4" fill="{color}"/>')
        parts.append(svg_text(x + bar_width / 2, y - 10, f"{value:.3f}", 14))
        parts.append(svg_text(x + bar_width / 2, height - 38, label, 15))
    parts.append(svg_text(24, top + plot_height / 2, "Doğruluk", 13, "middle"))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output


def write_learning_curve_chart(curves: dict[str, list[tuple[int, float]]]) -> Path:
    output = FIGURES_DIR / "similar_learning_curve.svg"
    width, height = 920, 520
    left, right, top, bottom = 90, 180, 70, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = {"selective": "#2563eb", "blind": "#16a34a"}
    labels = {"selective": "Selective", "blind": "Blind"}
    max_step = max(point[0] for points in curves.values() for point in points)
    min_y, max_y = 0.65, 1.00

    def point_to_xy(step: int, value: float) -> tuple[float, float]:
        x = left + ((step - 1) / (max_step - 1)) * plot_width
        y = top + plot_height - ((value - min_y) / (max_y - min_y)) * plot_height
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, "Benzer Soru Kümülatif Doğruluk Eğrisi", 24),
    ]
    for tick in range(0, 8):
        value = min_y + tick * (max_y - min_y) / 7
        y = top + plot_height - ((value - min_y) / (max_y - min_y)) * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(svg_text(left - 12, y + 5, f"{value:.2f}", 12, "end"))
    for tick in range(0, 6):
        step = 1 + tick * (max_step - 1) / 5
        x = left + ((step - 1) / (max_step - 1)) * plot_width
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#f3f4f6"/>')
        parts.append(svg_text(x, height - 38, f"{int(round(step))}", 12))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')

    for name, points in curves.items():
        points = sampled_curve(points, every=10)
        polyline = " ".join(
            f"{x:.1f},{y:.1f}" for x, y in (point_to_xy(step, value) for step, value in points)
        )
        parts.append(f'<polyline points="{polyline}" fill="none" stroke="{colors[name]}" stroke-width="3"/>')
        legend_y = 110 + 30 * (0 if name == "selective" else 1)
        parts.append(f'<line x1="{width-right+24}" y1="{legend_y}" x2="{width-right+62}" y2="{legend_y}" stroke="{colors[name]}" stroke-width="4"/>')
        parts.append(svg_text(width - right + 70, legend_y + 5, labels[name], 14, "start"))
    parts.append(svg_text(width / 2, height - 10, "Fine-tuning adımı", 13))
    parts.append(svg_text(24, top + plot_height / 2, "Kümülatif doğruluk", 13))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output


def write_transition_chart(transitions: dict[str, dict[str, int]]) -> Path:
    output = FIGURES_DIR / "test_transitions.svg"
    width, height = 860, 500
    left, right, top, bottom = 90, 40, 70, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    groups = ["selective", "blind"]
    series = [
        ("M1 hatası düzeldi", "fixed_m1_mistakes", "#16a34a"),
        ("M1 doğrusu bozuldu", "broke_m1_correct", "#dc2626"),
    ]
    max_value = 120
    group_width = 240
    bar_width = 86
    gap = (plot_width - len(groups) * group_width) / (len(groups) + 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 34, "Test Setinde Düzeltme / Bozma Dengesi", 24),
    ]
    for tick in range(0, 6):
        value = tick * 20
        y = top + plot_height - (value / max_value) * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(svg_text(left - 12, y + 5, value, 12, "end"))
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>')
    parts.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>')
    for group_index, group in enumerate(groups):
        group_x = left + gap + group_index * (group_width + gap)
        for series_index, (_, key, color) in enumerate(series):
            value = transitions[group][key]
            x = group_x + series_index * (bar_width + 20)
            bar_height = (value / max_value) * plot_height
            y = top + plot_height - bar_height
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="4" fill="{color}"/>')
            parts.append(svg_text(x + bar_width / 2, y - 10, value, 14))
        parts.append(svg_text(group_x + group_width / 2 - 20, height - 44, MODEL_LABELS[group], 15))
    for index, (label, _, color) in enumerate(series):
        x = left + 20 + index * 250
        y = height - 18
        parts.append(f'<rect x="{x}" y="{y-12}" width="18" height="18" fill="{color}"/>')
        parts.append(svg_text(x + 26, y + 2, label, 13, "start"))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output


def pil_font(size: int, bold: bool = False):
    if ImageFont is None:
        raise RuntimeError("Pillow is required for PNG export.")
    font_path = FONT_BOLD if bold else FONT_REGULAR
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_png_text(
    draw,
    xy: tuple[float, float],
    text: Any,
    size: int,
    *,
    bold: bool = False,
    fill: str = "#111827",
    anchor: str = "mm",
) -> None:
    draw.text(
        xy,
        str(text),
        font=pil_font(size, bold=bold),
        fill=fill,
        anchor=anchor,
    )


def draw_rotated_png_text(
    image,
    xy: tuple[float, float],
    text: Any,
    size: int,
    *,
    bold: bool = False,
    fill: str = "#111827",
) -> None:
    font = pil_font(size, bold=bold)
    text_image = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    bbox = text_draw.textbbox((0, 0), str(text), font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_image = Image.new("RGBA", (text_width + 12, text_height + 12), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((6 - bbox[0], 6 - bbox[1]), str(text), font=font, fill=fill)
    rotated = text_image.rotate(90, expand=True)
    x, y = xy
    image.paste(
        rotated,
        (int(x - rotated.width / 2), int(y - rotated.height / 2)),
        rotated,
    )


def save_png(image, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(300, 300), optimize=True)
    return output


def write_test_accuracy_png(evaluation: dict[str, dict[str, Any]]) -> Path:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for PNG export.")

    output = FIGURES_DIR / "test_accuracy.png"
    width, height = 2400, 1500
    left, right, top, bottom = 270, 120, 150, 260
    plot_width = width - left - right
    plot_height = height - top - bottom
    labels = ["M1", "Selective", "Blind"]
    values = [
        evaluation["baseline"]["accuracy"],
        evaluation["selective"]["accuracy"],
        evaluation["blind"]["accuracy"],
    ]
    colors = ["#4b5563", "#2563eb", "#16a34a"]
    max_value = 0.70
    bar_width = 320
    gap = (plot_width - len(labels) * bar_width) / (len(labels) + 1)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for tick in range(0, 8):
        value = tick / 10
        y = top + plot_height - (value / max_value) * plot_height
        draw.line((left, y, width - right, y), fill="#e5e7eb", width=3)
        draw_png_text(draw, (left - 32, y), f"{value:.1f}", 58, fill="#374151", anchor="rm")
    draw.line((left, top, left, height - bottom), fill="#111827", width=5)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#111827", width=5)
    draw_rotated_png_text(image, (78, top + plot_height / 2), "Doğruluk", 66, bold=True)

    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + gap + index * (bar_width + gap)
        bar_height = (value / max_value) * plot_height
        y = top + plot_height - bar_height
        draw.rounded_rectangle(
            (x, y, x + bar_width, top + plot_height),
            radius=18,
            fill=color,
        )
        draw_png_text(draw, (x + bar_width / 2, y - 66), f"{value:.3f}", 68, bold=True)
        draw_png_text(draw, (x + bar_width / 2, height - 135), label, 72, bold=True)

    return save_png(image, output)


def write_learning_curve_png(curves: dict[str, list[tuple[int, float]]]) -> Path:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for PNG export.")

    output = FIGURES_DIR / "similar_learning_curve.png"
    width, height = 2600, 1600
    left, right, top, bottom = 280, 620, 170, 250
    plot_width = width - left - right
    plot_height = height - top - bottom
    min_y, max_y = 0.65, 1.00
    max_step = max(point[0] for points in curves.values() for point in points)
    colors = {"selective": "#2563eb", "blind": "#16a34a"}
    labels = {"selective": "Selective", "blind": "Blind"}

    def point_to_xy(step: int, value: float) -> tuple[float, float]:
        x = left + ((step - 1) / (max_step - 1)) * plot_width
        y = top + plot_height - ((value - min_y) / (max_y - min_y)) * plot_height
        return x, y

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for tick in range(0, 8):
        value = min_y + tick * (max_y - min_y) / 7
        y = top + plot_height - ((value - min_y) / (max_y - min_y)) * plot_height
        draw.line((left, y, width - right, y), fill="#e5e7eb", width=3)
        draw_png_text(draw, (left - 34, y), f"{value:.2f}", 56, fill="#374151", anchor="rm")
    for tick in range(0, 6):
        step = 1 + tick * (max_step - 1) / 5
        x = left + ((step - 1) / (max_step - 1)) * plot_width
        draw.line((x, top, x, height - bottom), fill="#f3f4f6", width=2)
        draw_png_text(draw, (x, height - 160), f"{int(round(step))}", 56, fill="#374151")
    draw.line((left, top, left, height - bottom), fill="#111827", width=5)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#111827", width=5)
    draw_rotated_png_text(image, (82, top + plot_height / 2), "Kümülatif doğruluk", 64, bold=True)
    draw_png_text(draw, (left + plot_width / 2, height - 52), "Fine-tuning adımı", 66, bold=True)

    for name in ("selective", "blind"):
        points = sampled_curve(curves[name], every=10)
        coords = [point_to_xy(step, value) for step, value in points]
        draw.line(coords, fill=colors[name], width=9, joint="curve")
        for step, value in points[::5]:
            x, y = point_to_xy(step, value)
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=colors[name])
        final_x, final_y = coords[-1]
        draw_png_text(
            draw,
            (final_x + 18, final_y),
            f"{labels[name]} {points[-1][1]:.3f}",
            52,
            bold=True,
            fill=colors[name],
            anchor="lm",
        )

    legend_x, legend_y = width - right + 80, top + 100
    for index, name in enumerate(("selective", "blind")):
        y = legend_y + index * 82
        draw.line((legend_x, y, legend_x + 100, y), fill=colors[name], width=10)
        draw_png_text(draw, (legend_x + 125, y), labels[name], 54, bold=True, anchor="lm")

    return save_png(image, output)


def write_transition_png(transitions: dict[str, dict[str, int]]) -> Path:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for PNG export.")

    output = FIGURES_DIR / "test_transitions.png"
    width, height = 2400, 1500
    left, right, top, bottom = 270, 120, 150, 300
    plot_width = width - left - right
    plot_height = height - top - bottom
    groups = ["selective", "blind"]
    series = [
        ("M1 düzeldi", "fixed_m1_mistakes", "#16a34a"),
        ("M1 bozuldu", "broke_m1_correct", "#dc2626"),
    ]
    max_value = 120
    group_width = 620
    bar_width = 210
    inner_gap = 70
    gap = (plot_width - len(groups) * group_width) / (len(groups) + 1)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for tick in range(0, 7):
        value = tick * 20
        y = top + plot_height - (value / max_value) * plot_height
        draw.line((left, y, width - right, y), fill="#e5e7eb", width=3)
        draw_png_text(draw, (left - 34, y), value, 58, fill="#374151", anchor="rm")
    draw.line((left, top, left, height - bottom), fill="#111827", width=5)
    draw.line((left, height - bottom, width - right, height - bottom), fill="#111827", width=5)
    draw_rotated_png_text(image, (82, top + plot_height / 2), "Sayı", 66, bold=True)

    for group_index, group in enumerate(groups):
        group_x = left + gap + group_index * (group_width + gap)
        for series_index, (_, key, color) in enumerate(series):
            value = transitions[group][key]
            x = group_x + series_index * (bar_width + inner_gap)
            bar_height = (value / max_value) * plot_height
            y = top + plot_height - bar_height
            draw.rounded_rectangle(
                (x, y, x + bar_width, top + plot_height),
                radius=18,
                fill=color,
            )
            draw_png_text(draw, (x + bar_width / 2, y - 66), value, 68, bold=True)
        draw_png_text(draw, (group_x + group_width / 2 - 70, height - 175), MODEL_LABELS[group], 72, bold=True)

    legend_x, legend_y = left + plot_width / 2 - 390, top + 55
    for index, (label, _, color) in enumerate(series):
        x = legend_x + index * 480
        draw.rounded_rectangle((x, legend_y - 28, x + 45, legend_y + 17), radius=5, fill=color)
        draw_png_text(draw, (x + 65, legend_y - 5), label, 58, anchor="lm")

    return save_png(image, output)


def write_publication_pngs(
    evaluation: dict[str, dict[str, Any]],
    curves: dict[str, list[tuple[int, float]]],
    transitions: dict[str, dict[str, int]],
) -> list[Path]:
    expected_outputs = [
        FIGURES_DIR / "test_accuracy.png",
        FIGURES_DIR / "similar_learning_curve.png",
        FIGURES_DIR / "test_transitions.png",
    ]
    if Image is None or ImageDraw is None or ImageFont is None:
        return [path for path in expected_outputs if path.exists()]
    return [
        write_test_accuracy_png(evaluation),
        write_learning_curve_png(curves),
        write_transition_png(transitions),
    ]


def build_report(
    evaluation: dict[str, dict[str, Any]],
    overlaps: dict[str, dict[str, int]],
    transitions: dict[str, dict[str, int]],
    loops: dict[str, Any],
    figure_paths: list[Path],
) -> str:
    eval_rows = []
    for strategy in MODEL_ORDER:
        row = evaluation[strategy]
        eval_rows.append(
            [
                MODEL_LABELS[strategy],
                f"{row['correct_rows']} / {row['total_rows']}",
                f"{row['accuracy']:.3f}",
                row.get("m1_failed_test_correct", ""),
                f"{row['accuracy_gain_vs_m1']:.3f}" if strategy != "baseline" else "0",
            ]
        )

    loop_rows = [
        [
            MODEL_LABELS[name],
            metrics["steps"],
            f"{metrics['similar_correct']} / {metrics['steps']}",
            f"{metrics['similar_accuracy']:.3f}",
            metrics["accepted"],
            metrics["rejected"],
        ]
        for name, metrics in (("selective", loops["selective"]), ("blind", loops["blind"]))
    ]

    transition_rows = [
        [
            MODEL_LABELS[name],
            transitions[name]["fixed_m1_mistakes"],
            transitions[name]["broke_m1_correct"],
            transitions[name]["kept_m1_correct"],
            transitions[name]["still_wrong"],
            transitions[name]["net_correct_delta"],
        ]
        for name in ("selective", "blind")
    ]

    selective_blind = overlaps["selective_vs_blind"]
    m1_selective = overlaps["baseline_vs_selective"]
    m1_blind = overlaps["baseline_vs_blind"]

    report = f"""# GSM8K_TR Seçici Fine-Tuning Deneyi Final Raporu

Bu rapor, GSM8K_TR üzerinde yürütülen seçici fine-tuning deneyinin Faz 7 analiz çıktısıdır. Ana soru şuydu: M1 modelinin çözemediği sorular için teacher modelden alınan çözümlerle yapılan LoRA fine-tuning, benzer sorulara ve sabit test kümesine genelleme sağlıyor mu?

## Deney Özeti

- Base model: `Qwen/Qwen3.5-4B`
- Test kümesi: 500 numeric-only GSM8K_TR sorusu
- Eğitim kaynağı: M1'in yanlış cevapladığı doğrulanmış sorulardan seçilen 500 örnek
- Teacher model: `openai/gpt-oss-120b:free`
- Fine-tuning yöntemi: LoRA adapter
- Karşılaştırılan stratejiler:
  - Selective: Güncelleme, benzer soru doğru çözülürse tutuldu.
  - Blind: Her güncelleme tutuldu.

## Final Test Sonuçları

{md_table(["Model", "Doğru", "Doğruluk", "M1 yanlışlarında doğru", "M1'e göre fark"], eval_rows)}

Final test setinde en yüksek doğruluk blind final adapter ile elde edildi. Blind strateji baseline'a göre +6 doğru cevap kazandırdı. Selective strateji ise baseline'ın 16 doğru altında kaldı.

## Benzer Soru Ara Değerlendirmesi

{md_table(["Strateji", "Adım", "Benzer doğru", "Benzer doğruluk", "Kabul", "Ret"], loop_rows)}

Benzer soru değerlendirmesinde selective daha iyi görünüyordu: 405/500. Blind ise 389/500 doğru yaptı. Ancak bu lokal üstünlük final test setine taşınmadı.

`similar_learning_curve` figürü, her fine-tuning adımı k için ilk k benzer soru değerlendirmesindeki kümülatif doğru oranını gösterir. İlk birkaç adımda oran çok oynak olduğu için grafik 10 adımda bir örneklenmiştir; final noktaları selective için 0.810, blind için 0.778'dir.

Selective ve blind'ın benzer soru adım karşılaştırması:

{md_table(["Durum", "Sayı"], [
    ["İkisi de doğru", loops["similar_pairwise"]["both_correct"]],
    ["Sadece selective doğru", loops["similar_pairwise"]["only_selective"]],
    ["Sadece blind doğru", loops["similar_pairwise"]["only_blind"]],
    ["İkisi de yanlış", loops["similar_pairwise"]["both_wrong"]],
])}

## Test Setinde Düzeltme ve Bozma Dengesi

{md_table(["Model", "M1 hatası düzeldi", "M1 doğrusu bozuldu", "M1 doğrusu korundu", "Hâlâ yanlış", "Net fark"], transition_rows)}

Selective, M1'in yanlış yaptığı 79 test sorusunu düzeltti; fakat M1'in doğru yaptığı 95 soruyu bozdu. Bu yüzden net etkisi -16 oldu. Blind, 85 eski hatayı düzeltti ve 79 eski doğruyu bozdu; net etkisi +6 oldu.

Test seti overlap özeti:

{md_table(["Karşılaştırma", "İkisi de doğru", "Sadece ilk model", "Sadece ikinci model", "İkisi de yanlış"], [
    ["M1 vs Selective", m1_selective["both_correct"], m1_selective["only_baseline"], m1_selective["only_selective"], m1_selective["both_wrong"]],
    ["M1 vs Blind", m1_blind["both_correct"], m1_blind["only_baseline"], m1_blind["only_blind"], m1_blind["both_wrong"]],
    ["Selective vs Blind", selective_blind["both_correct"], selective_blind["only_selective"], selective_blind["only_blind"], selective_blind["both_wrong"]],
])}

## Araştırma Sorularının Yanıtı

### Bir sorunun çözümünü öğrenmek, benzer soruları çözmeyi sağlıyor mu?

Kısmen evet. Selective döngüde 500 benzer sorunun 405'i doğru çözüldü. Blind döngüde ise 389 benzer soru doğru çözüldü. Bu, teacher çözümüyle yapılan tek örneklik LoRA güncellemesinin çoğu durumda lokal benzer soruya transfer sağlayabildiğini gösteriyor.

Ancak bu sonuç tek başına genel test başarısı anlamına gelmedi. Selective, benzer soru başarısında daha iyi olmasına rağmen final testte baseline'ın altına düştü. Dolayısıyla benzer soru başarısı faydalı bir ara sinyal, ama yeterli bir genelleme ölçütü değil.

### Seçici model güncelleme, kör güncellemeden daha mı iyi?

Bu deneyde final test açısından hayır. Selective strateji lokal benzer soru testinde daha iyi görünmesine rağmen sabit test kümesinde 0.548 doğruluk aldı. Blind strateji 0.592 doğruluk ile hem selective'i hem de 0.580 doğruluk alan baseline'ı geçti.

Bu sonuç, mevcut seçici kabul kuralının fazla lokal ve açgözlü kaldığını düşündürüyor. Bir güncellemenin hemen ardından gelen tek benzer soruda başarısız olması, o güncellemenin ileride veya genel test dağılımında faydasız olduğu anlamına gelmeyebilir.

## Başarılı ve Başarısız Adımlar Üzerine Notlar

- Selective strateji 405 güncellemeyi kabul etti, 95 güncellemeyi geri aldı.
- Blind strateji tüm 500 güncellemeyi tuttu; buna rağmen benzer soru testinde 111 adım yanlış çıktı.
- Faz 5'te blind'ın selective'den iyi olduğu 30 benzer soru vardı. Faz 6 sonucu bu olgunun önemli olduğunu gösterdi: bazı "o anda yanlış görünen" güncellemeler final test genellemesi için faydalı olmuş olabilir.
- Final testte blind'ın daha az bozma yapması belirleyici oldu: selective 95 eski doğruyu bozarken blind 79 eski doğruyu bozdu.

## Figürler

{chr(10).join(f"- `{path.relative_to(ROOT)}`" for path in figure_paths)}

## Sonuç

Deneyin ana bulgusu şudur: Teacher çözümüyle tek örnek üzerinden yapılan LoRA güncellemeleri lokal benzer sorulara çoğu zaman aktarım sağlayabiliyor, fakat tek benzer soru üzerinden yapılan seçici kabul mekanizması final test genellemesini garanti etmiyor.

Bu deneyde en iyi final test sonucu blind stratejiden geldi. Blind strateji küçük ama pozitif bir kazanım sağladı: 0.580 baseline doğruluğundan 0.592'ye çıktı. Selective strateji ise lokal doğrulamada daha kontrollü görünmesine rağmen 0.548 ile geriledi.

## Sınırlılıklar ve Sonraki Adımlar

- Sonuçlar tek seed ve tek LoRA hiperparametre setiyle elde edildi.
- Kabul kriteri yalnızca bir benzer soru üzerinden verildi; daha küçük bir validation havuzu daha sağlam olabilir.
- Numeric-only evaluator final sayıyı ölçüyor; çözüm kalitesinin tamamını değerlendirmiyor.
- Daha düşük learning rate, replay örnekleri veya çoklu benzer soru kabul kriteri selective stratejinin bozma etkisini azaltabilir.
"""
    return report


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    evaluation = load_evaluation_rows()
    all_ids, correct_ids, _ = prediction_sets()
    loops = loop_metrics()

    overlaps = {
        "baseline_vs_selective": pairwise_overlap(all_ids, correct_ids, "baseline", "selective"),
        "baseline_vs_blind": pairwise_overlap(all_ids, correct_ids, "baseline", "blind"),
        "selective_vs_blind": pairwise_overlap(all_ids, correct_ids, "selective", "blind"),
    }
    transitions = {
        "selective": test_transition_metrics(all_ids, correct_ids, "selective"),
        "blind": test_transition_metrics(all_ids, correct_ids, "blind"),
    }

    figure_paths = [
        write_test_accuracy_chart(evaluation),
        write_learning_curve_chart(loops["curves"]),
        write_transition_chart(transitions),
    ]
    png_figure_paths = write_publication_pngs(
        evaluation=evaluation,
        curves=loops["curves"],
        transitions=transitions,
    )
    all_figure_paths = figure_paths + png_figure_paths

    analysis = {
        "evaluation": evaluation,
        "overlaps": overlaps,
        "transitions": transitions,
        "loop_metrics": {key: value for key, value in loops.items() if key != "curves"},
        "figures": [str(path.relative_to(ROOT)) for path in all_figure_paths],
        "svg_figures": [str(path.relative_to(ROOT)) for path in figure_paths],
        "png_figures": [str(path.relative_to(ROOT)) for path in png_figure_paths],
    }
    write_json(RESULTS_DIR / "phase7_analysis.json", analysis)

    report = build_report(
        evaluation=evaluation,
        overlaps=overlaps,
        transitions=transitions,
        loops=loops,
        figure_paths=all_figure_paths,
    )
    (RESULTS_DIR / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
