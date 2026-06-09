from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


FINAL_ANSWER_RE = re.compile(
    r"(?im)^\s*final\s+cevap\s*:\s*(?P<answer>.+?)\s*$"
)
NUMBER_RE = re.compile(
    r"(?<![\w/])[-+]?\d+(?:[.,\u00a0\u202f ]\d+)*(?:/\d+(?:[.,\u00a0\u202f ]\d+)*)?"
)


def extract_last_number(text: str | None) -> str | None:
    if not text:
        return None
    matches = list(NUMBER_RE.finditer(text))
    if not matches:
        return None
    return clean_number_text(matches[-1].group(0))


def extract_model_final_answer(text: str | None) -> str | None:
    if not text:
        return None

    final_answer_matches = list(FINAL_ANSWER_RE.finditer(text))
    if final_answer_matches:
        for match in reversed(final_answer_matches):
            answer = extract_last_number(match.group("answer"))
            if answer is not None:
                return answer

    lower_text = text.lower()
    marker = "final cevap:"
    marker_index = lower_text.rfind(marker)
    if marker_index != -1:
        return extract_last_number(text[marker_index + len(marker) :])

    return extract_last_number(text)


def clean_number_text(value: str) -> str:
    return (
        value.strip()
        .replace("\u2212", "-")
        .strip(" \t\r\n.,;:!?)]}")
        .lstrip("([{")
    )


def decimal_candidates(value: str | None) -> set[Decimal]:
    if value is None:
        return set()

    cleaned = clean_number_text(value)
    if not cleaned:
        return set()

    if "/" in cleaned:
        numerator_text, denominator_text = cleaned.split("/", maxsplit=1)
        numerator_candidates = decimal_candidates(numerator_text)
        denominator_candidates = decimal_candidates(denominator_text)
        return {
            numerator / denominator
            for numerator in numerator_candidates
            for denominator in denominator_candidates
            if denominator != 0
        }

    cleaned = cleaned.replace("%", "").replace(" ", "")
    candidates = {
        cleaned,
        cleaned.replace(",", "."),
        cleaned.replace(".", "").replace(",", "."),
        cleaned.replace(",", "").replace(".", "."),
        cleaned.replace(".", "").replace(",", ""),
    }

    decimals: set[Decimal] = set()
    for candidate in candidates:
        if not candidate or candidate in {"+", "-", "."}:
            continue
        try:
            decimals.add(Decimal(candidate))
        except InvalidOperation:
            continue
    return decimals


def primary_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None

    cleaned = clean_number_text(value).replace("%", "").replace(" ", "")
    if not cleaned:
        return None

    if "/" in cleaned:
        numerator_text, denominator_text = cleaned.split("/", maxsplit=1)
        numerator = primary_decimal(numerator_text)
        denominator = primary_decimal(denominator_text)
        if numerator is None or denominator in {None, Decimal(0)}:
            return None
        return numerator / denominator

    if "." in cleaned and "," in cleaned:
        last_dot = cleaned.rfind(".")
        last_comma = cleaned.rfind(",")
        if last_comma > last_dot:
            normalized = cleaned.replace(".", "").replace(",", ".")
        else:
            normalized = cleaned.replace(",", "")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts) > 2 or all(len(part) == 3 for part in parts[1:]):
            normalized = cleaned.replace(",", "")
        else:
            normalized = cleaned.replace(",", ".")
    elif "." in cleaned:
        parts = cleaned.split(".")
        if len(parts) > 2 or all(len(part) == 3 for part in parts[1:]):
            normalized = cleaned.replace(".", "")
        else:
            normalized = cleaned
    else:
        normalized = cleaned

    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def canonical_number(value: str | None) -> str | None:
    selected = primary_decimal(value)
    if selected is None:
        return None

    text = format(selected, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"", "-0", "+0"}:
        return "0"
    return text


def numbers_match(
    predicted: str | None,
    reference: str | None,
    tolerance: Decimal = Decimal("0.000001"),
) -> bool:
    predicted_candidates = decimal_candidates(predicted)
    reference_candidates = decimal_candidates(reference)
    if not predicted_candidates or not reference_candidates:
        return False

    return any(
        abs(predicted_value - reference_value) <= tolerance
        for predicted_value in predicted_candidates
        for reference_value in reference_candidates
    )
