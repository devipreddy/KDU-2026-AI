from __future__ import annotations

import argparse
import calendar
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from app.models.enums import DocumentType
from app.services.clinical_metadata import (
    DIAGNOSIS_TERMS,
    ICD_CODE_PATTERN,
    extract_hospitals,
    extract_physicians,
    ordered_unique,
)

QUERY_UNDERSTANDING_VERSION = "rule_based_query_understanding_v1"

MONTH_ALIASES = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
MONTH_NAME_PATTERN = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
DATE_VALUE_PATTERN = (
    rf"(?:\d{{4}}-\d{{1,2}}-\d{{1,2}})|"
    rf"(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}})|"
    rf"(?:(?:{MONTH_NAME_PATTERN})\.?\s+\d{{1,2}},?\s+\d{{4}})"
)
DATE_VALUE_RE = re.compile(rf"^{DATE_VALUE_PATTERN}$", flags=re.IGNORECASE)
BETWEEN_DATE_PATTERN = re.compile(
    rf"\b(?:between|from)\s+(?P<start>{DATE_VALUE_PATTERN})\s+"
    rf"(?:and|to|through|-)\s+(?P<end>{DATE_VALUE_PATTERN})",
    flags=re.IGNORECASE,
)
DATE_PATTERN = re.compile(DATE_VALUE_PATTERN, flags=re.IGNORECASE)
MONTH_YEAR_PATTERN = re.compile(
    rf"\b(?P<month>{MONTH_NAME_PATTERN})\.?\s+(?P<year>20\d{{2}})\b",
    flags=re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(?P<year>20\d{2})\b")
PATIENT_REF_PATTERNS = (
    re.compile(r"\bPATIENT[\s_-]?REF[\s_:-]*(?P<number>\d{1,6})\b", flags=re.IGNORECASE),
    re.compile(
        r"\bpatient\s+(?:reference|ref)\s*[:#=]?\s*(?P<number>\d{1,6})\b",
        flags=re.IGNORECASE,
    ),
)

DIAGNOSIS_CATEGORY_DIAGNOSES = {
    "cardiac": [
        "Atrial fibrillation",
        "Congestive heart failure",
        "Coronary artery disease",
        "Hypertension",
        "Stable angina",
    ],
    "endocrine": [
        "Type 2 diabetes mellitus",
        "Hyperglycemia",
        "Diabetic neuropathy",
        "Hypothyroidism",
        "Prediabetes",
    ],
    "respiratory": [
        "Asthma exacerbation",
        "Chronic obstructive pulmonary disease",
        "Pneumonia",
        "Obstructive sleep apnea",
        "Acute bronchitis",
    ],
    "renal": [
        "Chronic kidney disease stage 3",
        "Acute kidney injury",
        "Nephrolithiasis",
        "Proteinuria",
        "Electrolyte imbalance",
    ],
    "neurology": [
        "Migraine without aura",
        "Transient ischemic attack",
        "Peripheral neuropathy",
        "Seizure disorder",
        "Vertigo",
    ],
    "orthopedic": [
        "Osteoarthritis of knee",
        "Distal radius fracture",
        "Lumbar strain",
        "Rotator cuff tendinopathy",
        "Hip bursitis",
    ],
    "infectious_disease": [
        "Urinary tract infection",
        "Cellulitis",
        "Influenza",
        "COVID-19 infection",
        "Bacterial sinusitis",
        "Malaria",
    ],
    "gastroenterology": [
        "Gastroesophageal reflux disease",
        "Irritable bowel syndrome",
        "Diverticulitis",
        "Cholelithiasis",
        "Iron deficiency anemia",
    ],
    "behavioral_health": [
        "Generalized anxiety disorder",
        "Major depressive disorder",
        "Insomnia",
        "Adjustment disorder",
        "Post-traumatic stress disorder",
    ],
}

DIAGNOSIS_CATEGORY_ALIASES = {
    "cardiac": "cardiac",
    "cardiac issues": "cardiac",
    "cardiology": "cardiac",
    "heart": "cardiac",
    "heart disease": "cardiac",
    "heart rhythm": "cardiac",
    "irregular heartbeat": "cardiac",
    "chest pain": "cardiac",
    "blood pressure": "cardiac",
    "high blood pressure": "cardiac",
    "endocrine": "endocrine",
    "diabetes": "endocrine",
    "diabetic": "endocrine",
    "blood sugar": "endocrine",
    "glucose": "endocrine",
    "respiratory": "respiratory",
    "pulmonary": "respiratory",
    "lung": "respiratory",
    "breathing": "respiratory",
    "shortness of breath": "respiratory",
    "sob": "respiratory",
    "renal": "renal",
    "kidney": "renal",
    "kidney function": "renal",
    "nephrology": "renal",
    "neurology": "neurology",
    "neurologic": "neurology",
    "orthopedic": "orthopedic",
    "orthopedics": "orthopedic",
    "infectious disease": "infectious_disease",
    "infection": "infectious_disease",
    "uti": "infectious_disease",
    "flu": "infectious_disease",
    "gastroenterology": "gastroenterology",
    "gastrointestinal": "gastroenterology",
    "gi": "gastroenterology",
    "behavioral health": "behavioral_health",
    "mental health": "behavioral_health",
    "psychiatry": "behavioral_health",
}

DIAGNOSIS_QUERY_ALIASES = {
    "ckd": "Chronic kidney disease stage 3",
    "chronic kidney disease": "Chronic kidney disease stage 3",
    "copd": "Chronic obstructive pulmonary disease",
    "covid": "COVID-19 infection",
    "covid 19": "COVID-19 infection",
    "heart failure": "Congestive heart failure",
    "high blood pressure": "Hypertension",
    "blood pressure": "Hypertension",
    "diabetes": "Type 2 diabetes mellitus",
    "blood sugar": "Type 2 diabetes mellitus",
    "glucose control": "Type 2 diabetes mellitus",
    "diabetic neuropathy": "Diabetic neuropathy",
    "afib": "Atrial fibrillation",
    "atrial fib": "Atrial fibrillation",
    "irregular heartbeat": "Atrial fibrillation",
    "heart rhythm": "Atrial fibrillation",
    "shortness of breath": "Asthma exacerbation",
    "sob": "Asthma exacerbation",
    "sleep apnea": "Obstructive sleep apnea",
    "uti": "Urinary tract infection",
    "flu": "Influenza",
}

DOCUMENT_TYPE_ALIASES = {
    "typed pdf": DocumentType.TYPED_PDF,
    "typed pdfs": DocumentType.TYPED_PDF,
    "typed document": DocumentType.TYPED_PDF,
    "typed documents": DocumentType.TYPED_PDF,
    "scanned pdf": DocumentType.SCANNED_PDF,
    "scanned pdfs": DocumentType.SCANNED_PDF,
    "scanned document": DocumentType.SCANNED_PDF,
    "scanned documents": DocumentType.SCANNED_PDF,
    "scan": DocumentType.SCANNED_PDF,
    "scans": DocumentType.SCANNED_PDF,
    "clinical note": DocumentType.CLINICAL_NOTE,
    "clinical notes": DocumentType.CLINICAL_NOTE,
    "free text note": DocumentType.CLINICAL_NOTE,
    "free-text note": DocumentType.CLINICAL_NOTE,
    "typed note": DocumentType.CLINICAL_NOTE,
    "typed notes": DocumentType.CLINICAL_NOTE,
    "discharge summary": DocumentType.DISCHARGE_SUMMARY,
    "discharge summaries": DocumentType.DISCHARGE_SUMMARY,
    "lab report": DocumentType.LAB_REPORT,
    "lab reports": DocumentType.LAB_REPORT,
    "laboratory report": DocumentType.LAB_REPORT,
    "laboratory reports": DocumentType.LAB_REPORT,
    "labs": DocumentType.LAB_REPORT,
    "prescription": DocumentType.PRESCRIPTION,
    "prescriptions": DocumentType.PRESCRIPTION,
    "rx": DocumentType.PRESCRIPTION,
    "handwritten note": DocumentType.HANDWRITTEN_NOTE,
    "handwritten notes": DocumentType.HANDWRITTEN_NOTE,
    "handwritten": DocumentType.HANDWRITTEN_NOTE,
}
DOCUMENT_TYPE_QUERY_EXPANSIONS = {
    DocumentType.PRESCRIPTION: [
        DocumentType.PRESCRIPTION,
        DocumentType.HANDWRITTEN_NOTE,
        DocumentType.SCANNED_PDF,
        DocumentType.TYPED_PDF,
        DocumentType.OTHER,
    ],
}


@dataclass(frozen=True)
class TemporalFilter:
    field: str
    operator: str
    start_date: str | None
    end_date: str | None
    granularity: str
    matched_text: str


@dataclass(frozen=True)
class DiagnosisConcept:
    concept: str
    concept_type: str
    matched_text: str
    diagnoses: list[str]


@dataclass(frozen=True)
class QueryUnderstandingResult:
    original_query: str
    normalized_query: str
    temporal_filters: list[TemporalFilter]
    diagnosis_concepts: list[DiagnosisConcept]
    patient_refs: list[str]
    hospitals: list[str]
    physicians: list[str]
    document_types: list[str]
    icd_codes: list[str]
    metadata_filters: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "parser": QUERY_UNDERSTANDING_VERSION,
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "temporal_filters": [asdict(item) for item in self.temporal_filters],
            "diagnosis_concepts": [asdict(item) for item in self.diagnosis_concepts],
            "patient_refs": self.patient_refs,
            "hospitals": self.hospitals,
            "physicians": self.physicians,
            "document_types": self.document_types,
            "icd_codes": self.icd_codes,
            "metadata_filters": self.metadata_filters,
        }


def understand_query(query: str) -> QueryUnderstandingResult:
    normalized_query = normalize_query_text(query)
    temporal_filters = extract_temporal_filters(query)
    diagnosis_concepts = extract_diagnosis_concepts(query)
    patient_refs = extract_patient_refs(query)
    hospitals = [
        hospital for hospital in extract_hospitals(query) if not is_soft_facility_signal(hospital)
    ]
    physicians = extract_physicians(query)
    document_types = extract_document_types(query)
    icd_codes = ordered_unique(match.group(0).upper() for match in ICD_CODE_PATTERN.finditer(query))

    return QueryUnderstandingResult(
        original_query=query,
        normalized_query=normalized_query,
        temporal_filters=temporal_filters,
        diagnosis_concepts=diagnosis_concepts,
        patient_refs=patient_refs,
        hospitals=hospitals,
        physicians=physicians,
        document_types=document_types,
        icd_codes=icd_codes,
        metadata_filters=build_metadata_filters(
            temporal_filters=temporal_filters,
            diagnosis_concepts=diagnosis_concepts,
            patient_refs=patient_refs,
            hospitals=hospitals,
            physicians=physicians,
            document_types=document_types,
            icd_codes=icd_codes,
        ),
    )


def normalize_query_text(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def extract_temporal_filters(query: str) -> list[TemporalFilter]:
    filters_with_spans: list[tuple[tuple[int, int], TemporalFilter]] = []
    consumed_spans: list[tuple[int, int]] = []

    for match in BETWEEN_DATE_PATTERN.finditer(query):
        start = parse_date_value(match.group("start"))
        end = parse_date_value(match.group("end"))
        if start and end:
            if end < start:
                start, end = end, start
            filters_with_spans.append(
                (
                    match.span(),
                    TemporalFilter(
                        field="visit_date",
                        operator="between",
                        start_date=start.isoformat(),
                        end_date=end.isoformat(),
                        granularity="date_range",
                        matched_text=match.group(0),
                    ),
                )
            )
            consumed_spans.append(match.span())

    for match in iter_quarter_matches(query):
        if span_overlaps(match["span"], consumed_spans):
            continue
        quarter_start, quarter_end = quarter_bounds(match["year"], match["quarter"])
        operator = context_operator(query, match["span"][0])
        filters_with_spans.append(
            (
                match["span"],
                build_range_filter(
                    start=quarter_start,
                    end=quarter_end,
                    granularity="quarter",
                    matched_text=match["text"],
                    operator_hint=operator,
                ),
            )
        )
        consumed_spans.append(match["span"])

    for match in DATE_PATTERN.finditer(query):
        if span_overlaps(match.span(), consumed_spans):
            continue
        parsed = parse_date_value(match.group(0))
        if parsed is None:
            continue
        operator = context_operator(query, match.start())
        filters_with_spans.append(
            (
                match.span(),
                TemporalFilter(
                    field="visit_date",
                    operator=operator or "eq",
                    start_date=parsed.isoformat() if operator != "lte" else None,
                    end_date=parsed.isoformat() if operator != "gte" else None,
                    granularity="date",
                    matched_text=match.group(0),
                ),
            )
        )
        consumed_spans.append(match.span())

    for match in MONTH_YEAR_PATTERN.finditer(query):
        if span_overlaps(match.span(), consumed_spans):
            continue
        month = month_number(match.group("month"))
        year = int(match.group("year"))
        month_start = date(year, month, 1)
        month_end = date(year, month, calendar.monthrange(year, month)[1])
        filters_with_spans.append(
            (
                match.span(),
                build_range_filter(
                    start=month_start,
                    end=month_end,
                    granularity="month",
                    matched_text=match.group(0),
                    operator_hint=context_operator(query, match.start()),
                ),
            )
        )
        consumed_spans.append(match.span())

    for match in YEAR_PATTERN.finditer(query):
        if span_overlaps(match.span(), consumed_spans):
            continue
        year = int(match.group("year"))
        year_start = date(year, 1, 1)
        year_end = date(year, 12, 31)
        filters_with_spans.append(
            (
                match.span(),
                build_range_filter(
                    start=year_start,
                    end=year_end,
                    granularity="year",
                    matched_text=match.group("year"),
                    operator_hint=context_operator(query, match.start()),
                ),
            )
        )
        consumed_spans.append(match.span())

    return [item for _, item in sorted(filters_with_spans, key=lambda item: item[0][0])]


def iter_quarter_matches(query: str) -> list[dict[str, Any]]:
    patterns = (
        re.compile(r"\bQ(?P<quarter>[1-4])\s*(?P<year>20\d{2})\b", flags=re.IGNORECASE),
        re.compile(r"\b(?P<year>20\d{2})\s*Q(?P<quarter>[1-4])\b", flags=re.IGNORECASE),
        re.compile(
            r"\b(?P<word>first|second|third|fourth)\s+quarter(?:\s+of)?\s+"
            r"(?P<year>20\d{2})\b",
            flags=re.IGNORECASE,
        ),
    )
    quarter_words = {"first": 1, "second": 2, "third": 3, "fourth": 4}
    matches: list[dict[str, Any]] = []
    for pattern in patterns:
        for match in pattern.finditer(query):
            quarter = (
                int(match.group("quarter"))
                if "quarter" in match.groupdict() and match.group("quarter")
                else quarter_words[match.group("word").lower()]
            )
            matches.append(
                {
                    "span": match.span(),
                    "text": match.group(0),
                    "year": int(match.group("year")),
                    "quarter": quarter,
                }
            )
    return sorted(matches, key=lambda item: item["span"][0])


def extract_diagnosis_concepts(query: str) -> list[DiagnosisConcept]:
    concepts: list[DiagnosisConcept] = []
    seen: set[tuple[str, str]] = set()

    for term, canonical in sorted(
        {**DIAGNOSIS_TERMS, **DIAGNOSIS_QUERY_ALIASES}.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if not contains_query_term(query, term):
            continue
        key = ("diagnosis", canonical.lower())
        if key in seen:
            continue
        concepts.append(
            DiagnosisConcept(
                concept=canonical,
                concept_type="diagnosis",
                matched_text=term,
                diagnoses=[canonical],
            )
        )
        seen.add(key)

    for term, category in sorted(
        DIAGNOSIS_CATEGORY_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if not contains_query_term(query, term):
            continue
        key = ("category", category)
        if key in seen:
            continue
        concepts.append(
            DiagnosisConcept(
                concept=category,
                concept_type="category",
                matched_text=term,
                diagnoses=DIAGNOSIS_CATEGORY_DIAGNOSES[category],
            )
        )
        seen.add(key)

    return concepts


def extract_patient_refs(query: str) -> list[str]:
    patient_refs: list[str] = []
    for pattern in PATIENT_REF_PATTERNS:
        for match in pattern.finditer(query):
            patient_refs.append(normalize_patient_ref(match.group("number")))
    return ordered_unique(patient_refs)


def extract_document_types(query: str) -> list[str]:
    matched_types: list[str] = []
    for alias, document_type in sorted(
        DOCUMENT_TYPE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if contains_query_term(query, alias):
            matched_types.extend(
                expanded_type.value
                for expanded_type in DOCUMENT_TYPE_QUERY_EXPANSIONS.get(
                    document_type,
                    [document_type],
                )
            )
    return ordered_unique(matched_types)


def build_metadata_filters(
    *,
    temporal_filters: list[TemporalFilter],
    diagnosis_concepts: list[DiagnosisConcept],
    patient_refs: list[str],
    hospitals: list[str],
    physicians: list[str],
    document_types: list[str],
    icd_codes: list[str],
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if temporal_filters:
        filters["visit_date"] = [asdict(item) for item in temporal_filters]
    if patient_refs:
        filters["patient_ref"] = patient_refs
    if hospitals:
        filters["hospital"] = hospitals
    if physicians:
        filters["physician"] = physicians
    if document_types:
        filters["document_type"] = document_types
    if icd_codes:
        filters["icd_codes"] = icd_codes

    exact_diagnoses = [
        concept.concept for concept in diagnosis_concepts if concept.concept_type == "diagnosis"
    ]
    diagnosis_categories = [
        concept.concept for concept in diagnosis_concepts if concept.concept_type == "category"
    ]
    if exact_diagnoses:
        filters["diagnosis"] = ordered_unique(exact_diagnoses)
    if diagnosis_categories:
        filters["diagnosis_category"] = ordered_unique(diagnosis_categories)
    return filters


def is_soft_facility_signal(value: str) -> bool:
    return bool(re.match(r"^U\.S\.S\.", value, flags=re.IGNORECASE))


def build_range_filter(
    *,
    start: date,
    end: date,
    granularity: str,
    matched_text: str,
    operator_hint: str | None,
) -> TemporalFilter:
    if operator_hint == "gte":
        return TemporalFilter(
            field="visit_date",
            operator="gte",
            start_date=start.isoformat(),
            end_date=None,
            granularity=granularity,
            matched_text=matched_text,
        )
    if operator_hint == "lte":
        return TemporalFilter(
            field="visit_date",
            operator="lte",
            start_date=None,
            end_date=end.isoformat(),
            granularity=granularity,
            matched_text=matched_text,
        )
    return TemporalFilter(
        field="visit_date",
        operator="between",
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        granularity=granularity,
        matched_text=matched_text,
    )


def context_operator(query: str, start_offset: int) -> str | None:
    context = query[max(0, start_offset - 32) : start_offset].lower()
    if re.search(r"\b(?:after|since|from)\s+$", context):
        return "gte"
    if re.search(r"\b(?:before|until|through|prior to)\s+$", context):
        return "lte"
    if re.search(r"\b(?:on|at)\s+$", context):
        return "eq"
    return None


def parse_date_value(value: str) -> date | None:
    cleaned = normalize_query_text(value).replace(",", "")
    if not DATE_VALUE_RE.match(value.strip()):
        return None

    iso_match = re.fullmatch(r"(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})", cleaned)
    if iso_match:
        return safe_date(
            int(iso_match.group("year")),
            int(iso_match.group("month")),
            int(iso_match.group("day")),
        )

    slash_match = re.fullmatch(
        r"(?P<month>\d{1,2})[/-](?P<day>\d{1,2})[/-](?P<year>\d{2,4})",
        cleaned,
    )
    if slash_match:
        year = int(slash_match.group("year"))
        if year < 100:
            year += 2000
        return safe_date(year, int(slash_match.group("month")), int(slash_match.group("day")))

    month_match = re.fullmatch(
        rf"(?P<month>{MONTH_NAME_PATTERN})\.?\s+(?P<day>\d{{1,2}})\s+(?P<year>\d{{4}})",
        cleaned,
        flags=re.IGNORECASE,
    )
    if month_match:
        return safe_date(
            int(month_match.group("year")),
            month_number(month_match.group("month")),
            int(month_match.group("day")),
        )
    return None


def quarter_bounds(year: int, quarter: int) -> tuple[date, date]:
    start_month = ((quarter - 1) * 3) + 1
    end_month = start_month + 2
    return date(year, start_month, 1), date(
        year,
        end_month,
        calendar.monthrange(year, end_month)[1],
    )


def month_number(value: str) -> int:
    key = value.lower().rstrip(".")
    return MONTH_ALIASES[key]


def safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def span_overlaps(span: tuple[int, int], existing_spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(
        start < existing_end and end > existing_start
        for existing_start, existing_end in existing_spans
    )


def normalize_patient_ref(raw_number: str) -> str:
    return f"PATIENT_REF_{int(raw_number):04d}"


def contains_query_term(query: str, term: str) -> bool:
    escaped = re.escape(term).replace(r"\ ", r"\s+").replace(r"\-", r"[-\s]?")
    return re.search(rf"(?<!\w){escaped}(?:s)?(?!\w)", query, flags=re.IGNORECASE) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a healthcare retrieval query.")
    parser.add_argument("query", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = understand_query(" ".join(args.query))
    print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
