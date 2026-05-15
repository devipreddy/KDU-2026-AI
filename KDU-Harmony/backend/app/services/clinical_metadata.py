from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.models.document import Document

CLINICAL_METADATA_EXTRACTOR = "rule_based_clinical_metadata_v1"

DIAGNOSIS_TERMS = {
    "atrial fibrillation": "Atrial fibrillation",
    "congestive heart failure": "Congestive heart failure",
    "coronary artery disease": "Coronary artery disease",
    "hypertension": "Hypertension",
    "stable angina": "Stable angina",
    "type 2 diabetes mellitus": "Type 2 diabetes mellitus",
    "type 2 diabetes": "Type 2 diabetes mellitus",
    "hyperglycemia": "Hyperglycemia",
    "diabetic neuropathy": "Diabetic neuropathy",
    "hypothyroidism": "Hypothyroidism",
    "prediabetes": "Prediabetes",
    "asthma exacerbation": "Asthma exacerbation",
    "chronic obstructive pulmonary disease": "Chronic obstructive pulmonary disease",
    "pneumonia": "Pneumonia",
    "obstructive sleep apnea": "Obstructive sleep apnea",
    "acute bronchitis": "Acute bronchitis",
    "chronic kidney disease stage 3": "Chronic kidney disease stage 3",
    "acute kidney injury": "Acute kidney injury",
    "nephrolithiasis": "Nephrolithiasis",
    "proteinuria": "Proteinuria",
    "electrolyte imbalance": "Electrolyte imbalance",
    "migraine without aura": "Migraine without aura",
    "transient ischemic attack": "Transient ischemic attack",
    "peripheral neuropathy": "Peripheral neuropathy",
    "seizure disorder": "Seizure disorder",
    "vertigo": "Vertigo",
    "osteoarthritis of knee": "Osteoarthritis of knee",
    "distal radius fracture": "Distal radius fracture",
    "lumbar strain": "Lumbar strain",
    "rotator cuff tendinopathy": "Rotator cuff tendinopathy",
    "hip bursitis": "Hip bursitis",
    "urinary tract infection": "Urinary tract infection",
    "cellulitis": "Cellulitis",
    "influenza": "Influenza",
    "covid-19 infection": "COVID-19 infection",
    "bacterial sinusitis": "Bacterial sinusitis",
    "malaria": "Malaria",
    "gastroesophageal reflux disease": "Gastroesophageal reflux disease",
    "irritable bowel syndrome": "Irritable bowel syndrome",
    "diverticulitis": "Diverticulitis",
    "cholelithiasis": "Cholelithiasis",
    "iron deficiency anemia": "Iron deficiency anemia",
    "generalized anxiety disorder": "Generalized anxiety disorder",
    "major depressive disorder": "Major depressive disorder",
    "insomnia": "Insomnia",
    "adjustment disorder": "Adjustment disorder",
    "post-traumatic stress disorder": "Post-traumatic stress disorder",
}

MEDICATION_TERMS = {
    "metoprolol": "metoprolol",
    "beta blockers": "Beta blockers",
    "atorvastatin": "atorvastatin",
    "lisinopril": "lisinopril",
    "aspirin": "aspirin",
    "furosemide": "furosemide",
    "metformin": "metformin",
    "insulin glargine": "insulin glargine",
    "levothyroxine": "levothyroxine",
    "gabapentin": "gabapentin",
    "semaglutide": "semaglutide",
    "albuterol": "albuterol",
    "fluticasone": "fluticasone",
    "azithromycin": "azithromycin",
    "prednisone": "prednisone",
    "tiotropium": "tiotropium",
    "sodium bicarbonate": "sodium bicarbonate",
    "losartan": "losartan",
    "tamsulosin": "tamsulosin",
    "calcium acetate": "calcium acetate",
    "potassium chloride": "potassium chloride",
    "sumatriptan": "sumatriptan",
    "levetiracetam": "levetiracetam",
    "meclizine": "meclizine",
    "topiramate": "topiramate",
    "clopidogrel": "clopidogrel",
    "acetaminophen": "acetaminophen",
    "ibuprofen": "ibuprofen",
    "diclofenac": "diclofenac",
    "cyclobenzaprine": "cyclobenzaprine",
    "nitrofurantoin": "nitrofurantoin",
    "cephalexin": "cephalexin",
    "oseltamivir": "oseltamivir",
    "amoxicillin clavulanate": "amoxicillin clavulanate",
    "artemether lumefantrine": "artemether lumefantrine",
    "omeprazole": "omeprazole",
    "dicyclomine": "dicyclomine",
    "ferrous sulfate": "ferrous sulfate",
    "ondansetron": "ondansetron",
    "polyethylene glycol": "polyethylene glycol",
    "sertraline": "sertraline",
    "trazodone": "trazodone",
    "hydroxyzine": "hydroxyzine",
    "escitalopram": "escitalopram",
    "melatonin": "melatonin",
    "belladonna": "Belladonna",
    "amphogel": "Amphogel",
}

SYMPTOM_TERMS = {
    "palpitations": "palpitations",
    "chest pressure": "chest pressure",
    "shortness of breath": "shortness of breath",
    "lower extremity edema": "lower extremity edema",
    "exercise intolerance": "exercise intolerance",
    "increased thirst": "increased thirst",
    "fatigue": "fatigue",
    "numbness in feet": "numbness in feet",
    "weight gain": "weight gain",
    "elevated fasting glucose": "elevated fasting glucose",
    "wheezing": "wheezing",
    "productive cough": "productive cough",
    "nighttime awakenings": "nighttime awakenings",
    "fever with cough": "fever with cough",
    "reduced urine output": "reduced urine output",
    "flank pain": "flank pain",
    "ankle swelling": "ankle swelling",
    "abnormal creatinine": "abnormal creatinine",
    "headache": "headache",
    "brief speech difficulty": "brief speech difficulty",
    "tingling in hands": "tingling in hands",
    "dizziness": "dizziness",
    "visual disturbance": "visual disturbance",
    "joint stiffness": "joint stiffness",
    "wrist pain after fall": "wrist pain after fall",
    "low back pain": "low back pain",
    "shoulder weakness": "shoulder weakness",
    "lateral hip pain": "lateral hip pain",
    "fever": "fever",
    "localized redness": "localized redness",
    "dysuria": "dysuria",
    "nasal congestion": "nasal congestion",
    "body aches": "body aches",
    "cyclic fever with chills": "cyclic fever with chills",
    "epigastric burning": "epigastric burning",
    "abdominal cramping": "abdominal cramping",
    "left lower quadrant pain": "left lower quadrant pain",
    "postprandial nausea": "postprandial nausea",
    "fatigue with low ferritin": "fatigue with low ferritin",
    "persistent worry": "persistent worry",
    "low mood": "low mood",
    "difficulty sleeping": "difficulty sleeping",
    "reduced concentration": "reduced concentration",
    "stress-related irritability": "stress-related irritability",
}

KNOWN_HOSPITALS = (
    "Harmony General Hospital",
    "Northlake Medical Center",
    "Cedar Valley Clinic",
    "St. Isabel Regional",
    "Riverside Community Hospital",
    "Mercy West Health",
)

SECTION_ALIASES = {
    "assessment": "Assessment",
    "chief complaint": "Chief Complaint",
    "diagnosis": "Diagnosis",
    "diagnoses": "Diagnosis",
    "discharge summary": "Discharge Summary",
    "history": "History",
    "history of present illness": "History Of Present Illness",
    "hpi": "HPI",
    "icd": "ICD-10",
    "icd 10": "ICD-10",
    "icd-10": "ICD-10",
    "lab results": "Lab Results",
    "laboratory results": "Lab Results",
    "medication": "Medications",
    "medications": "Medications",
    "medication review": "Medication Review",
    "plan": "Plan",
    "prescription": "Prescription",
    "section": "Section",
    "symptom": "Symptoms",
    "symptoms": "Symptoms",
    "treatment plan": "Treatment Plan",
}

LABELS = (
    "Assessment",
    "Chief Complaint",
    "Date",
    "Date of Service",
    "Diagnosis",
    "Diagnoses",
    "Discharge Date",
    "Follow-up Date",
    "Hospital",
    "ICD",
    "ICD-10",
    "Medication",
    "Medications",
    "Physician",
    "Plan",
    "Provider",
    "Section",
    "Symptoms",
    "Visit Date",
)

NEXT_LABEL_LOOKAHEAD = rf"(?=\s+(?:{'|'.join(re.escape(label) for label in LABELS)})\s*:|[;\n]|$)"
DATE_VALUE_PATTERN = (
    r"(?:\d{4}-\d{1,2}-\d{1,2})|"
    r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|"
    r"(?:[A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
)
ICD_CODE_PATTERN = re.compile(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b")
PHYSICIAN_PATTERN = re.compile(r"\bDr\.[ \t]+[A-Z][A-Za-z.'-]+(?:[ \t]+[A-Z][A-Za-z.'-]+){0,3}\b")
HOSPITAL_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,5}\s+"
    r"(?:Hospital|Medical Center|Clinic|Regional|Health)\b"
)
MILITARY_FACILITY_PATTERN = re.compile(
    r"\bU\.?\s*S\.?\s*S\.?\s+[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4}"
    r"(?:\s*\([A-Z0-9 -]+\))?",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ClinicalDate:
    label: str
    value: str


@dataclass(frozen=True)
class DocumentSection:
    section: str
    order: int
    start_offset: int
    end_offset: int
    char_count: int


@dataclass(frozen=True)
class ClinicalMetadataResult:
    diagnoses: list[str]
    medications: list[str]
    symptoms: list[str]
    icd_codes: list[str]
    physicians: list[str]
    hospitals: list[str]
    dates: list[ClinicalDate]
    document_sections: list[DocumentSection]

    @property
    def entity_counts(self) -> dict[str, int]:
        return {
            "diagnoses": len(self.diagnoses),
            "medications": len(self.medications),
            "symptoms": len(self.symptoms),
            "icd_codes": len(self.icd_codes),
            "physicians": len(self.physicians),
            "hospitals": len(self.hospitals),
            "dates": len(self.dates),
            "document_sections": len(self.document_sections),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "extractor": CLINICAL_METADATA_EXTRACTOR,
            "entity_counts": self.entity_counts,
            "diagnoses": self.diagnoses,
            "medications": self.medications,
            "symptoms": self.symptoms,
            "icd_codes": self.icd_codes,
            "physicians": self.physicians,
            "hospitals": self.hospitals,
            "dates": [asdict(date) for date in self.dates],
            "document_sections": [asdict(section) for section in self.document_sections],
        }


def extract_clinical_metadata(text: str) -> ClinicalMetadataResult:
    diagnoses = extract_diagnoses(text)
    medications = extract_medications(text)
    symptoms = extract_symptoms(text)
    icd_codes = extract_icd_codes(text)
    physicians = extract_physicians(text)
    hospitals = extract_hospitals(text)
    dates = extract_dates(text)
    document_sections = extract_document_sections(text)

    return ClinicalMetadataResult(
        diagnoses=diagnoses,
        medications=medications,
        symptoms=symptoms,
        icd_codes=icd_codes,
        physicians=physicians,
        hospitals=hospitals,
        dates=dates,
        document_sections=document_sections,
    )


def apply_clinical_metadata_to_document(
    document: Document,
    clinical_metadata: ClinicalMetadataResult,
) -> None:
    if not document.diagnosis and clinical_metadata.diagnoses:
        document.diagnosis = clinical_metadata.diagnoses[0]
    if not document.icd_codes and clinical_metadata.icd_codes:
        document.icd_codes = clinical_metadata.icd_codes
    if not document.physician and clinical_metadata.physicians:
        document.physician = clinical_metadata.physicians[0]
    if not document.hospital and clinical_metadata.hospitals:
        document.hospital = clinical_metadata.hospitals[0]


def extract_diagnoses(text: str) -> list[str]:
    values: list[str] = []
    for labeled_value in extract_labeled_values(text, ("Diagnosis", "Diagnoses")):
        values.extend(match_terms(labeled_value, DIAGNOSIS_TERMS))
    values.extend(match_terms(text, DIAGNOSIS_TERMS))
    return ordered_unique(values)


def extract_medications(text: str) -> list[str]:
    labeled_medications = [
        clean_clinical_value(value)
        for value in extract_labeled_values(text, ("Medication", "Medications", "Prescription"))
    ]
    medications = [value for value in labeled_medications if value]
    for medication in match_terms(text, MEDICATION_TERMS):
        if not any(contains_term(existing, medication) for existing in medications):
            medications.append(medication)
    return ordered_unique(medications)


def extract_symptoms(text: str) -> list[str]:
    values: list[str] = []
    for labeled_value in extract_labeled_values(text, ("Chief Complaint", "Symptoms", "Symptom")):
        values.extend(match_terms(labeled_value, SYMPTOM_TERMS))
        cleaned = clean_symptom_value(labeled_value)
        if cleaned and not match_terms(cleaned, SYMPTOM_TERMS):
            values.append(cleaned)
    values.extend(match_terms(text, SYMPTOM_TERMS))
    return ordered_unique(values)


def extract_icd_codes(text: str) -> list[str]:
    return ordered_unique(match.group(0).upper() for match in ICD_CODE_PATTERN.finditer(text))


def extract_physicians(text: str) -> list[str]:
    values = [
        clean_clinical_value(value)
        for value in extract_labeled_values(text, ("Physician", "Provider"))
    ]
    values.extend(match.group(0) for match in PHYSICIAN_PATTERN.finditer(text))
    return ordered_unique(value for value in values if value)


def extract_hospitals(text: str) -> list[str]:
    values = [clean_clinical_value(value) for value in extract_labeled_values(text, ("Hospital",))]
    for hospital in KNOWN_HOSPITALS:
        if contains_term(text, hospital):
            values.append(hospital)
    values.extend(match.group(0) for match in HOSPITAL_PATTERN.finditer(text))
    values.extend(
        normalize_military_facility(match.group(0))
        for match in MILITARY_FACILITY_PATTERN.finditer(text)
    )
    return ordered_unique(value for value in values if value)


def normalize_military_facility(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = re.sub(
        r"\bU\.?\s*S\.?\s*S\.?\.?\b",
        "U.S.S.",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def extract_dates(text: str) -> list[ClinicalDate]:
    dates: list[ClinicalDate] = []
    seen: set[tuple[str, str]] = set()
    labeled_pattern = re.compile(
        rf"\b(?P<label>Date|Date of Service|Visit Date|Discharge Date|Follow-up Date)\s*:\s*"
        rf"(?P<value>{DATE_VALUE_PATTERN})",
        flags=re.IGNORECASE,
    )
    for match in labeled_pattern.finditer(text):
        label = canonical_date_label(match.group("label"))
        value = match.group("value")
        key = (label, value)
        if key not in seen:
            dates.append(ClinicalDate(label=label, value=value))
            seen.add(key)

    generic_pattern = re.compile(DATE_VALUE_PATTERN)
    labeled_spans = [match.span("value") for match in labeled_pattern.finditer(text)]
    for match in generic_pattern.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in labeled_spans):
            continue
        if has_birthdate_context(text, match.start()):
            continue
        value = match.group(0)
        key = ("date", value)
        if key not in seen:
            dates.append(ClinicalDate(label="date", value=value))
            seen.add(key)
    return dates


def extract_document_sections(text: str) -> list[DocumentSection]:
    heading_matches: list[tuple[re.Match[str], str]] = []
    for match in re.finditer(r"(?m)^(?P<heading>[A-Za-z][A-Za-z0-9 /-]{1,40}):", text):
        heading_key = normalize_heading_key(match.group("heading"))
        canonical = SECTION_ALIASES.get(heading_key)
        if canonical:
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            inline_value = text[match.end() : line_end].strip()
            if canonical == "Section" and inline_value:
                canonical = canonical_section_name(inline_value)
            heading_matches.append((match, canonical))

    sections: list[DocumentSection] = []
    for index, (match, canonical) in enumerate(heading_matches):
        start_offset = match.start()
        end_offset = (
            heading_matches[index + 1][0].start() if index + 1 < len(heading_matches) else len(text)
        )
        sections.append(
            DocumentSection(
                section=canonical,
                order=index,
                start_offset=start_offset,
                end_offset=end_offset,
                char_count=end_offset - start_offset,
            )
        )
    return sections


def extract_labeled_values(text: str, labels: tuple[str, ...]) -> list[str]:
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"\b(?:{label_pattern})\s*:\s*(?P<value>.+?){NEXT_LABEL_LOOKAHEAD}",
        flags=re.IGNORECASE,
    )
    return [clean_clinical_value(match.group("value")) for match in pattern.finditer(text)]


def match_terms(text: str, terms: dict[str, str]) -> list[str]:
    matches: list[str] = []
    for term, canonical in terms.items():
        if contains_term(text, term):
            matches.append(canonical)
    return ordered_unique(matches)


def contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.search(rf"(?<!\w){escaped}(?!\w)", text, flags=re.IGNORECASE) is not None


def clean_clinical_value(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" \t\r\n:-;,.")
    return cleaned


def clean_symptom_value(value: str) -> str:
    cleaned = clean_clinical_value(value)
    cleaned = re.sub(r"^patient\s+(?:reports|notes|describes)\s+", "", cleaned, flags=re.I)
    return clean_clinical_value(cleaned)


def canonical_date_label(label: str) -> str:
    return normalize_heading_key(label).replace(" ", "_").replace("-", "_")


def has_birthdate_context(text: str, start_offset: int) -> bool:
    context = text[max(0, start_offset - 32) : start_offset].lower()
    return "date of birth" in context or "dob" in context


def canonical_section_name(value: str) -> str:
    cleaned = clean_clinical_value(value)
    return SECTION_ALIASES.get(normalize_heading_key(cleaned), cleaned or "Section")


def normalize_heading_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace(".", "")).lower().strip()


def ordered_unique(values) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for raw_value in values:
        value = str(raw_value).strip()
        key = value.lower()
        if not value or key in seen:
            continue
        unique_values.append(value)
        seen.add(key)
    return unique_values
