"""
Medical Entity Extraction Module using Google LangExtract
==========================================================

This module demonstrates how to extract structured medical entities
from clinical text using LangExtract's few-shot learning approach.

Extracts:
  - Medications (name, dosage, route, frequency)
  - Adverse drug reactions (severity, timing, related medication)
  - Diagnoses (type, status)
  - Vital signs (measurement, value, status)
  - Lab results (test, value, reference range, status)

Usage:
    python medical_extraction.py

Output:
    - output/medical_extractions.jsonl   (LangExtract native format)
    - output/medical_extractions.json    (structured JSON)
    - output/medical_visualization.html  (interactive HTML visualization)

References:
    - Google LangExtract: https://github.com/google/langextract
    - Medication Examples: https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md
    - RadExtract Demo: https://huggingface.co/spaces/google/radextract
"""

import langextract as lx
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Few-shot examples — teach the model what to extract
# ---------------------------------------------------------------------------
# Following Google's medication_examples.md pattern:
# https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md

MEDICAL_PROMPT = """Extract all medical entities from clinical text in order of appearance.
Extract the following entity types:
- medication: Drug names with dosage, route, and frequency as attributes
- adverse_reaction: Side effects or adverse drug reactions with severity and related medication
- diagnosis: Medical conditions or diagnoses with status (active, resolved, suspected)
- vital_sign: Vital sign measurements with value and status (normal, abnormal, critical)
- lab_result: Laboratory test results with value, reference range, and status
- procedure: Medical procedures or interventions performed
"""

MEDICAL_EXAMPLES = [
    # ── Example 1: Medication + adverse reaction ──────────────────────────
    lx.data.ExampleData(
        text=(
            "Patient was prescribed Metformin 500mg orally twice daily for "
            "type 2 diabetes. She reported mild nausea after the first dose. "
            "Blood pressure was 142/90 mmHg. HbA1c level was 8.2%, above the "
            "normal range of 4.0-5.6%."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Metformin 500mg",
                attributes={
                    "route": "oral",
                    "frequency": "twice daily",
                    "indication": "type 2 diabetes",
                },
            ),
            lx.data.Extraction(
                extraction_class="diagnosis",
                extraction_text="type 2 diabetes",
                attributes={"status": "active"},
            ),
            lx.data.Extraction(
                extraction_class="adverse_reaction",
                extraction_text="mild nausea",
                attributes={
                    "severity": "mild",
                    "timing": "after the first dose",
                    "related_medication": "Metformin",
                },
            ),
            lx.data.Extraction(
                extraction_class="vital_sign",
                extraction_text="142/90 mmHg",
                attributes={
                    "measurement": "blood pressure",
                    "status": "abnormal",
                },
            ),
            lx.data.Extraction(
                extraction_class="lab_result",
                extraction_text="8.2%",
                attributes={
                    "test": "HbA1c",
                    "reference_range": "4.0-5.6%",
                    "status": "abnormal",
                },
            ),
        ],
    ),
    # ── Example 2: Multiple medications + procedure ───────────────────────
    lx.data.ExampleData(
        text=(
            "Post-operative day 1 following laparoscopic cholecystectomy. "
            "Patient received Cefazolin 1g IV every 8 hours for infection "
            "prophylaxis and Morphine 2mg IV PRN for pain management. "
            "Temperature was 37.2°C, within normal limits. "
            "WBC count was 11,200/µL, slightly elevated above the "
            "reference range of 4,500-11,000/µL."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="procedure",
                extraction_text="laparoscopic cholecystectomy",
                attributes={"timing": "post-operative day 1"},
            ),
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Cefazolin 1g",
                attributes={
                    "route": "IV",
                    "frequency": "every 8 hours",
                    "indication": "infection prophylaxis",
                },
            ),
            lx.data.Extraction(
                extraction_class="medication",
                extraction_text="Morphine 2mg",
                attributes={
                    "route": "IV",
                    "frequency": "PRN",
                    "indication": "pain management",
                },
            ),
            lx.data.Extraction(
                extraction_class="vital_sign",
                extraction_text="37.2°C",
                attributes={
                    "measurement": "temperature",
                    "status": "normal",
                },
            ),
            lx.data.Extraction(
                extraction_class="lab_result",
                extraction_text="11,200/µL",
                attributes={
                    "test": "WBC count",
                    "reference_range": "4,500-11,000/µL",
                    "status": "abnormal",
                },
            ),
        ],
    ),
]


# ---------------------------------------------------------------------------
# Sample clinical notes to extract from
# ---------------------------------------------------------------------------
CLINICAL_NOTES = [
    {
        "id": "note_001",
        "title": "Emergency Department Visit — Chest Pain",
        "text": (
            "A 58-year-old male presented to the emergency department with "
            "acute chest pain radiating to the left arm for the past 2 hours. "
            "History of hypertension and hyperlipidemia. Current medications "
            "include Lisinopril 10mg orally once daily and Atorvastatin 40mg "
            "orally at bedtime. Vital signs: heart rate 98 bpm, blood pressure "
            "168/95 mmHg, SpO2 96% on room air, temperature 36.8°C. "
            "Troponin I level was 0.08 ng/mL, above the normal threshold of "
            "0.04 ng/mL. ECG showed ST-segment elevation in leads II, III, "
            "and aVF. Patient was started on Aspirin 325mg orally stat, "
            "Heparin 5000 units IV bolus, and Nitroglycerin 0.4mg sublingual "
            "PRN for chest pain. Emergent cardiac catheterization was performed "
            "revealing 90% stenosis of the right coronary artery. "
            "Percutaneous coronary intervention with drug-eluting stent "
            "placement was successfully completed."
        ),
    },
    {
        "id": "note_002",
        "title": "Pediatric Follow-up — Asthma Management",
        "text": (
            "An 8-year-old female presented for routine asthma follow-up. "
            "Mother reports increased nighttime coughing over the past two "
            "weeks and use of rescue inhaler 4 times per week. Currently on "
            "Fluticasone 44mcg inhaled twice daily and Albuterol 90mcg "
            "inhaled PRN. Peak flow was 280 L/min, which is 75% of her "
            "predicted value and below the green zone threshold of 80%. "
            "Lungs showed mild bilateral wheezing on auscultation. "
            "SpO2 was 97% on room air, within normal limits. "
            "Assessment: poorly controlled persistent asthma. "
            "Plan: Step up Fluticasone to 110mcg inhaled twice daily, "
            "continue Albuterol PRN, and add Montelukast 5mg orally at "
            "bedtime. Follow-up in 4 weeks. Patient's mother reported "
            "mild oral thrush possibly related to Fluticasone use."
        ),
    },
    {
        "id": "note_003",
        "title": "ICU Progress Note — Sepsis",
        "text": (
            "Day 3 in ICU for severe sepsis secondary to urinary tract "
            "infection with E. coli bacteremia. Patient is a 72-year-old "
            "female with history of type 2 diabetes and chronic kidney "
            "disease stage 3. Current antibiotics: Meropenem 1g IV every "
            "8 hours, adjusted for renal function. Vasopressor support with "
            "Norepinephrine 0.15 mcg/kg/min IV continuous infusion. "
            "Vital signs: temperature 38.4°C, heart rate 112 bpm, blood "
            "pressure 92/58 mmHg on vasopressor support, respiratory rate "
            "22 breaths/min, SpO2 94% on 4L nasal cannula. Labs: WBC "
            "18,500/µL above reference range 4,500-11,000/µL, lactate "
            "3.2 mmol/L above normal of less than 2.0 mmol/L, creatinine "
            "2.8 mg/dL above baseline of 1.5 mg/dL, procalcitonin 12.5 "
            "ng/mL above threshold of 0.5 ng/mL. Blood cultures from day 1 "
            "growing E. coli sensitive to Meropenem. Insulin glargine 20 "
            "units subcutaneous at bedtime for diabetes management. Patient "
            "developed mild thrombocytopenia with platelet count of "
            "98,000/µL below reference range 150,000-400,000/µL."
        ),
    },
]


def run_extraction(
    notes: list[dict],
    model_id: str = "gemini-2.5-flash",
    api_key: str | None = None,
) -> list[dict]:
    """
    Run medical entity extraction on a list of clinical notes.

    Args:
        notes: List of dicts with 'id', 'title', and 'text' keys.
        model_id: LangExtract model identifier.
        api_key: API key (defaults to LANGEXTRACT_API_KEY env var).

    Returns:
        List of structured extraction results as dicts.
    """
    api_key = api_key or os.getenv("LANGEXTRACT_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Set LANGEXTRACT_API_KEY env var or pass api_key."
        )

    all_results = []

    for note in notes:
        print(f"\n{'='*70}")
        print(f"Processing: {note['title']}")
        print(f"{'='*70}")

        # ── Run LangExtract ──────────────────────────────────────────────
        result = lx.extract(
            text_or_documents=note["text"],
            prompt_description=MEDICAL_PROMPT,
            examples=MEDICAL_EXAMPLES,
            model_id=model_id,
            api_key=api_key,
            extraction_passes=2,    # Two passes for better recall
            max_workers=1,
        )

        # ── Collect extractions ──────────────────────────────────────────
        extractions = list(result.extractions)
        print(f"  Found {len(extractions)} entities\n")

        structured = {
            "note_id": note["id"],
            "title": note["title"],
            "model": model_id,
            "total_entities": len(extractions),
            "entities_by_class": {},
            "entities": [],
        }

        for ext in extractions:
            # Count by class
            cls = ext.extraction_class
            structured["entities_by_class"][cls] = (
                structured["entities_by_class"].get(cls, 0) + 1
            )

            entity = {
                "class": cls,
                "text": ext.extraction_text,
                "attributes": ext.attributes if ext.attributes else {},
            }
            structured["entities"].append(entity)

            # Pretty print
            attr_str = ""
            if ext.attributes:
                attr_str = " | " + ", ".join(
                    f"{k}={v}" for k, v in ext.attributes.items()
                )
            print(f"  [{cls:20s}] {ext.extraction_text}{attr_str}")

        all_results.append(structured)

        # ── Save JSONL via LangExtract native format ─────────────────────
        lx.io.save_annotated_documents(
            [result],
            output_dir=str(OUTPUT_DIR),
            output_name=f"{note['id']}_extractions.jsonl",
        )

        # ── Generate visualization ───────────────────────────────────────
        jsonl_path = OUTPUT_DIR / f"{note['id']}_extractions.jsonl"
        try:
            html = lx.visualize(str(jsonl_path))
            html_path = OUTPUT_DIR / f"{note['id']}_visualization.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"\n  Visualization saved: {html_path}")
        except Exception as e:
            print(f"\n  Visualization skipped: {e}")

    return all_results


def save_results(results: list[dict]) -> None:
    """Save combined results as a structured JSON file."""
    output_path = OUTPUT_DIR / "medical_extractions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}")
    print(f"Combined results saved: {output_path}")
    print(f"{'='*70}")


def print_summary(results: list[dict]) -> None:
    """Print a summary table of all extractions."""
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*70}")

    total_entities = 0
    global_counts: dict[str, int] = {}

    for r in results:
        total_entities += r["total_entities"]
        for cls, count in r["entities_by_class"].items():
            global_counts[cls] = global_counts.get(cls, 0) + count

    print(f"\n  Total clinical notes processed: {len(results)}")
    print(f"  Total entities extracted:       {total_entities}")
    print(f"\n  Entities by class:")
    for cls in sorted(global_counts, key=global_counts.get, reverse=True):
        print(f"    {cls:20s}  {global_counts[cls]:>4d}")

    print(f"\n  Output files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        print(f"    {f.name:45s}  {size:>8,d} bytes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        Medical Entity Extraction using Google LangExtract          ║
║                                                                    ║
║  Extracts medications, adverse reactions, diagnoses, vital signs,  ║
║  lab results, and procedures from clinical text.                   ║
║                                                                    ║
║  GitHub:  https://github.com/google/langextract                    ║
║  Demo:    https://huggingface.co/spaces/google/radextract          ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    results = run_extraction(CLINICAL_NOTES)
    save_results(results)
    print_summary(results)
