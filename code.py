import json
import time
import traceback
import pandas as pd

from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, HttpOptions, Tool


# ============================================================
# CONFIG
# ============================================================

PROJECT_ID = "labs-491605"
LOCATION = "global"

INPUT_FILE = "ADC Grid.xlsx"
OUTPUT_FILE = "ADC Grid_comp.xlsx"

MODEL_NAME = "gemini-2.5-flash"

# For testing only first 8 rows
MAX_ROWS_TO_PROCESS = 8

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

COLUMNS = [
    "Tumor Target Type",
    "Target Antigen",
    "ADC Component",
    "Isotype",
    "Payload",
    "Payload MoA",
    "Bystander Effect",
    "DAR",
    "Linker",
    "Linker Type",
    "Linker Stability",
    "Conjugation Type",
    "COGS per dose (bucket)",
    "Comment (on COGS per dose)",
    "Pricing",
    "Commentary",
    "Source",
]


# ============================================================
# NORMALISATION RULES
# ============================================================

NORMALIZATION_RULES = {
    "Tumor Target Type": [
        "Solid tumor",
        "Hematologic malignancy",
        "Both",
        "Unknown"
    ],

    "ADC Component": [
        "Monoclonal antibody",
        "Antibody-linker-payload",
        "Bispecific antibody-drug conjugate",
        "Unknown"
    ],

    "Isotype": [
        "IgG1",
        "IgG2",
        "IgG4",
        "IgG1-kappa",
        "IgG1-lambda",
        "Unknown"
    ],

    "Payload MoA": [
        "Microtubule inhibitor",
        "Topoisomerase I inhibitor",
        "DNA-damaging agent",
        "RNA polymerase II inhibitor",
        "Immunostimulatory payload",
        "Unknown"
    ],

    "Bystander Effect": [
        "Yes",
        "No",
        "Unknown"
    ],

    "DAR": [
        "2",
        "4",
        "6",
        "8",
        "Variable",
        "Unknown"
    ],

    "Linker Type": [
        "Cleavable",
        "Non-cleavable",
        "Unknown"
    ],

    "Linker Stability": [
        "Stable",
        "Moderately stable",
        "Unstable",
        "Unknown"
    ],

    "Conjugation Type": [
        "Lysine conjugation",
        "Cysteine conjugation",
        "Site-specific conjugation",
        "Engineered cysteine conjugation",
        "Unknown"
    ],

    "COGS per dose (bucket)": [
        "Low",
        "Medium",
        "High",
        "Unknown"
    ]
}


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1"),
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Converts every value to a safe Excel-compatible string.
    """
    if value is None:
        return "Unknown"

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)

    value = str(value).strip()

    if value == "":
        return "Unknown"

    return value


def extract_json(text):
    """
    Extracts JSON object from Gemini response text.

    Gemini should return only JSON, but sometimes it may include markdown
    or extra text. This function tries to safely extract the JSON object.
    """
    if text is None or str(text).strip() == "":
        raise ValueError("Empty Gemini response")

    raw = str(text).strip()

    # Remove markdown code fences
    raw = raw.replace("```json", "")
    raw = raw.replace("```JSON", "")
    raw = raw.replace("```", "")
    raw = raw.strip()

    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No valid JSON object found in Gemini response: {raw[:500]}")

    json_text = raw[start:end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by Gemini: {e}. Raw text: {json_text[:500]}")


def validate_required_keys(data):
    """
    Ensures all required columns exist in Gemini's returned JSON.
    Missing keys are filled as Unknown.
    """
    if not isinstance(data, dict):
        raise ValueError("Gemini response JSON is not a dictionary/object")

    for col in COLUMNS:
        if col not in data:
            data[col] = "Unknown"

    return data


def validate_normalized_values(data):
    """
    Ensures normalized columns contain only allowed values.

    If Gemini returns anything outside the allowed list,
    replace it with Unknown.
    """
    for col, allowed_values in NORMALIZATION_RULES.items():
        value = clean_value(data.get(col, "Unknown"))

        if value not in allowed_values:
            print(
                f"Normalization warning: column '{col}' had invalid value "
                f"'{value}'. Changed to 'Unknown'."
            )
            data[col] = "Unknown"

    return data


def build_prompt(asset):
    """
    Builds the Gemini prompt for one asset/regimen.
    """
    prompt = f"""
You are filling an oncology antibody-drug conjugate ADC spreadsheet using reliable public data.

Asset / Regimen: {asset}

Use Google Search grounding to find reliable public information.

Return ONLY one valid JSON object.
Do not use markdown.
Do not return arrays.
Do not return nested objects.
Every value must be a plain string.

Required JSON keys:
{json.dumps(COLUMNS, indent=2)}

Normalization rules:
{json.dumps(NORMALIZATION_RULES, indent=2)}

For every column listed in Normalization rules:
- Return only one value from the allowed list.
- Do not create new values.
- Do not return detailed phrases in normalized columns.
- Map detailed scientific/public-source descriptions to the closest allowed value.
- If the public data does not support a reliable value, return "Unknown".

For columns not listed in Normalization rules:
- Return short, evidence-based plain text.
- Do not return arrays.
- Do not return nested objects.
- If unknown, return "Unknown".

Specific mapping guidance:

For "DAR":
- If DAR is reported as approximately 2, average 2, or about 2, return "2".
- If DAR is reported as approximately 4, average 4, or about 4, return "4".
- If DAR is reported as approximately 6, average 6, or about 6, return "6".
- If DAR is reported as approximately 8, average 8, or about 8, return "8".
- If DAR is heterogeneous, range-based, not fixed, or varies by molecule, return "Variable".
- If not found, return "Unknown".

For "Isotype":
- humanized IgG1, human IgG1, IgG1 monoclonal antibody -> "IgG1"
- IgG1 kappa, IgG1κ, IgG1-kappa -> "IgG1-kappa"
- IgG1 lambda, IgG1λ, IgG1-lambda -> "IgG1-lambda"
- If not found, return "Unknown".

For "Payload MoA":
- MMAE, MMAF, auristatin, maytansinoid, DM1, DM4 -> "Microtubule inhibitor"
- Deruxtecan, DXd, SN-38, exatecan, camptothecin derivatives -> "Topoisomerase I inhibitor"
- PBD, duocarmycin, calicheamicin -> "DNA-damaging agent"
- Amanitin -> "RNA polymerase II inhibitor"
- STING agonist, TLR agonist, immune agonist payload -> "Immunostimulatory payload"
- If not found, return "Unknown".

For "Linker Type":
- acid-labile, acid-cleavable, hydrazone, hydrolyzable, pH-sensitive, peptide-based, protease-cleavable, enzymatically cleavable -> "Cleavable"
- thioether, MCC, SMCC, non-cleavable -> "Non-cleavable"
- If not found, return "Unknown".

For "Bystander Effect":
- If payload is membrane-permeable or public sources mention bystander killing/effect, return "Yes".
- If source says no/minimal bystander effect or payload is not membrane-permeable, return "No".
- If unclear, return "Unknown".

For "Tumor Target Type":
- Breast, lung, gastric, ovarian, urothelial, prostate, colorectal, pancreatic, or other carcinoma/sarcoma -> "Solid tumor"
- Leukemia, lymphoma, myeloma -> "Hematologic malignancy"
- If used across both solid tumors and hematologic malignancies -> "Both"
- If not found, return "Unknown".

For "COGS per dose (bucket)":
- Low: simpler/older ADC design or lower manufacturing complexity.
- Medium: typical ADC manufacturing complexity.
- High: complex ADC design, premium payload/linker, high DAR, site-specific conjugation, or difficult biologic manufacturing.
- If there is no reliable basis, return "Unknown".

Column-specific output rules:
- "Target Antigen" should be a short antigen name, for example HER2, TROP2, Nectin-4, CD30, CD33, BCMA, FRα, etc. If unknown, return "Unknown".
- "Payload" should be the payload name, for example MMAE, DM1, DXd, SN-38, PBD dimer, etc. If unknown, return "Unknown".
- "Linker" may contain detailed terms such as acid-labile hydrazone, peptide linker, thioether, MCC, SMCC, etc.
- "Linker Type" must only be Cleavable, Non-cleavable, or Unknown.
- "Pricing" should include public pricing information if available, otherwise "Unknown".
- "Commentary" should be short and evidence-based.
- "Source" must be a plain string containing 1-3 public URLs separated by commas.

General rules:
- If unknown, use "Unknown".
- Do not guess if public data is not reliable.
- Do not add extra text outside the JSON.
"""
    return prompt


def call_gemini_with_retries(asset):
    """
    Calls Gemini with retry handling.

    If Gemini/API/JSON parsing fails, it retries up to MAX_RETRIES times.
    """
    prompt = build_prompt(asset)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"Gemini attempt {attempt}/{MAX_RETRIES} for: {asset}")

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[Tool(google_search=GoogleSearch())],
                ),
            )

            response_text = getattr(response, "text", None)

            data = extract_json(response_text)
            data = validate_required_keys(data)
            data = validate_normalized_values(data)

            return data

        except Exception as e:
            last_error = e
            print(f"Attempt {attempt} failed for {asset}: {e}")

            if attempt < MAX_RETRIES:
                wait_time = RETRY_DELAY_SECONDS * attempt
                print(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)

    raise RuntimeError(f"All retries failed for {asset}. Last error: {last_error}")


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    print("Reading Excel file...")

    try:
        df = pd.read_excel(INPUT_FILE)
    except FileNotFoundError:
        print(f"Input file not found: {INPUT_FILE}")
        return
    except Exception as e:
        print(f"Failed to read Excel file: {e}")
        return

    if "Regimen / Asset" not in df.columns:
        print("ERROR: Excel file must contain a column named 'Regimen / Asset'")
        return

    # Ensure all output columns exist
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Process only first 8 rows for testing
    rows_to_process = df.head(MAX_ROWS_TO_PROCESS)

    print(f"Processing first {MAX_ROWS_TO_PROCESS} rows...")

    for idx, row in rows_to_process.iterrows():
        asset = clean_value(row.get("Regimen / Asset", "Unknown"))

        if asset == "Unknown" or asset.lower() == "nan":
            print(f"Skipping empty row {idx + 1}")
            continue

        print("=" * 80)
        print(f"Processing row {idx + 1}: {asset}")

        try:
            data = call_gemini_with_retries(asset)

            for col in COLUMNS:
                df.loc[idx, col] = clean_value(data.get(col, "Unknown"))

            print(f"Successfully processed: {asset}")

        except Exception as e:
            error_message = str(e)
            print(f"ERROR on {asset}: {error_message}")
            print(traceback.format_exc())

            df.loc[idx, "Commentary"] = f"ERROR: {error_message}"
            df.loc[idx, "Source"] = "Unknown"

        # Save after each row so progress is not lost
        try:
            df.to_excel(OUTPUT_FILE, index=False)
            print(f"Progress saved to: {OUTPUT_FILE}")
        except Exception as e:
            print(f"Failed to save Excel file after row {idx + 1}: {e}")

        # Small delay between rows to avoid rate-limit issues
        time.sleep(2)

    print("=" * 80)
    print(f"Done. Saved output to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
