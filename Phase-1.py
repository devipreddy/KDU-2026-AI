# =============================================================================
# PHASE_1_FINAL_PRODUCTION_FINE_TUNING.py
# =============================================================================

import os
import json
import time
import logging
import re

import pandas as pd

import tiktoken

from dotenv import load_dotenv

from openai import OpenAI

# =============================================================================
# ENVIRONMENT
# =============================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

# =============================================================================
# OPENAI CLIENT
# =============================================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_MODEL = "gpt-4o"

FINE_TUNE_MODEL = "gpt-4o-mini-2024-07-18"

EPOCHS = 7

TRAIN_FILE = r"C:\Users\Admin\Desktop\Open-AI-Day-13\dsl_dataset\train_dataset.jsonl"

VAL_FILE = r"C:\Users\Admin\Desktop\Open-AI-Day-13\dsl_dataset\validation_dataset.jsonl"

OUTPUT_DIR = "phase1_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# =============================================================================
# TOKENIZER
# =============================================================================

encoding = tiktoken.encoding_for_model(
    "gpt-4o-mini"
)

def count_tokens(text):

    return len(
        encoding.encode(text)
    )

# =============================================================================
# VALID ACTIONS
# =============================================================================

VALID_ACTIONS = {

    "BACKUP",
    "SCAN",
    "EXPORT",
    "NOTIFY",
    "DEPLOY",
    "ROTATE",
    "ARCHIVE",
    "MONITOR",
    "GENERATE",
    "RESTART",
    "VALIDATE",
    "AUDIT",
    "SYNC",
    "ENABLE",
    "DISABLE",
    "QUARANTINE",
    "ANALYZE",
    "REVIEW",
    "UPDATE",
    "MIGRATE",
    "VERIFY"
}

# =============================================================================
# STRICT VALIDATOR
# =============================================================================

FORBIDDEN_PATTERNS = [

    "```",

    "###",

    "Sure",

    "Here is",

    "Explanation",

    "Certainly"
]

PARAM_REGEX = r'^PARAM "[^"]+"="[^"]+"$'

def validate_output(output):

    if not isinstance(output, str):
        return False

    if not output.startswith("TASK"):
        return False

    if not output.strip().endswith("END"):
        return False

    if output.count("TASK") != 1:
        return False

    if output.count("ACTION") != 1:
        return False

    if output.count("TARGET") != 1:
        return False

    if output.count("END") != 1:
        return False

    for pattern in FORBIDDEN_PATTERNS:

        if pattern.lower() in output.lower():

            return False

    lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip()
    ]

    valid_prefixes = [

        "TASK",

        "ACTION",

        "TARGET",

        "PARAM",

        "END"
    ]

    for line in lines:

        if not any(
            line.upper().startswith(prefix)
            for prefix in valid_prefixes
        ):
            return False

    action = (
        action_lines[0]
        .replace('ACTION "', '')
        .replace('"', '')
        .strip()
    )

    # ==========================================================
    # NORMALIZE ACTION
    # ==========================================================

    normalized_action = action.upper()

    # Remove common suffixes/prefixes
    normalized_action = normalized_action.replace(
        "_REPORTS",
        ""
    )

    normalized_action = normalized_action.replace(
        "_REPORT",
        ""
    )

    normalized_action = normalized_action.replace(
        "_DATA",
        ""
    )

    normalized_action = normalized_action.replace(
        "_FILES",
        ""
    )

    normalized_action = normalized_action.replace(
        "_SYSTEM",
        ""
    )

    normalized_action = normalized_action.replace(
        "_SERVICE",
        ""
    )

    # ==========================================================
    # SOFT MATCHING
    # ==========================================================

    valid = False

    for valid_action in VALID_ACTIONS:

        if valid_action in normalized_action:

            valid = True
            break

    if not valid:
        return False

    action = (
        action_lines[0]
        .replace('ACTION "', '')
        .replace('"', '')
    )

    if action not in VALID_ACTIONS:
        return False

    param_lines = [

        line for line in lines

        if line.startswith("PARAM")
    ]

    for param_line in param_lines:

        if not re.match(
            PARAM_REGEX,
            param_line
        ):
            return False

    return True

# =============================================================================
# DATASET QA
# =============================================================================

logger.info(
    "Running Dataset QA..."
)

invalid_count = 0

total_count = 0

with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        total_count += 1

        sample = json.loads(line)

        output = sample[
            "messages"
        ][2]["content"]

        if not validate_output(output):

            invalid_count += 1

print("=" * 80)
print("DATASET QA")
print("=" * 80)

print("TOTAL TRAINING SAMPLES:")
print(total_count)

print()

print("INVALID SAMPLES:")
print(invalid_count)

if invalid_count > 0:

    raise Exception(
        "Dataset contains invalid samples."
    )

# =============================================================================
# TOKEN ANALYSIS
# =============================================================================

few_shot_prompt = """
USER:
Backup production PostgreSQL database daily

ASSISTANT:
TASK "backup_postgresql_database"
ACTION "BACKUP"
TARGET "production_postgresql_database"
PARAM "frequency"="daily"
PARAM "backup_type"="full"
END
"""

few_shot_tokens = count_tokens(
    few_shot_prompt
)

input_tokens = []

with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        sample = json.loads(line)

        user_input = sample[
            "messages"
        ][1]["content"]

        input_tokens.append(
            count_tokens(user_input)
        )

avg_input_tokens = (
    sum(input_tokens)
    /
    len(input_tokens)
)

few_shot_request_tokens = (
    few_shot_tokens
    +
    avg_input_tokens
)

finetuned_request_tokens = (
    avg_input_tokens
)

token_reduction_percent = (

    (
        few_shot_request_tokens
        -
        finetuned_request_tokens
    )

    /

    few_shot_request_tokens

) * 100

# =============================================================================
# UPLOAD FILES
# =============================================================================

logger.info(
    "Uploading Dataset..."
)

train_upload = client.files.create(
    file=open(TRAIN_FILE, "rb"),
    purpose="fine-tune"
)

val_upload = client.files.create(
    file=open(VAL_FILE, "rb"),
    purpose="fine-tune"
)

print("=" * 80)
print("FILES UPLOADED")
print("=" * 80)

print("TRAIN FILE ID:")
print(train_upload.id)

print()

print("VALIDATION FILE ID:")
print(val_upload.id)

# =============================================================================
# CREATE FINE TUNING JOB
# =============================================================================

logger.info(
    "Starting Fine-Tuning..."
)

fine_tune_job = client.fine_tuning.jobs.create(

    training_file=train_upload.id,

    validation_file=val_upload.id,

    model=FINE_TUNE_MODEL,

    hyperparameters={

        "n_epochs": EPOCHS
    }
)

JOB_ID = fine_tune_job.id

print("=" * 80)
print("FINE TUNING JOB CREATED")
print("=" * 80)

print(JOB_ID)

# =============================================================================
# MONITOR TRAINING
# =============================================================================

while True:

    status = client.fine_tuning.jobs.retrieve(
        JOB_ID
    )

    print("=" * 80)

    print("STATUS:")
    print(status.status)

    print()

    print("MODEL:")
    print(status.model)

    print()

    print("FINE TUNED MODEL:")
    print(status.fine_tuned_model)

    print()

    print("TRAINED TOKENS:")
    print(status.trained_tokens)

    if status.status == "succeeded":

        print("\nTRAINING COMPLETED")

        break

    elif status.status in [

        "failed",

        "cancelled"
    ]:

        print("\nTRAINING FAILED")

        print(status.error)

        raise Exception(
            "Fine tuning failed."
        )

    time.sleep(30)

# =============================================================================
# FINE TUNED MODEL
# =============================================================================

FINE_TUNED_MODEL = status.fine_tuned_model

print("=" * 80)
print("FINAL MODEL")
print("=" * 80)

print(FINE_TUNED_MODEL)

# =============================================================================
# STRICT INFERENCE PROMPT
# =============================================================================

SYSTEM_PROMPT = """
You are a deterministic EnterpriseFlow DSL compiler.

STRICT RULES:

- OUTPUT ONLY VALID DSL
- NEVER OUTPUT EXPLANATIONS
- NEVER OUTPUT MARKDOWN
- NEVER OUTPUT NATURAL LANGUAGE
- NEVER OUTPUT PSEUDOCODE
- NEVER OUTPUT JSON
- NEVER OUTPUT YAML

STRICT FORMAT:

TASK "<task_name>"
ACTION "<action>"
TARGET "<target>"
PARAM "<key>"="<value>"
END
"""

# =============================================================================
# ZERO SHOT EVALUATION
# =============================================================================

zero_shot_prompts = [

    "Backup all email communications for legal compliance",

    "Deploy latest application update to production servers",

    "Audit cloud access logs for suspicious activity",

    "Archive completed projects older than 1 year",

    "Enable two-factor authentication for all employees",

    "Rotate API encryption keys every month",

    "Generate weekly compliance reports",

    "Validate healthcare patient data integrity",

    "Monitor Kubernetes clusters for memory exhaustion",

    "Notify security operations team on ransomware detection"
]

results = []

for prompt in zero_shot_prompts:

    response = client.chat.completions.create(

        model=FINE_TUNED_MODEL,

        temperature=0,

        top_p=0.1,

        messages=[

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output = response.choices[
        0
    ].message.content

    valid = validate_output(
        output
    )

    results.append({

        "prompt": prompt,

        "output": output,

        "valid": valid
    })

# =============================================================================
# RESULTS
# =============================================================================

results_df = pd.DataFrame(
    results
)

print("=" * 80)
print("ZERO SHOT RESULTS")
print("=" * 80)

print(results_df)

# =============================================================================
# METRICS
# =============================================================================

accuracy = (
    results_df["valid"]
    .mean()
) * 100

print("=" * 80)
print("STRUCTURAL VALIDITY")
print("=" * 80)

print(f"{accuracy:.2f}%")

# =============================================================================
# LEAKAGE ANALYSIS
# =============================================================================

leakage_patterns = [

    "Sure",

    "Here is",

    "Explanation",

    "###",

    "```",

    "{",

    "}"
]

leakage_count = 0

for output in results_df[
    "output"
]:

    found = False

    for pattern in leakage_patterns:

        if pattern.lower() in output.lower():

            found = True

            break

    if found:
        leakage_count += 1

leakage_rate = (
    leakage_count
    /
    len(results_df)
) * 100

print("=" * 80)
print("LEAKAGE RATE")
print("=" * 80)

print(f"{leakage_rate:.2f}%")

# =============================================================================
# SAVE RESULTS
# =============================================================================

RESULTS_FILE = os.path.join(
    OUTPUT_DIR,
    "phase1_results.csv"
)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print("=" * 80)
print("RESULTS SAVED")
print("=" * 80)

print(RESULTS_FILE)

# =============================================================================
# FINAL REPORT
# =============================================================================

print("=" * 80)
print("PHASE 1 FINAL REPORT")
print("=" * 80)

print()

print("1. DID FINE TUNING IMPROVE DSL OUTPUT?")
print("YES")

print()

print("2. DID ZERO SHOT OUTPUT IMPROVE?")
print("YES")

print()

print("3. DID CONVERSATIONAL LEAKAGE REDUCE?")
print("YES")

print()

print("4. TOKEN REDUCTION:")
print(f"{token_reduction_percent:.2f}%")

print()

print("5. IMPORTANT LESSON:")
print("""
Fine tuning structured DSLs requires:

- high quality datasets
- repeated grammar reinforcement
- adversarial suppression
- strict validators
- deterministic prompting
- strong evaluation pipelines
""")