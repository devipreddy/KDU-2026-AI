# ========================================================================
# PRODUCTION_GRADE_DSL_DATASET_GENERATOR.py
# ========================================================================
#
# PURPOSE:
# Generate an enterprise-grade high-quality supervised fine-tuning dataset
# for strict deterministic DSL generation.
#
# THIS VERSION FIXES:
# - weak syntax reinforcement
# - conversational leakage
# - markdown leakage
# - pseudo-code drift
# - schema hallucination
# - DSL inconsistency
#
# THIS IS DESIGNED FOR:
# - OpenAI SFT
# - GPT-4o-mini fine-tuning
# - deterministic structured output alignment
#
# ========================================================================

import os
import json
import random
import hashlib
import logging
from datetime import datetime
from typing import Dict
from typing import List

from openai import OpenAI

# ========================================================================
# LOGGING
# ========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

# ========================================================================
# CONFIGURATION
# ========================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)
OUTPUT_DIR = "dsl_dataset"

TRAIN_FILE = os.path.join(
    OUTPUT_DIR,
    "train_dataset.jsonl"
)

VAL_FILE = os.path.join(
    OUTPUT_DIR,
    "validation_dataset.jsonl"
)

DATASET_METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "dataset_metadata.json"
)

TOTAL_EXAMPLES = 570

VALIDATION_SIZE = 100

SEED = 42

MODEL_NAME = "gpt-4o"

random.seed(SEED)

# ========================================================================
# OPENAI CLIENT
# ========================================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)

# ========================================================================
# OUTPUT DIRECTORY
# ========================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ========================================================================
# STRICT DSL SPEC
# ========================================================================

DSL_SPEC = """
You are a deterministic EnterpriseFlow DSL compiler.

STRICT RULES:

1. OUTPUT ONLY VALID DSL
2. NEVER OUTPUT EXPLANATIONS
3. NEVER OUTPUT MARKDOWN
4. NEVER OUTPUT PSEUDOCODE
5. NEVER OUTPUT FUNCTION CALLS
6. NEVER OUTPUT JSON
7. NEVER OUTPUT YAML
8. NEVER OUTPUT COMMENTS
9. NEVER OUTPUT NATURAL LANGUAGE
10. STRICTLY FOLLOW DSL FORMAT

VALID DSL FORMAT:

TASK "<task_name>"
ACTION "<action>"
TARGET "<target>"
PARAM "<key>"="<value>"
PARAM "<key>"="<value>"
END

MANDATORY:
- TASK required
- ACTION required
- TARGET required
- END required

ALL KEYWORDS MUST BE UPPERCASE.

ALL VALUES MUST USE DOUBLE QUOTES.

VALID ACTIONS:

BACKUP
SCAN
EXPORT
NOTIFY
DEPLOY
ROTATE
ARCHIVE
MONITOR
GENERATE
RESTART
VALIDATE
AUDIT
SYNC
ENABLE
DISABLE
QUARANTINE
ANALYZE
REVIEW
UPDATE
MIGRATE
VERIFY

INVALID OUTPUT EXAMPLES:

archive_logs(days=90)

{
  "action": "backup"
}

### Report

Sure, here is the DSL:

The correct answer is:

ONLY OUTPUT VALID DSL.
"""

# ========================================================================
# HIGH QUALITY FEW SHOT EXAMPLES
# ========================================================================

FEW_SHOTS = """
USER:
Backup production PostgreSQL database daily

ASSISTANT:
TASK "backup_postgresql_database"
ACTION "BACKUP"
TARGET "production_postgresql_database"
PARAM "frequency"="daily"
PARAM "backup_type"="full"
END

USER:
Notify SOC team if credential dumping is detected

ASSISTANT:
TASK "notify_soc_credential_dumping"
ACTION "NOTIFY"
TARGET "soc_team"
PARAM "threat"="credential_dumping"
PARAM "severity"="high"
END

USER:
Archive logs older than 365 days

ASSISTANT:
TASK "archive_old_logs"
ACTION "ARCHIVE"
TARGET "system_logs"
PARAM "older_than_days"="365"
END

USER:
Rotate API encryption keys monthly

ASSISTANT:
TASK "rotate_api_keys"
ACTION "ROTATE"
TARGET "api_encryption_keys"
PARAM "frequency"="monthly"
END

USER:
Scan employee uploads for ransomware

ASSISTANT:
TASK "scan_employee_uploads"
ACTION "SCAN"
TARGET "employee_uploads"
PARAM "scan_type"="ransomware"
END

USER:
Deploy fraud detection service to production

ASSISTANT:
TASK "deploy_fraud_detection_service"
ACTION "DEPLOY"
TARGET "fraud_detection_service"
PARAM "environment"="production"
END

USER:
Validate healthcare patient data integrity

ASSISTANT:
TASK "validate_patient_data"
ACTION "VALIDATE"
TARGET "healthcare_patient_data"
PARAM "validation_type"="integrity"
END

USER:
Generate weekly compliance report

ASSISTANT:
TASK "generate_compliance_report"
ACTION "GENERATE"
TARGET "compliance_report"
PARAM "frequency"="weekly"
END

USER:
Enable two-factor authentication for all users

ASSISTANT:
TASK "enable_two_factor_authentication"
ACTION "ENABLE"
TARGET "user_accounts"
PARAM "authentication_type"="two_factor"
END

USER:
Audit cloud access logs for suspicious activity

ASSISTANT:
TASK "audit_cloud_access_logs"
ACTION "AUDIT"
TARGET "cloud_access_logs"
PARAM "activity_type"="suspicious"
END
"""

# ========================================================================
# ADVERSARIAL TRAINING EXAMPLES
# ========================================================================

"""
THESE ARE CRITICAL.

THEY TEACH:
- markdown suppression
- conversational suppression
- pseudo-code suppression
- invalid syntax rejection
"""

ADVERSARIAL_GUIDANCE = """
VERY IMPORTANT:

NEVER GENERATE:

- markdown
- bullet points
- explanations
- pseudocode
- JSON
- YAML
- comments
- natural language
- helper text

BAD OUTPUTS:

### Deployment

Sure! Here is the DSL:

deploy(app="fraud")

{
  "task": "backup"
}

CORRECT OUTPUTS MUST ALWAYS FOLLOW:

TASK ...
ACTION ...
TARGET ...
PARAM ...
END
"""

# ========================================================================
# DATASET GENERATION PROMPT
# ========================================================================

GENERATION_PROMPT = f"""
{DSL_SPEC}

{ADVERSARIAL_GUIDANCE}

You are generating a PRODUCTION-GRADE dataset
for supervised fine-tuning.

Generate EXTREMELY HIGH QUALITY examples.

CRITICAL REQUIREMENTS:

1. STRICT DSL ONLY
2. NO CONVERSATIONAL OUTPUT
3. NO MARKDOWN
4. NO PSEUDOCODE
5. NO JSON
6. PERFECT DSL FORMAT
7. HIGH VARIETY
8. ENTERPRISE SCENARIOS
9. DETERMINISTIC STRUCTURE
10. CONSISTENT SYNTAX

DOMAINS TO COVER:

- cybersecurity
- healthcare
- finance
- cloud infrastructure
- kubernetes
- DevOps
- observability
- networking
- compliance
- incident response
- SIEM
- monitoring
- IAM
- data governance
- SOC workflows
- enterprise automation

VERY IMPORTANT:

TASK NAMES MUST:
- use snake_case
- be deterministic
- reflect the user intent

RETURN STRICT JSON ONLY.

FORMAT:

[
  {{
    "input": "...",
    "output": "TASK ..."
  }}
]

Generate 100 examples.
"""

# ========================================================================
# VALIDATION
# ========================================================================

REQUIRED_KEYWORDS = [
    "TASK",
    "ACTION",
    "TARGET",
    "END"
]

FORBIDDEN_PATTERNS = [
    "```",
    "###",
    "{",
    "}",
    "(",
    ")",
    "Sure",
    "Here is",
    "Explanation"
]

def validate_output(output: str):

    if not isinstance(output, str):
        return False

    for keyword in REQUIRED_KEYWORDS:

        if keyword not in output:
            return False

    for pattern in FORBIDDEN_PATTERNS:

        if pattern in output:
            return False

    if not output.startswith("TASK"):
        return False

    if not output.strip().endswith("END"):
        return False

    lines = output.splitlines()

    for line in lines:

        line = line.strip()

        if line == "":
            continue

        valid_prefixes = [
            "TASK",
            "ACTION",
            "TARGET",
            "PARAM",
            "END"
        ]

        if not any(
            line.startswith(prefix)
            for prefix in valid_prefixes
        ):
            return False
        
        if output.count("TASK") != 1:
            return False

        if output.count("ACTION") != 1:
            return False

        if output.count("TARGET") != 1:
            return False

        if output.count("END") != 1:
            return False

    return True

# ========================================================================
# CHAT FORMAT CONVERSION
# ========================================================================

def convert_to_chat_format(
    input_text,
    output_text
):

    return {

        "messages": [

            {
                "role": "system",
                "content": DSL_SPEC
            },

            {
                "role": "user",
                "content": input_text
            },

            {
                "role": "assistant",
                "content": output_text
            }
        ]
    }

# ========================================================================
# GENERATE DATASET BATCH
# ========================================================================

def generate_batch():

    response = client.chat.completions.create(

        model=MODEL_NAME,

        temperature=0.3,

        response_format={
            "type": "json_object"
        },

        messages=[

            {
                "role": "system",
                "content": GENERATION_PROMPT
            },

            {
                "role": "user",
                "content": """
Generate examples using STRICT FORMAT:

{
  "examples": [
    {
      "input": "...",
      "output": "TASK ..."
    }
  ]
}

RETURN VALID JSON ONLY.
"""
            }
        ]
    )

    content = response.choices[
        0
    ].message.content

    parsed = json.loads(content)

    if "examples" not in parsed:

        raise Exception(
            "Missing 'examples' key in response."
        )

    examples = parsed["examples"]

    if not isinstance(examples, list):

        raise Exception(
            "'examples' is not a list."
        )

    validated_examples = []

    for item in examples:

        if not isinstance(item, dict):
            continue

        if "input" not in item:
            continue

        if "output" not in item:
            continue

        validated_examples.append(item)

    return validated_examples
# ========================================================================
# DATASET GENERATION LOOP
# ========================================================================

logger.info(
    "Generating Production Dataset..."
)

dataset = []

unique_inputs = set()

while len(dataset) < TOTAL_EXAMPLES:

    try:

        batch = generate_batch()

        for item in batch:

            if len(dataset) >= TOTAL_EXAMPLES:
                break

            input_text = item[
                "input"
            ].strip()

            output_text = item[
                "output"
            ].strip()

            if input_text in unique_inputs:
                continue

            if not validate_output(
                output_text
            ):
                continue

            unique_inputs.add(
                input_text
            )

            dataset.append({

                "input": input_text,

                "output": output_text
            })

        logger.info(
            f"Dataset Size: {len(dataset)}"
        )

    except Exception as e:

        logger.error(str(e))

# ========================================================================
# SHUFFLE DATASET
# ========================================================================

random.shuffle(dataset)

# ========================================================================
# TRAIN / VALIDATION SPLIT
# ========================================================================

validation_dataset = dataset[
    :VALIDATION_SIZE
]

training_dataset = dataset[
    VALIDATION_SIZE:
]

# ========================================================================
# CONVERT TO CHAT FORMAT
# ========================================================================

training_chat_dataset = [

    convert_to_chat_format(
        item["input"],
        item["output"]
    )

    for item in training_dataset
]

validation_chat_dataset = [

    convert_to_chat_format(
        item["input"],
        item["output"]
    )

    for item in validation_dataset
]

# ========================================================================
# SAVE JSONL
# ========================================================================

with open(
    TRAIN_FILE,
    "w"
) as f:

    for item in training_chat_dataset:

        f.write(
            json.dumps(item)
            + "\n"
        )

with open(
    VAL_FILE,
    "w"
) as f:

    for item in validation_chat_dataset:

        f.write(
            json.dumps(item)
            + "\n"
        )

# ========================================================================
# DATASET METADATA
# ========================================================================

dataset_hash = hashlib.md5(

    json.dumps(
        dataset
    ).encode()

).hexdigest()

metadata = {

    "dataset_version": "v2.0.0",

    "created_at": str(
        datetime.utcnow()
    ),

    "total_examples": len(dataset),

    "training_examples": len(
        training_dataset
    ),

    "validation_examples": len(
        validation_dataset
    ),

    "model_used": MODEL_NAME,

    "seed": SEED,

    "dataset_hash": dataset_hash
}

with open(
    DATASET_METADATA_FILE,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )

# ========================================================================
# FINAL REPORT
# ========================================================================

print("=" * 80)
print("PRODUCTION DATASET GENERATED")
print("=" * 80)

print()

print("TOTAL EXAMPLES:")
print(len(dataset))

print()

print("TRAINING EXAMPLES:")
print(len(training_dataset))

print()

print("VALIDATION EXAMPLES:")
print(len(validation_dataset))

print()

print("TRAIN FILE:")
print(TRAIN_FILE)

print()

print("VALIDATION FILE:")
print(VAL_FILE)

print()

print("DATASET HASH:")
print(dataset_hash)

print()

print("STATUS: HIGH QUALITY DATASET READY")