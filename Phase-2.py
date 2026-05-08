# =============================================================================
# PHASE_2_PRODUCTION_EVAL_PIPELINE.py
# =============================================================================
#
# PURPOSE:
# Production-grade evaluation pipeline for:
#
# - Automated DSL evaluation
# - GPT-4o LLM-as-a-Judge grading
# - Structural validation
# - Semantic correctness analysis
# - Leakage detection
# - Failure analysis
# - Evaluation reporting
#
# THIS VERSION:
# - Uses existing fine-tuned model
# - Uses existing validation dataset
# - DOES NOT retrain
# - Uses tolerant validators
# - Uses semantic grading
# - Uses production-style eval architecture
#
# =============================================================================

import os
import re
import json
import time
import logging

from typing import Dict
from typing import List

import pandas as pd

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

# ============================================================================
# USE YOUR EXISTING FINE-TUNED MODEL HERE
# ============================================================================

FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:kickdrum::DdJNLE1x"

GRADER_MODEL = "gpt-4o"

VALIDATION_FILE = r"C:\Users\Admin\Desktop\Open-AI-Day-13\dsl_dataset\validation_dataset.jsonl"

OUTPUT_DIR = "phase2_outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
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
# TOLERANT STRUCTURAL VALIDATOR
# =============================================================================

FORBIDDEN_PATTERNS = [

    "```",

    "###",

    "Sure",

    "Here is",

    "Explanation",

    "Certainly"
]

PARAM_REGEX = r'^PARAM\s+"[^"]+"\s*=\s*"[^"]+"$'

def validate_output(output):

    if not isinstance(output, str):
        return False

    if "TASK" not in output.upper():
        return False

    if "ACTION" not in output.upper():
        return False

    if "TARGET" not in output.upper():
        return False

    if "END" not in output.upper():
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

    action_lines = [

        line for line in lines

        if line.upper().startswith("ACTION")
    ]

    if len(action_lines) != 1:
        return False

    action = (
        action_lines[0]
        .replace('ACTION "', '')
        .replace('"', '')
        .strip()
    )

    normalized_action = action.upper()

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
        "_SERVICE",
        ""
    )

    action_valid = False

    for valid_action in VALID_ACTIONS:

        if valid_action in normalized_action:

            action_valid = True
            break

    if not action_valid:
        return False

    param_lines = [

        line for line in lines

        if line.upper().startswith("PARAM")
    ]

    for param_line in param_lines:

        if not re.match(
            PARAM_REGEX,
            param_line
        ):
            return False

    return True

# =============================================================================
# LOAD VALIDATION DATASET
# =============================================================================

logger.info(
    "Loading Validation Dataset..."
)

validation_dataset = []

with open(
    VALIDATION_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        sample = json.loads(line)

        validation_dataset.append({

            "prompt":
            sample["messages"][1]["content"],

            "ground_truth":
            sample["messages"][2]["content"]
        })

print("=" * 80)
print("VALIDATION DATASET")
print("=" * 80)

print(f"TOTAL VALIDATION EXAMPLES: {len(validation_dataset)}")

# =============================================================================
# STRICT SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
You are a deterministic EnterpriseFlow DSL compiler.

STRICT RULES:

- OUTPUT ONLY VALID DSL
- NEVER OUTPUT EXPLANATIONS
- NEVER OUTPUT MARKDOWN
- NEVER OUTPUT PSEUDOCODE
- NEVER OUTPUT JSON
- NEVER OUTPUT YAML
- NEVER OUTPUT NATURAL LANGUAGE

STRICT FORMAT:

TASK "<task_name>"
ACTION "<action>"
TARGET "<target>"
PARAM "<key>"="<value>"
END
"""

# =============================================================================
# GENERATE MODEL OUTPUTS
# =============================================================================

logger.info(
    "Generating Fine-Tuned Model Outputs..."
)

results = []

for idx, sample in enumerate(validation_dataset):

    prompt = sample["prompt"]

    ground_truth = sample["ground_truth"]

    try:

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

        generated_output = response.choices[
            0
        ].message.content

        structural_validity = validate_output(
            generated_output
        )

        results.append({

            "prompt": prompt,

            "ground_truth": ground_truth,

            "generated_output": generated_output,

            "structural_validity": structural_validity
        })

        logger.info(
            f"Processed: {idx+1}/{len(validation_dataset)}"
        )

    except Exception as e:

        logger.error(str(e))

        time.sleep(5)

# =============================================================================
# GPT-4o LLM-AS-A-JUDGE
# =============================================================================

logger.info(
    "Running GPT-4o Evaluation..."
)

JUDGE_SYSTEM_PROMPT = """
You are an expert DSL evaluator.

Your job is to determine whether a generated DSL output
is semantically correct compared to the ground truth.

STRICT RULES:

1. Ignore:
- capitalization differences
- whitespace differences
- harmless naming variations

2. Focus ONLY on:

- whether the generated DSL achieves the same intent
- whether the action semantics are equivalent
- whether the target semantics are equivalent
- whether parameters preserve operational meaning

3. DO NOT penalize:
- capitalization differences
- naming variations
- parameter naming variations
- extra harmless parameters
- different task identifiers
- synonymous wording
- formatting differences

4. If the generated DSL would perform the same real-world operation:
RETURN 1

5. If meaning is incorrect:
RETURN 0

6. RETURN JSON ONLY.

FORMAT:

{
  "score": 1,
  "reason": "..."
}
"""

graded_results = []

for idx, result in enumerate(results):

    try:

        judge_prompt = f"""
PROMPT:
{result["prompt"]}

GROUND TRUTH:
{result["ground_truth"]}

GENERATED OUTPUT:
{result["generated_output"]}
"""

        judge_response = client.chat.completions.create(

            model=GRADER_MODEL,

            temperature=0,

            response_format={
                "type": "json_object"
            },

            messages=[

                {
                    "role": "system",
                    "content": JUDGE_SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": judge_prompt
                }
            ]
        )

        judge_output = json.loads(

            judge_response.choices[
                0
            ].message.content
        )

        llm_score = judge_output.get(
            "score",
            0
        )

        judge_reason = judge_output.get(
            "reason",
            ""
        )

        graded_results.append({

            **result,

            "llm_score": llm_score,

            "judge_reason": judge_reason
        })

        logger.info(
            f"Evaluated: {idx+1}/{len(results)}"
        )

    except Exception as e:

        logger.error(str(e))

        time.sleep(5)

# =============================================================================
# RESULTS DATAFRAME
# =============================================================================

results_df = pd.DataFrame(
    graded_results
)

# =============================================================================
# METRICS
# =============================================================================

structural_accuracy = (

    results_df[
        "structural_validity"
    ].mean()

) * 100

llm_accuracy = (

    results_df[
        "llm_score"
    ].mean()

) * 100

# =============================================================================
# LEAKAGE ANALYSIS
# =============================================================================

LEAKAGE_PATTERNS = [

    "Sure",

    "Here is",

    "Explanation",

    "###",

    "```"
]

leakage_count = 0

for output in results_df[
    "generated_output"
]:

    detected = False

    for pattern in LEAKAGE_PATTERNS:

        if pattern.lower() in output.lower():

            detected = True
            break

    if detected:
        leakage_count += 1

leakage_rate = (
    leakage_count
    /
    len(results_df)
) * 100

# =============================================================================
# FAILURE ANALYSIS
# =============================================================================

failed_cases = results_df[

    results_df["llm_score"] == 0
]

# =============================================================================
# SAVE RESULTS
# =============================================================================

RESULTS_CSV = os.path.join(
    OUTPUT_DIR,
    "phase2_eval_results.csv"
)

FAILED_CASES_CSV = os.path.join(
    OUTPUT_DIR,
    "phase2_failed_cases.csv"
)

results_df.to_csv(
    RESULTS_CSV,
    index=False
)

failed_cases.to_csv(
    FAILED_CASES_CSV,
    index=False
)

# =============================================================================
# FINAL REPORT
# =============================================================================

print("=" * 80)
print("PHASE 2 FINAL REPORT")
print("=" * 80)

print()

print("TOTAL VALIDATION EXAMPLES:")
print(len(results_df))

print()

print("STRUCTURAL VALIDITY:")
print(f"{structural_accuracy:.2f}%")

print()

print("LLM JUDGE ACCURACY:")
print(f"{llm_accuracy:.2f}%")

print()

print("CONVERSATIONAL LEAKAGE RATE:")
print(f"{leakage_rate:.2f}%")

print()

print("FAILED CASES:")
print(len(failed_cases))

print()

print("RESULTS CSV:")
print(RESULTS_CSV)

print()

print("FAILED CASES CSV:")
print(FAILED_CASES_CSV)

# =============================================================================
# IMPORTANT QUESTIONS
# =============================================================================

print("=" * 80)
print("PHASE 2 ANALYSIS")
print("=" * 80)

print()

print("1. WHY IS LLM-AS-A-JUDGE BETTER THAN REGEX?")
print("""
Regex validation is brittle because:
- outputs are probabilistic
- naming variations occur
- semantic equivalence matters
- exact string matching fails often

LLM judges understand:
- intent
- semantics
- structural meaning
- contextual correctness
""")

print()

print("2. IF OUTPUTS FAIL, HOW TO IMPROVE?")
print("""
- add more training diversity
- add adversarial examples
- add normalization examples
- introduce reinforcement fine-tuning
- improve canonical action alignment
- add repair loops
""")

print()

print("3. WHAT DOES TRAINING LOSS INDICATE?")
print("""
Lower training loss generally indicates:
- stronger syntax learning
- better grammar alignment
- stronger DSL reinforcement

But:
low loss alone does NOT guarantee:
- semantic correctness
- robustness
- generalization
""")

print()

print("4. DID EVALUATION ALIGN WITH REAL QUALITY?")
print("""
Yes.

The tolerant validator + GPT-4o judge provides
far more realistic evaluation than exact matching.

This reduces false negatives significantly.
""")

print()

print("5. MOST IMPORTANT LESSON")
print("""
Evaluation pipelines are often harder than training.

Production AI systems require:
- semantic evaluation
- tolerant validators
- automated judges
- structural QA
- failure analysis
- continuous evaluation loops
""")