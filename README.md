# Production Guardrails, Cloud Content Safety & Observability

## Overview

This repository contains a complete hands-on implementation of a production-grade AI safety architecture for customer-service style LLM applications.

The project demonstrates how modern AI systems cannot rely solely on the base model’s internal alignment and safety training. Instead, production systems require layered defenses including:

- Prompt Injection Protection
- PII Detection & Redaction
- Open-Source Guardrails
- Cloud Content Safety APIs
- Observability & Tracing
- Moderation Threshold Analysis

The implementation uses:

- Ollama + Llama 3 (Local LLM)
- AWS Bedrock Guardrails
- LangSmith Observability
- Python-based deterministic guardrails

---

# Repository Structure

```text
.
├── Phase1_Prompt_Injection_and_PII_Protection.ipynb
├── Phase2_AWS_Bedrock_Guardrails.ipynb
├── Phase3_LangSmith_Observability.ipynb
├── requirements.txt
└── README.md