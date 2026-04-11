# Multimodal Context-Aware AI Assistant

Production-oriented LangChain service that supports:

- Context-aware personalization using hidden `user_id` profiles
- Strict JSON responses enforced with structured output schemas
- Thread-level memory via LangGraph checkpointing
- Cross-session memory via persisted interaction history
- Multimodal image analysis
- Dynamic model selection and style middleware

## Architecture

The implementation maps directly onto the requested LLD:

- `FastAPI` exposes `POST /api/v1/chat`
- `SQLAlchemy + SQLite` persist user profiles and interaction history
- `LangChain create_agent(...)` powers the assistant
- `ChatOpenRouter` provides the model layer through OpenRouter's OpenAI-compatible routing API
- `LangGraph` checkpointer maintains short-term memory while the service is running
- Middleware injects personalization context, style guidance, and model routing
- Tools provide weather lookup and uploaded-image analysis
- Pydantic schemas enforce the response contract

## Request Shape

```json
{
  "user_id": "user_123",
  "thread_id": "support-thread-1",
  "message": "What's the weather like today?",
  "mode": "expert",
  "image": {
    "data": "base64-encoded-image-or-data-url",
    "mime_type": "image/png"
  }
}
```

`thread_id` is optional and defaults to `"default"`. It was added so the service can maintain multiple conversation threads per user.

## Response Shape

```json
{
  "status": "success",
  "data": {
    "response": "It is 21.4 C and cloudy in San Francisco, US.",
    "temperature": 21.4,
    "summary": "cloudy",
    "location": "San Francisco, US",
    "image_description": null,
    "objects": []
  },
  "metadata": {
    "model_used": "qwen/qwen3.6-plus:free",
    "tools_invoked": ["get_current_weather"],
    "thread_id": "support-thread-1",
    "timestamp": "2026-04-11T17:25:10.123456+00:00"
  }
}
```

## Quick Start

1. Create a virtual environment.
2. Install dependencies:

```bash
python3 -m pip install -e ".[dev]"
```

3. Copy the environment template:

```bash
cp .env.example .env
```

4. Set your OpenRouter key in `.env`:

```bash
ASSISTANT_OPENROUTER_API_KEY=your-openrouter-key
```

5. Start the API:

```bash
uvicorn multimodal_assistant.api.main:app --reload
```

## Notes

- If `ASSISTANT_OPENROUTER_API_KEY` is configured, the service uses the full LangChain agent with middleware-driven routing through OpenRouter.
- If no model key is present, the app falls back to a deterministic offline runner so local development and tests still work.
- Cross-session continuity is preserved through persisted interaction history, which is injected back into the agent as hidden context on subsequent requests.
- Default free-model settings pin `qwen/qwen3.6-plus:free` for text and vision so the app avoids unstable router aliases. Override these in `.env` if you want different models.
- The project was verified locally on Python 3.14, but `langchain_core` currently emits an upstream compatibility warning there. For production deployments, Python 3.11 to 3.13 is the safer target today.
- Weather defaults to a deterministic mock provider, but the design is pluggable and can be upgraded to a live provider.

## Tests

```bash
pytest
```
