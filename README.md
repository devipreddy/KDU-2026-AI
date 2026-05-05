# Voyager ChatKit Travel Lab

Production-grade hands-on lab for a server-driven travel booking chat surface using:

- FastAPI with `ChatKitServer`
- Secure signed `client_secret` session tokens
- ChatKit streaming through the local backend
- Server-streamed widgets with hidden custom actions
- Next.js customer chat and human handoff console

## Run The Backend

```powershell
cd backend
python -m pip install -e .[dev]
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload --port 8000
```

The backend uses `gpt-4o-mini` when `OPENAI_API_KEY` is set. Without an API key it falls back to the local mock travel model so the lab can still demonstrate streaming, widgets, actions, and handoff without spending tokens.

## Run The Frontend

```powershell
cd frontend
Copy-Item .env.local.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000` for the customer chat. Open `http://localhost:3000/handoff` for the human specialist console.

## Useful Flows

1. Ask the customer chat for a trip, for example `Find a Tokyo trip with hotel`.
2. Click `Book Now` in the streamed widget. The frontend catches the client widget action and forwards a hidden `travel.book_now.confirm` action to FastAPI.
3. Paste a thread id from a different browser profile into the thread-isolation panel. FastAPI rejects cross-user access before ChatKit can stream that thread.
4. Click `Request specialist`, then open `/handoff`, claim the thread, send a manual response, and release it back to AI.

## Environment

Backend `.env`:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=
USE_MOCK_MODEL=false
JWT_SECRET=replace-with-at-least-32-random-characters
HUMAN_AGENT_DASHBOARD_TOKEN=agent-demo-token
ALLOWED_ORIGINS=http://localhost:3000
```

Frontend `.env.local`:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_CHATKIT_DOMAIN_KEY=local-dev
```
