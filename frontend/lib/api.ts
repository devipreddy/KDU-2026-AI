import type {
  AgentThreadResponse,
  SessionResponse,
  ThreadSummary,
} from "./types";

export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export const CHATKIT_DOMAIN_KEY =
  process.env.NEXT_PUBLIC_CHATKIT_DOMAIN_KEY ?? "local-dev";

export function apiUrl(path: string): string {
  return `${BACKEND_URL}${path}`;
}

export function websocketUrl(path: string, clientSecret: string): string {
  const url = new URL(path, BACKEND_URL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("client_secret", clientSecret);
  return url.toString();
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) {
        detail = body.detail;
      }
    } catch {
      // Keep the HTTP status text when the backend returned a non-JSON error.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function createCustomerSession(): Promise<SessionResponse> {
  const response = await fetch(apiUrl("/api/session"), {
    method: "POST",
    credentials: "include",
  });
  return parseJsonResponse<SessionResponse>(response);
}

export async function createAgentSession(
  agentToken: string,
): Promise<SessionResponse> {
  const response = await fetch(apiUrl("/api/agent/session"), {
    method: "POST",
    credentials: "include",
    headers: {
      "X-Agent-Token": agentToken,
    },
  });
  return parseJsonResponse<SessionResponse>(response);
}

export async function authedFetch<T>(
  path: string,
  clientSecret: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${clientSecret}`);
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(apiUrl(path), {
    ...init,
    credentials: "include",
    headers,
  });
  return parseJsonResponse<T>(response);
}

export function getThreadSummary(
  threadId: string,
  clientSecret: string,
): Promise<ThreadSummary> {
  return authedFetch<ThreadSummary>(
    `/api/threads/${encodeURIComponent(threadId)}/summary`,
    clientSecret,
  );
}

export function listHandoffQueue(
  clientSecret: string,
): Promise<ThreadSummary[]> {
  return authedFetch<ThreadSummary[]>("/api/agent/handoff/queue", clientSecret);
}

export function getAgentThread(
  threadId: string,
  clientSecret: string,
): Promise<AgentThreadResponse> {
  return authedFetch<AgentThreadResponse>(
    `/api/agent/threads/${encodeURIComponent(threadId)}`,
    clientSecret,
  );
}

export function claimHandoff(
  threadId: string,
  clientSecret: string,
  displayName: string,
): Promise<{ status: string }> {
  return authedFetch<{ status: string }>(
    `/api/agent/handoff/${encodeURIComponent(threadId)}/claim`,
    clientSecret,
    {
      method: "POST",
      body: JSON.stringify({ display_name: displayName }),
    },
  );
}

export function sendHumanMessage(
  threadId: string,
  clientSecret: string,
  text: string,
): Promise<{ status: string }> {
  return authedFetch<{ status: string }>(
    `/api/agent/handoff/${encodeURIComponent(threadId)}/message`,
    clientSecret,
    {
      method: "POST",
      body: JSON.stringify({ text }),
    },
  );
}

export function releaseHandoff(
  threadId: string,
  clientSecret: string,
  resumeAi: boolean,
): Promise<{ status: string }> {
  return authedFetch<{ status: string }>(
    `/api/agent/handoff/${encodeURIComponent(threadId)}/release`,
    clientSecret,
    {
      method: "POST",
      body: JSON.stringify({ resume_ai: resumeAi }),
    },
  );
}
