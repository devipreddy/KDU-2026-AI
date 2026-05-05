"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  ArrowLeft,
  CheckCircle2,
  Headphones,
  LoaderCircle,
  LogIn,
  MessageSquare,
  RefreshCw,
  Send,
  Unlock,
  Wifi,
} from "lucide-react";

import {
  claimHandoff,
  createAgentSession,
  getAgentThread,
  listHandoffQueue,
  releaseHandoff,
  sendHumanMessage,
  websocketUrl,
} from "@/lib/api";
import type {
  AgentThreadItem,
  AgentThreadResponse,
  RealtimeEvent,
  SessionResponse,
  ThreadSummary,
} from "@/lib/types";

const DEFAULT_AGENT_TOKEN = "agent-demo-token";

function itemText(item: AgentThreadItem): string {
  if (typeof item.copy_text === "string" && item.copy_text) {
    return item.copy_text;
  }

  if (Array.isArray(item.content)) {
    const text = item.content
      .map((part) => {
        if (typeof part.text === "string") {
          return part.text;
        }
        if (typeof part.value === "string") {
          return part.value;
        }
        return "";
      })
      .filter(Boolean)
      .join(" ");
    if (text) {
      return text;
    }
  }

  return item.type ? `Thread item: ${item.type}` : "Thread item";
}

function itemActor(item: AgentThreadItem): string {
  const type = String(item.type ?? "").toLowerCase();
  if (type.includes("user")) {
    return "Customer";
  }
  if (type.includes("hidden")) {
    return "System";
  }
  if (type.includes("widget")) {
    return "Widget";
  }
  return "Assistant";
}

function modeLabel(mode: string): string {
  if (mode === "handoff_requested") {
    return "Waiting";
  }
  if (mode === "human") {
    return "Human";
  }
  return "AI";
}

export function HandoffDashboard() {
  const [agentToken, setAgentToken] = useState(DEFAULT_AGENT_TOKEN);
  const [displayName, setDisplayName] = useState("Travel Specialist");
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [queue, setQueue] = useState<ThreadSummary[]>([]);
  const [selectedThreadId, setSelectedThreadId] = useState<string | null>(null);
  const [thread, setThread] = useState<AgentThreadResponse | null>(null);
  const [message, setMessage] = useState("");
  const [status, setStatus] = useState("Signed out");
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function signIn(token = agentToken) {
    setLoading("Signing in");
    setError(null);
    try {
      const nextSession = await createAgentSession(token);
      setSession(nextSession);
      setStatus("Agent session ready");
      window.localStorage.setItem("travel-agent-token", token);
      await refreshQueue(nextSession.client_secret);
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Agent login failed";
      setError(detail);
      setStatus("Sign in failed");
    } finally {
      setLoading(null);
    }
  }

  async function refreshQueue(clientSecret = session?.client_secret) {
    if (!clientSecret) {
      return;
    }
    setError(null);
    try {
      const nextQueue = await listHandoffQueue(clientSecret);
      setQueue(nextQueue);
      setStatus(`Queue updated: ${nextQueue.length} thread${nextQueue.length === 1 ? "" : "s"}`);
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Unable to refresh queue";
      setError(detail);
    }
  }

  async function openThread(threadId: string, clientSecret = session?.client_secret) {
    if (!clientSecret) {
      return;
    }
    setSelectedThreadId(threadId);
    setLoading("Loading thread");
    setError(null);
    try {
      const nextThread = await getAgentThread(threadId, clientSecret);
      setThread(nextThread);
      setStatus(`Thread opened: ${threadId}`);
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Unable to open thread";
      setError(detail);
    } finally {
      setLoading(null);
    }
  }

  async function claimSelectedThread() {
    if (!selectedThreadId || !session) {
      return;
    }
    setLoading("Claiming handoff");
    setError(null);
    try {
      await claimHandoff(selectedThreadId, session.client_secret, displayName);
      await openThread(selectedThreadId, session.client_secret);
      await refreshQueue(session.client_secret);
      setStatus("Thread claimed");
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Unable to claim handoff";
      setError(detail);
    } finally {
      setLoading(null);
    }
  }

  async function sendMessage() {
    if (!selectedThreadId || !session || !message.trim()) {
      return;
    }
    const text = message.trim();
    setMessage("");
    setLoading("Sending message");
    setError(null);
    try {
      await sendHumanMessage(selectedThreadId, session.client_secret, text);
      await openThread(selectedThreadId, session.client_secret);
      setStatus("Manual reply sent");
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Unable to send message";
      setError(detail);
      setMessage(text);
    } finally {
      setLoading(null);
    }
  }

  async function releaseSelectedThread(resumeAi: boolean) {
    if (!selectedThreadId || !session) {
      return;
    }
    setLoading(resumeAi ? "Resuming AI" : "Releasing to queue");
    setError(null);
    try {
      await releaseHandoff(selectedThreadId, session.client_secret, resumeAi);
      await openThread(selectedThreadId, session.client_secret);
      await refreshQueue(session.client_secret);
      setStatus(resumeAi ? "AI resumed" : "Thread returned to queue");
    } catch (err) {
      const detail = err instanceof Error ? err.message : "Unable to release handoff";
      setError(detail);
    } finally {
      setLoading(null);
    }
  }

  useEffect(() => {
    const saved = window.localStorage.getItem("travel-agent-token");
    const token = saved || DEFAULT_AGENT_TOKEN;
    setAgentToken(token);
    void signIn(token);
  }, []);

  useEffect(() => {
    if (!session?.client_secret) {
      return;
    }

    const socket = new WebSocket(
      websocketUrl("/ws/handoff/queue", session.client_secret),
    );

    socket.addEventListener("open", () => setStatus("Queue websocket connected"));
    socket.addEventListener("message", (event) => {
      try {
        const payload = JSON.parse(String(event.data)) as RealtimeEvent;
        setStatus(`Queue event: ${payload.type}`);
      } catch {
        setStatus("Queue event received");
      }
      void refreshQueue(session.client_secret);
    });
    socket.addEventListener("close", () => setStatus("Queue websocket closed"));
    socket.addEventListener("error", () => setStatus("Queue websocket error"));

    return () => {
      socket.close();
    };
  }, [session?.client_secret]);

  useEffect(() => {
    if (!session?.client_secret || !selectedThreadId) {
      return;
    }

    const socket = new WebSocket(
      websocketUrl(
        `/ws/threads/${encodeURIComponent(selectedThreadId)}`,
        session.client_secret,
      ),
    );

    socket.addEventListener("message", () => {
      void openThread(selectedThreadId, session.client_secret);
    });

    return () => {
      socket.close();
    };
  }, [session?.client_secret, selectedThreadId]);

  const selectedSummary = queue.find(
    (item) => item.thread_id === selectedThreadId,
  );
  const canSend = Boolean(
    selectedThreadId &&
      session &&
      message.trim() &&
      thread?.thread.metadata?.conversation_mode === "human",
  );

  return (
    <main className="handoff-workspace">
      <header className="handoff-header">
        <div>
          <p className="eyebrow">Human desk</p>
          <h1>Travel specialist console</h1>
        </div>
        <Link className="ghost-button" href="/">
          <ArrowLeft size={16} aria-hidden />
          Customer chat
        </Link>
      </header>

      <section className="agent-login" aria-label="Agent session">
        <label className="field compact">
          <span>Agent token</span>
          <input
            value={agentToken}
            onChange={(event) => setAgentToken(event.target.value)}
            spellCheck={false}
          />
        </label>
        <label className="field compact">
          <span>Display name</span>
          <input
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
          />
        </label>
        <button className="primary-button" type="button" onClick={() => void signIn()}>
          <LogIn size={16} aria-hidden />
          Sign in
        </button>
        <button
          className="secondary-button"
          type="button"
          onClick={() => void refreshQueue()}
          disabled={!session}
        >
          <RefreshCw size={16} aria-hidden />
          Refresh
        </button>
        <p className="console-status">
          {loading ? <LoaderCircle className="spin" size={16} aria-hidden /> : <Wifi size={16} aria-hidden />}
          {loading ?? status}
        </p>
      </section>

      {error ? <p className="dashboard-error">{error}</p> : null}

      <div className="handoff-grid">
        <aside className="queue-panel" aria-label="Handoff queue">
          <div className="section-heading">
            <Headphones size={18} aria-hidden />
            <h2>Queue</h2>
          </div>
          <div className="queue-list">
            {queue.length ? (
              queue.map((item) => (
                <button
                  key={item.thread_id}
                  className="queue-item"
                  type="button"
                  data-selected={item.thread_id === selectedThreadId}
                  onClick={() => void openThread(item.thread_id)}
                >
                  <span className="queue-title">
                    {item.title || item.thread_id}
                    <small>{modeLabel(item.conversation_mode)}</small>
                  </span>
                  <span className="queue-preview">
                    {item.last_message_preview || "No preview yet"}
                  </span>
                </button>
              ))
            ) : (
              <p className="empty-state">No handoff threads waiting.</p>
            )}
          </div>
        </aside>

        <section className="thread-panel" aria-label="Selected conversation">
          <div className="thread-toolbar">
            <div>
              <p className="eyebrow">Selected thread</p>
              <h2>{selectedThreadId ?? "No thread selected"}</h2>
              {selectedSummary?.assigned_agent_name ? (
                <p className="small-note">
                  Claimed by {selectedSummary.assigned_agent_name}
                </p>
              ) : null}
            </div>
            <div className="toolbar-actions">
              <button
                className="secondary-button"
                type="button"
                onClick={claimSelectedThread}
                disabled={!selectedThreadId || !session || Boolean(loading)}
              >
                <CheckCircle2 size={16} aria-hidden />
                Claim
              </button>
              <button
                className="secondary-button"
                type="button"
                onClick={() => void releaseSelectedThread(true)}
                disabled={!selectedThreadId || !session || Boolean(loading)}
              >
                <Unlock size={16} aria-hidden />
                Resume AI
              </button>
            </div>
          </div>

          <div className="transcript">
            {thread?.items.length ? (
              thread.items.map((item, index) => (
                <article
                  className="transcript-item"
                  data-actor={itemActor(item).toLowerCase()}
                  key={item.id ?? `${item.type}-${index}`}
                >
                  <span>{itemActor(item)}</span>
                  <p>{itemText(item)}</p>
                </article>
              ))
            ) : (
              <div className="empty-state large">
                <MessageSquare size={24} aria-hidden />
                <p>Select a waiting thread to review the conversation.</p>
              </div>
            )}
          </div>

          <form
            className="agent-composer"
            onSubmit={(event) => {
              event.preventDefault();
              void sendMessage();
            }}
          >
            <textarea
              value={message}
              onChange={(event) => setMessage(event.target.value)}
              placeholder="Send a manual response while this thread is in human mode..."
              rows={3}
            />
            <button className="primary-button" type="submit" disabled={!canSend || Boolean(loading)}>
              <Send size={16} aria-hidden />
              Send
            </button>
          </form>
        </section>
      </div>
    </main>
  );
}
