"use client";

import Link from "next/link";
import Script from "next/script";
import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  ArrowRight,
  Headphones,
  LoaderCircle,
  LockKeyhole,
  Plane,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Wifi,
} from "lucide-react";

import {
  CHATKIT_DOMAIN_KEY,
  apiUrl,
  createCustomerSession,
  getThreadSummary,
  websocketUrl,
} from "@/lib/api";
import { secureChatKitFetch, waitForChatKitElement, type ChatKitElement } from "@/lib/chatkit";
import type { ChatKitWidgetAction, RealtimeEvent, SessionResponse } from "@/lib/types";

type ActionState = {
  key: string;
  label: string;
} | null;

type StreamState = "booting" | "ready" | "streaming" | "error";

function actionType(action: ChatKitWidgetAction): string {
  return action.type ?? action.action?.type ?? "";
}

function actionPayload(action: ChatKitWidgetAction): Record<string, unknown> {
  return action.payload ?? action.action?.payload ?? {};
}

function randomIdempotencyKey(): string {
  if ("crypto" in window && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function shortToken(token: string): string {
  return `${token.slice(0, 18)}...${token.slice(-10)}`;
}

export function ChatKitExperience() {
  const chatkitRef = useRef<ChatKitElement | null>(null);
  const clientSecretRef = useRef<string | null>(null);
  const currentThreadRef = useRef<string | null>(null);
  const actionHandlerRef = useRef<(action: ChatKitWidgetAction) => Promise<void>>(
    async () => {},
  );
  const actionLocksRef = useRef<Set<string>>(new Set());

  const [scriptReady, setScriptReady] = useState(false);
  const [session, setSession] = useState<SessionResponse | null>(null);
  const [streamState, setStreamState] = useState<StreamState>("booting");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [foreignThreadId, setForeignThreadId] = useState("");
  const [challengeResult, setChallengeResult] = useState<string>(
    "Paste a thread id from another browser profile to test ownership checks.",
  );
  const [actionState, setActionState] = useState<ActionState>(null);
  const [realtimeState, setRealtimeState] = useState("idle");
  const [conversationMode, setConversationMode] = useState("ai");
  const [error, setError] = useState<string | null>(null);
  const [activity, setActivity] = useState<string[]>([]);

  function pushActivity(message: string) {
    setActivity((items) => [message, ...items].slice(0, 6));
  }

  async function refreshSession() {
    setError(null);
    setStreamState((state) => (state === "booting" ? "booting" : "ready"));
    try {
      const nextSession = await createCustomerSession();
      clientSecretRef.current = nextSession.client_secret;
      setSession(nextSession);
      pushActivity("Client secret refreshed from FastAPI");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to start session";
      setError(message);
      setStreamState("error");
    }
  }

  async function sendServerAction(
    type: string,
    payload: Record<string, unknown>,
  ) {
    const element = chatkitRef.current;
    if (!element?.sendCustomAction) {
      throw new Error("ChatKit is still loading. Try again in a moment.");
    }
    await element.sendCustomAction({ type, payload });
  }

  async function refreshThreadSummary(
    targetThreadId = currentThreadRef.current,
  ): Promise<string | null> {
    const token = clientSecretRef.current;
    if (!targetThreadId || !token) {
      return null;
    }

    const summary = await getThreadSummary(targetThreadId, token);
    setConversationMode(summary.conversation_mode);
    return summary.conversation_mode;
  }

  async function handleWidgetAction(action: ChatKitWidgetAction) {
    const type = actionType(action);
    const payload = actionPayload(action);

    if (type !== "travel.book_now") {
      pushActivity(`Ignored unsupported client action: ${type || "unknown"}`);
      return;
    }

    const offerId = String(payload.offer_id ?? "");
    if (!offerId) {
      setError("The booking widget did not include an offer id.");
      return;
    }

    const lockKey = `${currentThreadRef.current ?? "new"}:${offerId}`;
    if (actionLocksRef.current.has(lockKey)) {
      pushActivity("Duplicate booking click suppressed locally");
      return;
    }

    actionLocksRef.current.add(lockKey);
    setActionState({ key: lockKey, label: `Holding ${offerId}` });
    setError(null);

    try {
      await sendServerAction("travel.book_now.confirm", {
        offer_id: offerId,
        idempotency_key: randomIdempotencyKey(),
      });
      pushActivity("Book Now forwarded as a hidden server action");
    } catch (err) {
      const message = err instanceof Error ? err.message : "Booking action failed";
      setError(message);
    } finally {
      actionLocksRef.current.delete(lockKey);
      setActionState(null);
    }
  }

  actionHandlerRef.current = handleWidgetAction;

  useEffect(() => {
    void refreshSession();
  }, []);

  useEffect(() => {
    const element = chatkitRef.current;
    if (!element) {
      return;
    }

    const onThreadChange = (event: Event) => {
      const detail = (event as CustomEvent<{ threadId: string | null }>).detail;
      currentThreadRef.current = detail.threadId;
      setThreadId(detail.threadId);
      if (detail.threadId) {
        pushActivity(`Thread opened: ${detail.threadId}`);
        void refreshThreadSummary(detail.threadId).catch(() => {
          setConversationMode("ai");
        });
      } else {
        setConversationMode("ai");
      }
    };
    const onStart = () => {
      setStreamState("streaming");
      pushActivity("Streaming response started");
    };
    const onEnd = () => {
      setStreamState("ready");
      pushActivity("Streaming response completed");
    };
    const onError = (event: Event) => {
      const detail = (event as CustomEvent<{ error?: Error }>).detail;
      setStreamState("error");
      setError(detail?.error?.message ?? "ChatKit request failed");
      pushActivity("ChatKit reported an error");
    };

    element.addEventListener("chatkit.thread.change", onThreadChange);
    element.addEventListener("chatkit.response.start", onStart);
    element.addEventListener("chatkit.response.end", onEnd);
    element.addEventListener("chatkit.error", onError);

    return () => {
      element.removeEventListener("chatkit.thread.change", onThreadChange);
      element.removeEventListener("chatkit.response.start", onStart);
      element.removeEventListener("chatkit.response.end", onEnd);
      element.removeEventListener("chatkit.error", onError);
    };
  }, []);

  useEffect(() => {
    if (!scriptReady || !session) {
      return;
    }

    let cancelled = false;

    async function configureChatKit() {
      await waitForChatKitElement();
      if (cancelled || !chatkitRef.current) {
        return;
      }

      chatkitRef.current.setOptions?.({
        api: {
          url: apiUrl("/api/chatkit"),
          domainKey: CHATKIT_DOMAIN_KEY,
          fetch: secureChatKitFetch(() => clientSecretRef.current),
        },
        initialThread: currentThreadRef.current,
        theme: {
          colorScheme: "light",
        },
        header: {
          enabled: true,
          title: {
            enabled: true,
            text: "Voyager Travel Desk",
          },
        },
        composer: {
          placeholder: "Ask for flights, hotels, or a complete package...",
        },
        startScreen: {
          greeting: "Where should we send you next?",
          prompts: [
            {
              label: "Tokyo hotel trip",
              prompt: "Find a Tokyo trip with hotel",
            },
            {
              label: "Lisbon weekend",
              prompt: "Book a Lisbon long weekend",
            },
            {
              label: "Beach budget",
              prompt: "Show beach vacations under budget",
            },
          ],
        },
        widgets: {
          onAction: (action: ChatKitWidgetAction) =>
            actionHandlerRef.current(action),
        },
      });
      setStreamState("ready");
      pushActivity("ChatKit connected to local backend");
    }

    void configureChatKit();

    return () => {
      cancelled = true;
    };
  }, [scriptReady, session]);

  useEffect(() => {
    if (!session?.client_secret || !threadId) {
      setRealtimeState("idle");
      return;
    }

    const socket = new WebSocket(
      websocketUrl(`/ws/threads/${encodeURIComponent(threadId)}`, session.client_secret),
    );
    setRealtimeState("connecting");

    socket.addEventListener("open", () => {
      setRealtimeState("connected");
    });

    socket.addEventListener("message", (event) => {
      try {
        const payload = JSON.parse(String(event.data)) as RealtimeEvent;
        pushActivity(`Realtime event: ${payload.type}`);
        if (typeof payload.payload?.conversation_mode === "string") {
          setConversationMode(payload.payload.conversation_mode);
        }
      } catch {
        pushActivity("Realtime update received");
      }
      void chatkitRef.current?.fetchUpdates?.();
      void refreshThreadSummary();
    });

    socket.addEventListener("close", () => {
      setRealtimeState("closed");
    });

    socket.addEventListener("error", () => {
      setRealtimeState("error");
    });

    return () => {
      socket.close();
    };
  }, [session?.client_secret, threadId]);

  function startNewThread() {
    currentThreadRef.current = null;
    setThreadId(null);
    setConversationMode("ai");
    setChallengeResult("Started a fresh thread for the current signed session.");
    chatkitRef.current?.setThreadId?.(null);
  }

  async function testForeignThread() {
    const candidate = foreignThreadId.trim();
    if (!candidate || !session) {
      return;
    }

    setError(null);
    setChallengeResult(`Opening ${candidate} through ChatKit and validating ownership...`);
    currentThreadRef.current = candidate;
    setThreadId(candidate);
    chatkitRef.current?.setThreadId?.(candidate);

    try {
      await getThreadSummary(candidate, session.client_secret);
      setChallengeResult(
        "The backend returned this thread for the current user. If it belongs to another user, validation needs tightening.",
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Access denied";
      setChallengeResult(`Rejected by FastAPI validation: ${message}`);
      pushActivity("Cross-thread access attempt rejected");
      currentThreadRef.current = null;
      setThreadId(null);
      setConversationMode("ai");
      chatkitRef.current?.setThreadId?.(null);
    }
  }

  async function requestHumanHandoff() {
    if (!threadId) {
      setError("Start or open a thread before requesting handoff.");
      return;
    }

    setActionState({ key: "handoff", label: "Requesting handoff" });
    setError(null);
    try {
      await sendServerAction("support.request_handoff", {
        reason: "Customer requested a live travel specialist from the frontend.",
      });
      const mode = await refreshThreadSummary();
      if (mode !== "handoff_requested" && mode !== "human") {
        setError("Handoff request was sent, but the thread mode did not change yet.");
        pushActivity("Handoff request sent without confirmed mode change");
        return;
      }
      pushActivity(`Human handoff mode confirmed: ${mode}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unable to request handoff";
      setError(message);
    } finally {
      setActionState(null);
    }
  }

  const ready = Boolean(scriptReady && session && streamState !== "booting");

  return (
    <main className="workspace">
      <Script
        src="https://cdn.platform.openai.com/deployments/chatkit/chatkit.js"
        strategy="afterInteractive"
        onReady={() => setScriptReady(true)}
        onLoad={() => setScriptReady(true)}
        onError={() => {
          setError("Unable to load the ChatKit browser script.");
          setStreamState("error");
        }}
      />

      <section className="chat-area" aria-label="Travel chat">
        <div className="chat-topbar">
          <div>
            <p className="eyebrow">Voyager</p>
            <h1>Travel booking desk</h1>
          </div>
          <div className="topbar-actions">
            <Link className="ghost-button" href="/handoff">
              <Headphones size={16} aria-hidden />
              Agent console
            </Link>
            <button className="icon-button" type="button" onClick={startNewThread}>
              <RotateCcw size={17} aria-hidden />
              <span className="sr-only">Start new thread</span>
            </button>
          </div>
        </div>

        <div className="chatkit-frame" data-ready={ready}>
          {!ready ? (
            <div className="loading-panel">
              <LoaderCircle className="spin" size={24} aria-hidden />
              <span>Starting secure chat session...</span>
            </div>
          ) : null}
          <openai-chatkit ref={chatkitRef} className="chatkit-element" />
        </div>
      </section>

      <aside className="control-rail" aria-label="Session controls">
        <section className="rail-section status-panel">
          <div className="section-heading">
            <ShieldCheck size={18} aria-hidden />
            <h2>Session</h2>
          </div>
          <dl className="facts">
            <div>
              <dt>State</dt>
              <dd data-state={streamState}>{streamState}</dd>
            </div>
            <div>
              <dt>Realtime</dt>
              <dd>{realtimeState}</dd>
            </div>
            <div>
              <dt>Thread</dt>
              <dd title={threadId ?? "New thread"}>{threadId ?? "new"}</dd>
            </div>
            <div>
              <dt>Mode</dt>
              <dd>{conversationMode}</dd>
            </div>
            <div>
              <dt>Role</dt>
              <dd>{session?.role ?? "pending"}</dd>
            </div>
          </dl>
          {session ? (
            <p className="token-preview" title={session.client_secret}>
              {shortToken(session.client_secret)}
            </p>
          ) : null}
          <button className="secondary-button" type="button" onClick={refreshSession}>
            <RefreshCw size={16} aria-hidden />
            Refresh token
          </button>
        </section>

        <section className="rail-section">
          <div className="section-heading">
            <LockKeyhole size={18} aria-hidden />
            <h2>Thread isolation</h2>
          </div>
          <label className="field">
            <span>Hardcoded thread id</span>
            <input
              value={foreignThreadId}
              onChange={(event) => setForeignThreadId(event.target.value)}
              placeholder="thread_..."
              spellCheck={false}
            />
          </label>
          <button
            className="primary-button"
            type="button"
            onClick={testForeignThread}
            disabled={!session || !foreignThreadId.trim()}
          >
            Validate ownership
            <ArrowRight size={16} aria-hidden />
          </button>
          <p className="small-note">{challengeResult}</p>
        </section>

        <section className="rail-section">
          <div className="section-heading">
            <Headphones size={18} aria-hidden />
            <h2>Handoff</h2>
          </div>
          <button
            className="primary-button warm"
            type="button"
            onClick={requestHumanHandoff}
            disabled={!threadId || Boolean(actionState)}
          >
            {actionState?.key === "handoff" ? (
              <LoaderCircle className="spin" size={16} aria-hidden />
            ) : (
              <Plane size={16} aria-hidden />
            )}
            Request specialist
          </button>
          <p className="small-note">
            AI replies pause once the server switches this thread out of AI mode.
          </p>
        </section>

        <section className="rail-section activity-panel">
          <div className="section-heading">
            <Wifi size={18} aria-hidden />
            <h2>Activity</h2>
          </div>
          {error ? (
            <p className="error-line">
              <AlertCircle size={16} aria-hidden />
              {error}
            </p>
          ) : null}
          {actionState ? (
            <p className="busy-line">
              <LoaderCircle className="spin" size={16} aria-hidden />
              {actionState.label}
            </p>
          ) : null}
          <ol className="activity-list">
            {activity.length ? (
              activity.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)
            ) : (
              <li>Waiting for chat activity</li>
            )}
          </ol>
        </section>
      </aside>
    </main>
  );
}
