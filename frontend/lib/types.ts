export type ActorRole = "customer" | "agent";

export type SessionResponse = {
  client_secret: string;
  expires_at: string;
  user_id: string;
  session_id: string;
  role: ActorRole;
};

export type ThreadSummary = {
  thread_id: string;
  title: string | null;
  conversation_mode: "ai" | "handoff_requested" | "human" | string;
  assigned_agent_name: string | null;
  claimed_at: string | null;
  last_message_preview: string | null;
  last_updated_at: string | null;
};

export type AgentThreadItem = {
  id?: string;
  type?: string;
  content?: Array<Record<string, unknown>>;
  copy_text?: string;
  created_at?: string;
  [key: string]: unknown;
};

export type AgentThreadResponse = {
  thread: {
    id: string;
    title?: string | null;
    metadata?: Record<string, unknown>;
    [key: string]: unknown;
  };
  items: AgentThreadItem[];
};

export type ChatKitWidgetAction = {
  type?: string;
  payload?: Record<string, unknown>;
  action?: {
    type?: string;
    payload?: Record<string, unknown>;
  };
  [key: string]: unknown;
};

export type RealtimeEvent = {
  type: string;
  payload?: Record<string, unknown>;
};
