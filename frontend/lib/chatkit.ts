export type ChatKitElement = HTMLElement & {
  setOptions?: (options: Record<string, unknown>) => void;
  setThreadId?: (threadId: string | null) => void;
  fetchUpdates?: () => void | Promise<void>;
  sendCustomAction?: (action: {
    type: string;
    payload?: Record<string, unknown>;
  }) => void | Promise<void>;
};

export async function waitForChatKitElement(): Promise<void> {
  if (!("customElements" in window)) {
    return;
  }
  await window.customElements.whenDefined("openai-chatkit");
}

export function secureChatKitFetch(
  getClientSecret: () => string | null,
): typeof fetch {
  return async (input, init = {}) => {
    const headers = new Headers(init.headers);
    const clientSecret = getClientSecret();
    if (clientSecret) {
      headers.set("Authorization", `Bearer ${clientSecret}`);
    }

    return fetch(input, {
      ...init,
      credentials: "include",
      headers,
    });
  };
}
