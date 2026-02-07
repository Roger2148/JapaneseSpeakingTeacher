import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import SettingsPanel from "./components/SettingsPanel";
import { Message, UserSettings } from "./types";

const SETTINGS_STORAGE_KEY = "jst.settings";
const LAST_USERNAME_KEY = "jst.last_username";
const DEFAULT_STARTER_TEXT =
  "こんにちは。日本語の会話練習を始めましょう。Speak in Japanese or English.";

type WsStatus = "disconnected" | "connecting" | "connected" | "error";
type SttStatus = "idle" | "transcribing" | "done" | "error";
type LlmStatus = "idle" | "generating" | "done" | "error";
type TtsStatus = "idle" | "synthesizing" | "done" | "error";
type AuthStatus = "checking" | "guest" | "authed";
type PendingVoiceCapture = {
  blob: Blob;
  durationSec: number;
  sttText: string;
};

type HistorySummary = {
  id: string;
  title: string;
  created_at: string;
  message_count: number;
};

const defaultSettings: UserSettings = {
  tutorStyle: "balanced",
  replyLanguage: "jp_en",
  correctionIntensity: "medium",
  responseLength: "short",
  showLiveTranscript: true,
  autoPlayAssistantVoice: true,
  showStatusPanel: false
};

const newId = () => `msg_${Date.now()}_${Math.random().toString(16).slice(2)}`;

const createStarterMessage = (text: string): Message => ({
  id: newId(),
  role: "assistant",
  text: text.trim() || DEFAULT_STARTER_TEXT,
  createdAt: new Date().toISOString()
});

function readSettings(): UserSettings {
  const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
  if (!raw) {
    return defaultSettings;
  }

  try {
    return { ...defaultSettings, ...(JSON.parse(raw) as Partial<UserSettings>) };
  } catch {
    return defaultSettings;
  }
}

function formatBytes(bytes: number | null): string {
  if (!bytes || bytes <= 0) {
    return "-";
  }
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function formatDuration(seconds: number): string {
  const safe = Math.max(0, seconds);
  const mm = Math.floor(safe / 60)
    .toString()
    .padStart(2, "0");
  const ss = Math.round(safe % 60)
    .toString()
    .padStart(2, "0");
  return `${mm}:${ss}`;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const slice = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...slice);
  }
  return btoa(binary);
}

function buildWsUrl(): string {
  const explicitUrl = import.meta.env.VITE_API_WS_URL as string | undefined;
  if (explicitUrl) {
    return explicitUrl;
  }

  const baseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (baseUrl) {
    const parsed = new URL(baseUrl);
    const protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${parsed.host}/ws/audio`;
  }

  const parsed = new URL(buildApiBaseUrl());
  const protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${parsed.host}/ws/audio`;
}

function buildApiBaseUrl(): string {
  const explicitUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (explicitUrl) {
    return explicitUrl;
  }

  const explicitPort = String(import.meta.env.VITE_API_PORT ?? "").trim();
  const defaultPort = window.location.protocol === "https:" ? "8443" : "8000";
  const port = /^\d+$/.test(explicitPort) ? explicitPort : defaultPort;
  return `${window.location.protocol}//${window.location.hostname}:${port}`;
}

function toAbsoluteApiUrl(apiBaseUrl: string, maybeRelativeUrl: string): string {
  if (/^https?:\/\//i.test(maybeRelativeUrl)) {
    return maybeRelativeUrl;
  }
  if (maybeRelativeUrl.startsWith("/")) {
    return `${apiBaseUrl}${maybeRelativeUrl}`;
  }
  return `${apiBaseUrl}/${maybeRelativeUrl}`;
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result;
      if (typeof result !== "string") {
        reject(new Error("Failed to encode blob."));
        return;
      }
      const marker = "base64,";
      const markerIndex = result.indexOf(marker);
      if (markerIndex < 0) {
        reject(new Error("Unexpected data URL format."));
        return;
      }
      resolve(result.slice(markerIndex + marker.length));
    };
    reader.onerror = () => {
      reject(new Error("Failed to read blob."));
    };
    reader.readAsDataURL(blob);
  });
}

function displayHistoryTitle(item: HistorySummary): string {
  const clean = (item.title ?? "").trim();
  if (clean) {
    return clean;
  }
  if (item.created_at) {
    return `Conversation ${new Date(item.created_at).toLocaleString()}`;
  }
  return "Untitled conversation";
}

function getMicSupportDetail(): string {
  if (typeof navigator === "undefined") {
    return "Microphone APIs are not available in this environment.";
  }
  if (!window.isSecureContext) {
    return "Microphone is blocked on insecure HTTP. On iPhone/Safari, open this app via HTTPS.";
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "This browser does not expose getUserMedia for microphone capture.";
  }
  if (typeof MediaRecorder === "undefined") {
    return "This browser does not support MediaRecorder yet.";
  }
  return "Microphone recording is not supported in this browser.";
}

export default function App() {
  const [authStatus, setAuthStatus] = useState<AuthStatus>("checking");
  const [authUsername, setAuthUsername] = useState("");
  const [authInput, setAuthInput] = useState(
    () => localStorage.getItem(LAST_USERNAME_KEY) ?? ""
  );
  const [authError, setAuthError] = useState("");
  const [settings, setSettings] = useState<UserSettings>(readSettings);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSaveOpen, setIsSaveOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [historyItems, setHistoryItems] = useState<HistorySummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [isMicBusy, setIsMicBusy] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [lastAudioBytes, setLastAudioBytes] = useState<number | null>(null);
  const [lastMimeType, setLastMimeType] = useState("-");
  const [audioPreviewUrl, setAudioPreviewUrl] = useState("");
  const [micPermission, setMicPermission] = useState<PermissionState | "unknown">(
    "unknown"
  );
  const [micError, setMicError] = useState("");
  const [backendWsStatus, setBackendWsStatus] = useState<WsStatus>("disconnected");
  const [backendSessionId, setBackendSessionId] = useState("-");
  const [backendEvent, setBackendEvent] = useState("-");
  const [backendChunks, setBackendChunks] = useState(0);
  const [backendBytes, setBackendBytes] = useState<number | null>(null);
  const [sttStatus, setSttStatus] = useState<SttStatus>("idle");
  const [lastTranscript, setLastTranscript] = useState("-");
  const [sttLatencyMs, setSttLatencyMs] = useState<number | null>(null);
  const [llmStatus, setLlmStatus] = useState<LlmStatus>("idle");
  const [llmLatencyMs, setLlmLatencyMs] = useState<number | null>(null);
  const [llmModelName, setLlmModelName] = useState("-");
  const [ttsStatus, setTtsStatus] = useState<TtsStatus>("idle");
  const [ttsLatencyMs, setTtsLatencyMs] = useState<number | null>(null);
  const [ttsProvider, setTtsProvider] = useState("-");
  const [ttsVoice, setTtsVoice] = useState("-");
  const [liveTranscript, setLiveTranscript] = useState("");
  const [pendingVoiceCapture, setPendingVoiceCapture] =
    useState<PendingVoiceCapture | null>(null);
  const [isDebugOpen, setIsDebugOpen] = useState(false);
  const [playingMessageId, setPlayingMessageId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [starterText, setStarterText] = useState(DEFAULT_STARTER_TEXT);
  const [messages, setMessages] = useState<Message[]>([
    createStarterMessage(DEFAULT_STARTER_TEXT)
  ]);
  const [isAssistantTyping, setIsAssistantTyping] = useState(false);
  const [saveStatus, setSaveStatus] = useState("");
  const threadRef = useRef<HTMLElement | null>(null);
  const composerInputRef = useRef<HTMLInputElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const recordingStartedAtRef = useRef<number>(0);
  const audioPreviewUrlRef = useRef("");
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const waitingForRecordingSummaryRef = useRef(false);
  const chunkSendChainRef = useRef<Promise<void>>(Promise.resolve());
  const recordingStartedAckResolverRef = useRef<((ok: boolean) => void) | null>(
    null
  );
  const partialTranscriptTimerRef = useRef<number | null>(null);
  const partialRequestInFlightRef = useRef(false);
  const partialRequestIdRef = useRef(0);
  const latestPartialResponseIdRef = useRef(0);
  const chunkCountRef = useRef(0);
  const playingMessageIdRef = useRef<string | null>(null);
  const messageAudioUrlsRef = useRef<string[]>([]);
  const messageAudioElementMapRef = useRef<Record<string, HTMLAudioElement | null>>({});
  const wsUrl = useMemo(() => buildWsUrl(), []);
  const apiBaseUrl = useMemo(() => buildApiBaseUrl(), []);

  const micSupported =
    typeof navigator !== "undefined" &&
    typeof MediaRecorder !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia;
  const micSupportDetail = useMemo(() => getMicSupportDetail(), []);
  const isAuthed = authStatus === "authed";

  const settleRecordingStartAck = useCallback((ok: boolean) => {
    const resolver = recordingStartedAckResolverRef.current;
    if (resolver) {
      recordingStartedAckResolverRef.current = null;
      resolver(ok);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  }, [settings]);

  useEffect(() => {
    audioPreviewUrlRef.current = audioPreviewUrl;
  }, [audioPreviewUrl]);

  useEffect(() => {
    if (!navigator.permissions?.query) {
      return;
    }

    let active = true;
    navigator.permissions
      .query({ name: "microphone" as PermissionName })
      .then((status) => {
        if (!active) {
          return;
        }
        setMicPermission(status.state);
        status.onchange = () => setMicPermission(status.state);
      })
      .catch(() => {
        setMicPermission("unknown");
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const thread = threadRef.current;
    if (!thread) {
      return;
    }
    thread.scrollTop = thread.scrollHeight;
  }, [messages, isAssistantTyping]);

  useEffect(() => {
    chunkCountRef.current = chunkCount;
  }, [chunkCount]);

  useEffect(() => {
    if (!micSupported) {
      setMicError(micSupportDetail);
    }
  }, [micSupported, micSupportDetail]);

  useEffect(() => {
    playingMessageIdRef.current = playingMessageId;
  }, [playingMessageId]);

  const closeWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    waitingForRecordingSummaryRef.current = false;
    settleRecordingStartAck(false);
    partialRequestInFlightRef.current = false;
    setBackendWsStatus("disconnected");
    setBackendSessionId("-");
  }, [settleRecordingStartAck]);

  const resetConversationToStarter = useCallback(
    (text?: string) => {
      const resolvedText = (text ?? starterText).trim() || DEFAULT_STARTER_TEXT;
      setStarterText(resolvedText);
      setMessages([createStarterMessage(resolvedText)]);
      setDraft("");
      setPendingVoiceCapture(null);
      setLastTranscript("-");
      setLiveTranscript("");
      setIsAssistantTyping(false);
      setPlayingMessageId(null);
    },
    [starterText]
  );

  const fetchWelcomeStarterText = useCallback(async (): Promise<string> => {
    try {
      const response = await fetch(`${apiBaseUrl}/topics/welcome`, {
        method: "GET",
        credentials: "include"
      });
      if (!response.ok) {
        return DEFAULT_STARTER_TEXT;
      }
      const payload = (await response.json()) as { message?: string };
      const text =
        typeof payload.message === "string" && payload.message.trim()
          ? payload.message.trim()
          : DEFAULT_STARTER_TEXT;
      setStarterText(text);
      return text;
    } catch {
      return DEFAULT_STARTER_TEXT;
    }
  }, [apiBaseUrl]);

  const handleUnauthorized = useCallback(() => {
    closeWebSocket();
    setAuthStatus("guest");
    setAuthUsername("");
    setHistoryItems([]);
    setHistoryError("");
    setAuthError("Session expired. Please log in again.");
    setStarterText(DEFAULT_STARTER_TEXT);
    resetConversationToStarter(DEFAULT_STARTER_TEXT);
  }, [closeWebSocket, resetConversationToStarter]);

  const loadHistoryList = useCallback(
    async (autoLoadLatest: boolean) => {
      if (!isAuthed) {
        return;
      }
      setHistoryLoading(true);
      setHistoryError("");
      try {
        const response = await fetch(`${apiBaseUrl}/history/list`, {
          method: "GET",
          credentials: "include"
        });
        if (response.status === 401) {
          handleUnauthorized();
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const list = (await response.json()) as HistorySummary[];
        setHistoryItems(Array.isArray(list) ? list : []);

        if (autoLoadLatest && Array.isArray(list) && list.length > 0) {
          const latestId = list[0].id;
          const latestResponse = await fetch(`${apiBaseUrl}/history/${latestId}`, {
            method: "GET",
            credentials: "include"
          });
          if (latestResponse.status === 401) {
            handleUnauthorized();
            return;
          }
          if (latestResponse.ok) {
            const item = (await latestResponse.json()) as {
              messages?: Array<{
                role?: "user" | "assistant";
                text?: string;
                kind?: "text" | "voice";
                sttText?: string;
                createdAt?: string;
              }>;
            };
            const nextMessages =
              Array.isArray(item.messages) && item.messages.length > 0
                ? item.messages
                    .filter(
                      (message) =>
                        (message.role === "user" || message.role === "assistant") &&
                        typeof message.text === "string"
                    )
                    .map((message) => ({
                      id: newId(),
                      role: message.role as "user" | "assistant",
                      text: (message.text ?? "").trim(),
                      kind: (message.kind === "voice" ? "voice" : "text") as
                        | "voice"
                        | "text",
                      sttText: message.sttText,
                      createdAt:
                        message.createdAt && message.createdAt.trim()
                          ? message.createdAt
                          : new Date().toISOString()
                    }))
                    .filter((message) => message.text.length > 0)
                : [];
            if (nextMessages.length > 0) {
              setMessages(nextMessages);
            }
          }
        }
      } catch (error) {
        const detail = error instanceof Error ? error.message : "Unknown error";
        setHistoryError(`History load failed: ${detail}`);
      } finally {
        setHistoryLoading(false);
      }
    },
    [apiBaseUrl, handleUnauthorized, isAuthed]
  );

  const loadHistoryConversation = useCallback(
    async (itemId: string) => {
      if (!isAuthed) {
        return;
      }
      if (!itemId || !itemId.trim()) {
        setSaveStatus("Open failed: invalid conversation id.");
        clearSaveStatusLater();
        return;
      }
      setHistoryLoading(true);
      setHistoryError("");
      try {
        const response = await fetch(`${apiBaseUrl}/history/${itemId}`, {
          method: "GET",
          credentials: "include"
        });
        if (response.status === 401) {
          handleUnauthorized();
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const item = (await response.json()) as {
          messages?: Array<{
            role?: "user" | "assistant";
            text?: string;
            kind?: "text" | "voice";
            sttText?: string;
            createdAt?: string;
          }>;
        };
        const nextMessages =
          Array.isArray(item.messages) && item.messages.length > 0
            ? item.messages
                .filter(
                  (message) =>
                    (message.role === "user" || message.role === "assistant") &&
                    typeof message.text === "string"
                )
                .map((message) => ({
                  id: newId(),
                  role: message.role as "user" | "assistant",
                  text: (message.text ?? "").trim(),
                  kind: (message.kind === "voice" ? "voice" : "text") as
                    | "voice"
                    | "text",
                  sttText: message.sttText,
                  createdAt:
                    message.createdAt && message.createdAt.trim()
                      ? message.createdAt
                      : new Date().toISOString()
                }))
                .filter((message) => message.text.length > 0)
            : [];
        setMessages(
          nextMessages.length > 0
            ? nextMessages
            : [createStarterMessage(starterText)]
        );
        setDraft("");
        setPendingVoiceCapture(null);
        setLastTranscript("-");
        setLiveTranscript("");
        setIsHistoryOpen(false);
        setSaveStatus(
          nextMessages.length > 0
            ? "Loaded saved conversation."
            : "Loaded conversation (no messages found in that record)."
        );
        clearSaveStatusLater();
      } catch (error) {
        const detail = error instanceof Error ? error.message : "Unknown error";
        setHistoryError(`History load failed: ${detail}`);
        setSaveStatus(`Open failed: ${detail}`);
        clearSaveStatusLater();
      } finally {
        setHistoryLoading(false);
      }
    },
    [apiBaseUrl, handleUnauthorized, starterText]
  );

  const deleteHistoryConversation = useCallback(
    async (itemId: string) => {
      if (!isAuthed) {
        return;
      }
      setHistoryLoading(true);
      setHistoryError("");
      try {
        const response = await fetch(`${apiBaseUrl}/history/${itemId}`, {
          method: "DELETE",
          credentials: "include"
        });
        if (response.status === 401) {
          handleUnauthorized();
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        setHistoryItems((current) => current.filter((item) => item.id !== itemId));
        setSaveStatus("Deleted saved conversation.");
        clearSaveStatusLater();
      } catch (error) {
        const detail = error instanceof Error ? error.message : "Unknown error";
        setHistoryError(`History delete failed: ${detail}`);
      } finally {
        setHistoryLoading(false);
      }
    },
    [apiBaseUrl, handleUnauthorized, isAuthed]
  );

  const checkAuthSession = useCallback(async () => {
    setAuthStatus("checking");
    setAuthError("");
    try {
      const response = await fetch(`${apiBaseUrl}/auth/me`, {
        method: "GET",
        credentials: "include"
      });
      if (response.status === 401) {
        setAuthStatus("guest");
        return;
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = (await response.json()) as { username?: string };
      if (!data.username) {
        setAuthStatus("guest");
        return;
      }
      setAuthUsername(data.username);
      setAuthStatus("authed");
      const welcomeText = await fetchWelcomeStarterText();
      resetConversationToStarter(welcomeText);
      void loadHistoryList(true);
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown error";
      setAuthError(`Auth check failed: ${detail}`);
      setAuthStatus("guest");
    }
  }, [apiBaseUrl, fetchWelcomeStarterText, loadHistoryList, resetConversationToStarter]);

  const loginWithUsername = useCallback(async () => {
    const username = authInput.trim();
    if (!username) {
      setAuthError("Please enter a username.");
      return;
    }
    setAuthStatus("checking");
    setAuthError("");
    try {
      const response = await fetch(`${apiBaseUrl}/auth/login`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ username })
      });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const payload = (await response.json()) as { detail?: string };
          if (payload.detail) {
            detail = payload.detail;
          }
        } catch {
          // keep fallback detail
        }
        throw new Error(detail);
      }
      const data = (await response.json()) as { username?: string };
      const resolvedUsername = data.username?.trim() || username;
      localStorage.setItem(LAST_USERNAME_KEY, resolvedUsername);
      setAuthUsername(resolvedUsername);
      setAuthInput(resolvedUsername);
      setAuthStatus("authed");
      const welcomeText = await fetchWelcomeStarterText();
      resetConversationToStarter(welcomeText);
      void loadHistoryList(true);
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown error";
      setAuthError(`Login failed: ${detail}`);
      setAuthStatus("guest");
    }
  }, [
    apiBaseUrl,
    authInput,
    fetchWelcomeStarterText,
    loadHistoryList,
    resetConversationToStarter
  ]);

  const logout = useCallback(async () => {
    try {
      await fetch(`${apiBaseUrl}/auth/logout`, {
        method: "POST",
        credentials: "include"
      });
    } catch {
      // best-effort logout
    }
    closeWebSocket();
    setAuthStatus("guest");
    setAuthUsername("");
    setHistoryItems([]);
    setIsHistoryOpen(false);
    resetConversationToStarter();
  }, [apiBaseUrl, closeWebSocket, resetConversationToStarter]);

  useEffect(() => {
    void checkAuthSession();
    // Initial auth bootstrap only; avoid a request loop from callback identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const connectWebSocket = useCallback(() => {
    if (!isAuthed) {
      return;
    }
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    setBackendWsStatus("connecting");
    setBackendSessionId("-");
    setBackendEvent("connecting");
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      setBackendWsStatus("connected");
      setBackendEvent("connected");
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data as string) as Record<string, unknown>;
        const eventType =
          typeof payload.type === "string" ? payload.type : "unknown_event";
        if (eventType === "server_ready" && typeof payload.session_id === "string") {
          setBackendSessionId(payload.session_id);
          setBackendEvent(eventType);
          return;
        }
        if (eventType === "chunk_received") {
          if (typeof payload.total_chunks === "number") {
            setBackendChunks(payload.total_chunks);
          }
          if (typeof payload.total_bytes === "number") {
            setBackendBytes(payload.total_bytes);
          }
          if (!waitingForRecordingSummaryRef.current) {
            setBackendEvent(eventType);
          }
          return;
        }
        if (eventType === "recording_summary") {
          waitingForRecordingSummaryRef.current = false;
          partialRequestInFlightRef.current = false;
          if (typeof payload.total_chunks === "number") {
            setBackendChunks(payload.total_chunks);
          }
          if (typeof payload.total_bytes === "number") {
            setBackendBytes(payload.total_bytes);
          }
          setBackendEvent(eventType);
          return;
        }
        if (eventType === "recording_started_ack") {
          settleRecordingStartAck(true);
          setBackendEvent(eventType);
          return;
        }
        if (eventType === "transcription_started") {
          setSttStatus("transcribing");
          setSttLatencyMs(null);
          setBackendEvent(eventType);
          return;
        }
        if (eventType === "transcription_result") {
          partialRequestInFlightRef.current = false;
          const text =
            typeof payload.text === "string" ? payload.text.trim() : "";
          if (text) {
            setDraft(text);
            setLastTranscript(text);
            setPendingVoiceCapture((current) =>
              current ? { ...current, sttText: text } : current
            );
          } else {
            setLastTranscript("(empty transcript)");
          }
          if (typeof payload.latency_ms === "number") {
            setSttLatencyMs(payload.latency_ms);
          }
          setSttStatus("done");
          setMicError("");
          setBackendEvent(eventType);
          return;
        }
        if (eventType === "partial_transcription_result") {
          partialRequestInFlightRef.current = false;
          const requestId =
            typeof payload.request_id === "number" ? payload.request_id : 0;
          if (requestId < latestPartialResponseIdRef.current) {
            return;
          }
          latestPartialResponseIdRef.current = requestId;
          const text =
            typeof payload.text === "string" ? payload.text.trim() : "";
          setLiveTranscript(text);
          if (text) {
            setLastTranscript(text);
          }
          setBackendEvent(eventType);
          return;
        }
        if (eventType === "partial_transcription_error") {
          partialRequestInFlightRef.current = false;
          setBackendEvent(eventType);
          return;
        }
        if (eventType === "transcription_error") {
          partialRequestInFlightRef.current = false;
          setSttStatus("error");
          if (typeof payload.detail === "string") {
            setMicError(`STT error: ${payload.detail}`);
          }
          setBackendEvent(eventType);
          return;
        }
        if (eventType === "error") {
          settleRecordingStartAck(false);
        }
        setBackendEvent(eventType);
      } catch {
        setBackendEvent("invalid_json");
      }
    };

    ws.onerror = () => {
      setBackendWsStatus("error");
      setBackendEvent("socket_error");
    };

    ws.onclose = () => {
      wsRef.current = null;
      waitingForRecordingSummaryRef.current = false;
      settleRecordingStartAck(false);
      partialRequestInFlightRef.current = false;
      setBackendWsStatus("disconnected");
      setBackendSessionId("-");
      setBackendEvent("disconnected");
    };

    wsRef.current = ws;
  }, [isAuthed, settleRecordingStartAck, wsUrl]);

  const sendWsJson = useCallback(
    (payload: Record<string, unknown>): boolean => {
      const socket = wsRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        setBackendEvent("ws_not_ready");
        connectWebSocket();
        return false;
      }
      socket.send(JSON.stringify(payload));
      return true;
    },
    [connectWebSocket]
  );

  useEffect(() => {
    if (!isAuthed) {
      closeWebSocket();
      return;
    }
    connectWebSocket();
    return () => {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
      }
      if (partialTranscriptTimerRef.current !== null) {
        window.clearInterval(partialTranscriptTimerRef.current);
      }
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      partialRequestInFlightRef.current = false;
      chunkSendChainRef.current = Promise.resolve();
      if (sourceNodeRef.current) {
        sourceNodeRef.current.disconnect();
      }
      if (analyserRef.current) {
        analyserRef.current.disconnect();
      }
      if (audioContextRef.current) {
        void audioContextRef.current.close();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (audioPreviewUrlRef.current) {
        URL.revokeObjectURL(audioPreviewUrlRef.current);
      }
      for (const url of messageAudioUrlsRef.current) {
        URL.revokeObjectURL(url);
      }
      messageAudioUrlsRef.current = [];
      messageAudioElementMapRef.current = {};
      closeWebSocket();
    };
  }, [closeWebSocket, connectWebSocket, isAuthed]);

  const reconnectWebSocket = () => {
    closeWebSocket();
    window.setTimeout(() => {
      connectWebSocket();
    }, 120);
  };

  const sleep = (ms: number) =>
    new Promise<void>((resolve) => {
      window.setTimeout(resolve, ms);
    });

  const ensureWebSocketReady = useCallback(
    async (timeoutMs = 3000): Promise<boolean> => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        return true;
      }
      connectWebSocket();
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          return true;
        }
        await sleep(50);
      }
      return false;
    },
    [connectWebSocket]
  );

  const requestRecordingStart = useCallback(
    async (mimeType: string): Promise<boolean> => {
      const ready = await ensureWebSocketReady();
      if (!ready) {
        return false;
      }

      settleRecordingStartAck(false);

      const ackPromise = new Promise<boolean>((resolve) => {
        recordingStartedAckResolverRef.current = resolve;
        window.setTimeout(() => {
          if (recordingStartedAckResolverRef.current === resolve) {
            settleRecordingStartAck(false);
          }
        }, 2500);
      });

      waitingForRecordingSummaryRef.current = false;
      const sent = sendWsJson({
        type: "recording_started",
        mime_type: mimeType,
        timeslice_ms: 250
      });
      if (!sent) {
        settleRecordingStartAck(false);
        return false;
      }

      return ackPromise;
    },
    [ensureWebSocketReady, sendWsJson, settleRecordingStartAck]
  );

  const requestPartialTranscript = useCallback(() => {
    if (!isRecording) {
      return;
    }
    if (partialRequestInFlightRef.current) {
      return;
    }
    if (chunkCountRef.current <= 0) {
      return;
    }

    partialRequestInFlightRef.current = true;
    const requestId = partialRequestIdRef.current + 1;
    partialRequestIdRef.current = requestId;
    const sent = sendWsJson({
      type: "transcription_partial_request",
      request_id: requestId
    });
    if (!sent) {
      partialRequestInFlightRef.current = false;
    }
  }, [isRecording, sendWsJson]);

  useEffect(() => {
    if (partialTranscriptTimerRef.current !== null) {
      window.clearInterval(partialTranscriptTimerRef.current);
      partialTranscriptTimerRef.current = null;
    }

    if (!isRecording || !settings.showLiveTranscript) {
      partialRequestInFlightRef.current = false;
      return;
    }

    partialRequestInFlightRef.current = false;
    latestPartialResponseIdRef.current = 0;
    setLiveTranscript("");

    const kickoff = window.setTimeout(() => {
      requestPartialTranscript();
    }, 450);
    partialTranscriptTimerRef.current = window.setInterval(() => {
      requestPartialTranscript();
    }, 1200);

    return () => {
      window.clearTimeout(kickoff);
      if (partialTranscriptTimerRef.current !== null) {
        window.clearInterval(partialTranscriptTimerRef.current);
        partialTranscriptTimerRef.current = null;
      }
      partialRequestInFlightRef.current = false;
    };
  }, [isRecording, requestPartialTranscript, settings.showLiveTranscript]);

  const stopRecordingTimer = () => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  const stopAudioMeter = () => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }
    if (audioContextRef.current) {
      void audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setAudioLevel(0);
  };

  const stopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const startAudioMeter = (stream: MediaStream) => {
    const BrowserAudioContext =
      window.AudioContext ||
      (window as Window & { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;

    if (!BrowserAudioContext) {
      return;
    }

    const context = new BrowserAudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 256;
    const sourceNode = context.createMediaStreamSource(stream);
    sourceNode.connect(analyser);

    audioContextRef.current = context;
    analyserRef.current = analyser;
    sourceNodeRef.current = sourceNode;
    const bins = new Uint8Array(analyser.frequencyBinCount);

    const tick = () => {
      analyser.getByteFrequencyData(bins);
      let total = 0;
      for (const value of bins) {
        total += value;
      }
      const average = bins.length > 0 ? total / bins.length : 0;
      setAudioLevel(average / 255);
      animationFrameRef.current = window.requestAnimationFrame(tick);
    };

    animationFrameRef.current = window.requestAnimationFrame(tick);
  };

  const startRecording = async () => {
    if (!isAuthed) {
      setSaveStatus("Please log in first.");
      clearSaveStatusLater();
      return;
    }
    if (!micSupported) {
      setMicError("Microphone recording is not supported in this browser.");
      return;
    }

    setIsMicBusy(true);
    setMicError("");
    setSaveStatus("");
    setPendingVoiceCapture(null);
    setLiveTranscript("");
    partialRequestIdRef.current = 0;
    latestPartialResponseIdRef.current = 0;
    partialRequestInFlightRef.current = false;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      streamRef.current = stream;
      setMicPermission("granted");
      startAudioMeter(stream);

      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
        "audio/ogg;codecs=opus"
      ];
      const mimeType = mimeCandidates.find((item) =>
        MediaRecorder.isTypeSupported(item)
      );
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      const normalizedMimeType = recorder.mimeType || mimeType || "default";
      const started = await requestRecordingStart(normalizedMimeType);
      if (!started) {
        setMicError("Backend recording session not ready. Please retry.");
        stopAudioMeter();
        stopStream();
        return;
      }

      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];
      setChunkCount(0);
      setLastMimeType(normalizedMimeType);
      setLastAudioBytes(null);

      setSttStatus("idle");
      setSttLatencyMs(null);

      recorder.ondataavailable = (event) => {
        if (event.data.size <= 0) {
          return;
        }

        audioChunksRef.current.push(event.data);
        setChunkCount((current) => current + 1);

        chunkSendChainRef.current = chunkSendChainRef.current
          .then(async () => {
            const buffer = await event.data.arrayBuffer();
            const sent = sendWsJson({
              type: "audio_chunk",
              chunk_b64: arrayBufferToBase64(buffer),
              chunk_size: buffer.byteLength
            });
            if (!sent) {
              throw new Error("ws_not_ready");
            }
          })
          .catch(() => {
            setBackendEvent("chunk_send_failed");
          });
      };

      recorder.onstop = () => {
        const duration =
          recordingStartedAtRef.current > 0
            ? (Date.now() - recordingStartedAtRef.current) / 1000
            : 0;
        const blob = new Blob(audioChunksRef.current, {
          type: recorder.mimeType || "audio/webm"
        });

        setRecordingSeconds(duration);
        setLastAudioBytes(blob.size);
        if (audioPreviewUrlRef.current) {
          URL.revokeObjectURL(audioPreviewUrlRef.current);
        }
        const url = URL.createObjectURL(blob);
        setAudioPreviewUrl(url);
        setDraft("");
        setPendingVoiceCapture({
          blob,
          durationSec: duration,
          sttText: ""
        });

        const sendStoppedEvent = () => {
          setSttStatus("transcribing");
          const sent = sendWsJson({
            type: "recording_stopped",
            duration_sec: Number(duration.toFixed(3)),
            client_chunk_count: audioChunksRef.current.length,
            client_bytes: blob.size
          });
          waitingForRecordingSummaryRef.current = sent;
        };

        void chunkSendChainRef.current
          .then(() => {
            sendStoppedEvent();
          })
          .catch(() => {
            setBackendEvent("chunk_send_failed");
            sendStoppedEvent();
          });
      };

      recorder.onerror = () => {
        setMicError("MediaRecorder error happened while recording.");
      };

      recorder.start(250);
      recordingStartedAtRef.current = Date.now();
      setRecordingSeconds(0);
      setIsRecording(true);
      stopRecordingTimer();
      timerRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - recordingStartedAtRef.current) / 1000;
        setRecordingSeconds(elapsed);
      }, 100);
    } catch (error) {
      stopAudioMeter();
      stopStream();
      const asError = error as Error & { name?: string };
      if (asError.name === "NotAllowedError") {
        setMicPermission("denied");
        setMicError("Microphone permission denied. Please allow microphone access.");
      } else if (asError.name === "NotFoundError") {
        setMicError("No microphone input device was found.");
      } else {
        setMicError("Could not start recording. Please retry.");
      }
    } finally {
      setIsMicBusy(false);
    }
  };

  const stopRecording = () => {
    stopRecordingTimer();
    setIsRecording(false);

    const recorder = mediaRecorderRef.current;
    mediaRecorderRef.current = null;
    if (recorder && recorder.state !== "inactive") {
      recorder.addEventListener(
        "stop",
        () => {
          stopAudioMeter();
          stopStream();
        },
        { once: true }
      );
      recorder.stop();
      return;
    }
    stopAudioMeter();
    stopStream();
  };

  const clearSaveStatusLater = () => {
    window.setTimeout(() => {
      setSaveStatus("");
    }, 2000);
  };

  const exportConversationAsText = () => {
    const content = messages
      .map((message) => {
        const time = new Date(message.createdAt).toLocaleString();
        return `[${time}] ${message.role.toUpperCase()}: ${message.text}`;
      })
      .join("\n");
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const timestamp = new Date().toISOString().replace(/:/g, "-");
    link.href = url;
    link.download = `conversation-${timestamp}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setSaveStatus("Saved text export.");
    clearSaveStatusLater();
  };

  const saveConversationToHistoryPool = () => {
    if (!isAuthed) {
      setSaveStatus("Please log in before saving conversation.");
      clearSaveStatusLater();
      return;
    }
    const save = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/history/save`, {
          method: "POST",
          credentials: "include",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            messages: messages.map((message) => ({
              role: message.role,
              text: message.text,
              kind: message.kind ?? "text",
              sttText: message.sttText,
              createdAt: message.createdAt
            }))
          })
        });
        if (response.status === 401) {
          handleUnauthorized();
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        setSaveStatus("Saved to server history.");
        clearSaveStatusLater();
        void loadHistoryList(false);
      } catch (error) {
        const detail = error instanceof Error ? error.message : "Unknown error";
        setSaveStatus(`Save failed: ${detail}`);
        clearSaveStatusLater();
      }
    };
    void save();
  };

  const saveAudioPackage = () => {
    if (!isAuthed) {
      setSaveStatus("Please log in before exporting audio package.");
      clearSaveStatusLater();
      return;
    }
    const run = async () => {
      try {
        setSaveStatus("Preparing audio package...");
        const exportMessages: Array<{
          role: "user" | "assistant";
          text: string;
          audio_b64?: string;
          audio_mime?: string;
        }> = [];

        for (const message of messages) {
          const entry: {
            role: "user" | "assistant";
            text: string;
            audio_b64?: string;
            audio_mime?: string;
          } = {
            role: message.role,
            text: message.text
          };

          if (message.audioUrl) {
            try {
              const audioResponse = await fetch(message.audioUrl, {
                credentials: "include"
              });
              if (audioResponse.ok) {
                const blob = await audioResponse.blob();
                if (blob.size > 0) {
                  entry.audio_b64 = await blobToBase64(blob);
                  entry.audio_mime = blob.type || undefined;
                }
              }
            } catch {
              // best-effort: skip missing audio clip
            }
          }
          exportMessages.push(entry);
        }

        const response = await fetch(`${apiBaseUrl}/export/audio-package`, {
          method: "POST",
          credentials: "include",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            title: `conversation-${new Date().toISOString()}`,
            messages: exportMessages
          })
        });
        if (response.status === 401) {
          handleUnauthorized();
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const fileBlob = await response.blob();
        const fileUrl = URL.createObjectURL(fileBlob);
        const link = document.createElement("a");
        const stamp = new Date().toISOString().replace(/:/g, "-");
        link.href = fileUrl;
        link.download = `conversation-audio-${stamp}.zip`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(fileUrl);
        setSaveStatus("Saved audio package (.zip).");
        clearSaveStatusLater();
      } catch (error) {
        const detail = error instanceof Error ? error.message : "Unknown error";
        setSaveStatus(`Audio export failed: ${detail}`);
        clearSaveStatusLater();
      }
    };
    void run();
  };

  const statusText = isRecording ? "Recording" : "Idle";

  const playMessageAudio = useCallback((messageId: string) => {
    const audio = messageAudioElementMapRef.current[messageId];
    if (!audio) {
      return;
    }

    const currentlyPlayingId = playingMessageIdRef.current;
    if (currentlyPlayingId && currentlyPlayingId !== messageId) {
      const previousAudio = messageAudioElementMapRef.current[currentlyPlayingId];
      if (previousAudio) {
        previousAudio.pause();
        previousAudio.currentTime = 0;
      }
    }

    audio.currentTime = 0;
    void audio.play().then(
      () => {
        setPlayingMessageId(messageId);
      },
      () => {
        setPlayingMessageId(null);
        setSaveStatus("Audio playback failed in this browser.");
        clearSaveStatusLater();
      }
    );
  }, []);

  const togglePlayMessageAudio = (messageId: string) => {
    const audio = messageAudioElementMapRef.current[messageId];
    if (!audio) {
      return;
    }

    if (playingMessageIdRef.current === messageId && !audio.paused) {
      audio.pause();
      audio.currentTime = 0;
      setPlayingMessageId(null);
      return;
    }

    playMessageAudio(messageId);
  };

  const synthesizeAssistantAudio = useCallback(
    async (messageId: string, text: string) => {
      setTtsStatus("synthesizing");
      setTtsLatencyMs(null);
      setBackendEvent("tts_synthesizing");

      try {
        const response = await fetch(`${apiBaseUrl}/tts`, {
          method: "POST",
          credentials: "include",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            text,
            reply_language: settings.replyLanguage
          })
        });

        if (!response.ok) {
          if (response.status === 401) {
            handleUnauthorized();
            return;
          }
          let detail = `HTTP ${response.status}`;
          try {
            const errorBody = (await response.json()) as { detail?: string };
            if (typeof errorBody.detail === "string" && errorBody.detail.trim()) {
              detail = errorBody.detail.trim();
            }
          } catch {
            // keep HTTP code fallback
          }
          throw new Error(detail);
        }

        const data = (await response.json()) as {
          audio_url?: string;
          provider?: string;
          voice?: string;
          latency_ms?: number;
        };
        if (!data.audio_url || !data.audio_url.trim()) {
          throw new Error("Missing audio_url in TTS response.");
        }
        const audioUrl = toAbsoluteApiUrl(apiBaseUrl, data.audio_url.trim());
        setMessages((current) =>
          current.map((message) =>
            message.id === messageId
              ? {
                  ...message,
                  kind: "voice",
                  audioUrl
                }
              : message
          )
        );
        if (typeof data.latency_ms === "number") {
          setTtsLatencyMs(data.latency_ms);
        }
        if (typeof data.provider === "string" && data.provider.trim()) {
          setTtsProvider(data.provider.trim());
        }
        if (typeof data.voice === "string" && data.voice.trim()) {
          setTtsVoice(data.voice.trim());
        }
        setTtsStatus("done");
        setBackendEvent("tts_result");

        if (settings.autoPlayAssistantVoice) {
          window.setTimeout(() => {
            playMessageAudio(messageId);
          }, 140);
        }
      } catch (error) {
        const detail =
          error instanceof Error && error.message ? error.message : "Unknown error";
        setTtsStatus("error");
        setBackendEvent("tts_error");
        setSaveStatus(`TTS unavailable: ${detail}`);
        clearSaveStatusLater();
      }
    },
    [
      apiBaseUrl,
      handleUnauthorized,
      playMessageAudio,
      settings.autoPlayAssistantVoice,
      settings.replyLanguage
    ]
  );

  const submitComposerMessage = useCallback(async () => {
    if (!isAuthed) {
      setSaveStatus("Please log in first.");
      clearSaveStatusLater();
      return;
    }
    const fallbackVoiceText = pendingVoiceCapture?.sttText.trim() ?? "";
    const text = draft.trim() || fallbackVoiceText;
    if (!text) {
      setSaveStatus("No transcript yet. Edit text or re-record first.");
      clearSaveStatusLater();
      return;
    }

    const messageId = newId();
    let userMessage: Message;
    if (pendingVoiceCapture) {
      const audioUrl = URL.createObjectURL(pendingVoiceCapture.blob);
      messageAudioUrlsRef.current.push(audioUrl);
      userMessage = {
        id: messageId,
        role: "user",
        kind: "voice",
        text,
        sttText: pendingVoiceCapture.sttText || text,
        audioUrl,
        audioDurationSec: pendingVoiceCapture.durationSec,
        createdAt: new Date().toISOString()
      };
    } else {
      userMessage = {
        id: messageId,
        role: "user",
        kind: "text",
        text,
        createdAt: new Date().toISOString()
      };
    }
    setMessages((current) => [...current, userMessage]);
    setDraft("");
    setPendingVoiceCapture(null);
    setLastTranscript("-");
    setLiveTranscript("");
    setIsAssistantTyping(true);
    setLlmStatus("generating");
    setLlmLatencyMs(null);
    setTtsStatus("idle");
    setTtsLatencyMs(null);
    setBackendEvent("llm_generating");

    const historyTurns = [...messages, userMessage].map((turn) => ({
      role: turn.role,
      text: turn.text
    }));

    try {
      const response = await fetch(`${apiBaseUrl}/chat`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          user_text: text,
          settings,
          history: historyTurns
        })
      });

      if (!response.ok) {
        if (response.status === 401) {
          handleUnauthorized();
          return;
        }
        let detail = `HTTP ${response.status}`;
        try {
          const errorBody = (await response.json()) as { detail?: string };
          if (typeof errorBody.detail === "string" && errorBody.detail.trim()) {
            detail = errorBody.detail.trim();
          }
        } catch {
          // fall back to HTTP code detail when parsing fails
        }
        throw new Error(detail);
      }

      const data = (await response.json()) as {
        text?: string;
        model?: string;
        latency_ms?: number;
      };
      const assistantText =
        typeof data.text === "string" && data.text.trim()
          ? data.text.trim()
          : "(Empty LLM response)";
      if (typeof data.latency_ms === "number") {
        setLlmLatencyMs(data.latency_ms);
      }
      if (typeof data.model === "string" && data.model.trim()) {
        setLlmModelName(data.model.trim());
      }

      const assistantMessage: Message = {
        id: newId(),
        role: "assistant",
        text: assistantText,
        createdAt: new Date().toISOString()
      };
      setMessages((current) => [...current, assistantMessage]);
      setLlmStatus("done");
      setBackendEvent("llm_result");
      void synthesizeAssistantAudio(assistantMessage.id, assistantText);
    } catch (error) {
      const detail =
        error instanceof Error && error.message ? error.message : "Unknown error";
      const assistantMessage: Message = {
        id: newId(),
        role: "assistant",
        text: `[LLM unavailable] ${detail}`,
        createdAt: new Date().toISOString()
      };
      setMessages((current) => [...current, assistantMessage]);
      setLlmStatus("error");
      setTtsStatus("idle");
      setBackendEvent("llm_error");
    } finally {
      setIsAssistantTyping(false);
    }
  }, [
    apiBaseUrl,
    draft,
    handleUnauthorized,
    isAuthed,
    messages,
    pendingVoiceCapture,
    settings,
    synthesizeAssistantAudio
  ]);

  const sendMessage = (event: FormEvent) => {
    event.preventDefault();
    void submitComposerMessage();
  };

  const focusComposerInput = () => {
    const input = composerInputRef.current;
    if (!input) {
      return;
    }
    input.focus();
    if (draft.trim()) {
      input.setSelectionRange(0, input.value.length);
    }
  };

  const reRecordAfterCapture = () => {
    if (isRecording || isMicBusy) {
      return;
    }
    setPendingVoiceCapture(null);
    setDraft("");
    setLastTranscript("-");
    setLiveTranscript("");
    setSaveStatus("");
    void startRecording();
  };

  const startNewConversation = () => {
    const run = async () => {
      const welcomeText = await fetchWelcomeStarterText();
      resetConversationToStarter(welcomeText);
      setSaveStatus("Started a new conversation. Saved history is unchanged.");
      clearSaveStatusLater();
    };
    void run();
  };

  const hasPendingVoiceDecision = !!pendingVoiceCapture && !isRecording;
  const pendingDraftText = draft.trim();
  const pendingSttText = pendingVoiceCapture?.sttText.trim() ?? "";
  const canSendPendingCapture =
    (pendingDraftText.length > 0 || pendingSttText.length > 0) &&
    !isAssistantTyping &&
    !isMicBusy;

  if (authStatus === "checking") {
    return (
      <div className="auth-shell">
        <section className="auth-card">
          <h1>Japanese Speaking Teacher</h1>
          <p>Checking session...</p>
        </section>
      </div>
    );
  }

  if (authStatus === "guest") {
    return (
      <div className="auth-shell">
        <section className="auth-card">
          <h1>Japanese Speaking Teacher</h1>
          <p>Create/login with a username (no password).</p>
          <form
            className="auth-form"
            onSubmit={(event) => {
              event.preventDefault();
              void loginWithUsername();
            }}
          >
            <input
              value={authInput}
              onChange={(event) => setAuthInput(event.target.value)}
              placeholder="Username"
              aria-label="Username"
            />
            <button type="submit">Enter</button>
          </form>
          {authError ? <p className="auth-error">{authError}</p> : null}
        </section>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="top-bar">
        <div>
          <p className="eyebrow">Japanese Speaking Teacher</p>
          <h1>Conversation Studio</h1>
        </div>
        <div className="top-actions">
          <span className="user-chip">{authUsername}</span>
          <button className="ghost-button" onClick={startNewConversation}>
            New Chat
          </button>
          <button className="ghost-button" onClick={() => setIsHistoryOpen(true)}>
            History
          </button>
          <button className="ghost-button" onClick={() => setIsSaveOpen(true)}>
            Save
          </button>
          <button className="ghost-button" onClick={() => void logout()}>
            Logout
          </button>
          <button onClick={() => setIsSettingsOpen(true)}>Settings</button>
        </div>
      </header>

      {isRecording && settings.showLiveTranscript ? (
        <section className="live-overlay" aria-live="polite">
          <article className="live-overlay-card">
            <div className="live-overlay-head">
              <p className="live-overlay-label">Live Transcript</p>
              <p className="live-overlay-badge">REC {recordingSeconds.toFixed(1)}s</p>
            </div>
            <div className="live-overlay-body">
              <p className={`live-overlay-text ${liveTranscript ? "" : "placeholder"}`}>
                {liveTranscript || "Listening... keep speaking naturally."}
              </p>
            </div>
            <div className="live-overlay-foot">
              <div className="live-overlay-level-track">
                <div
                  className="live-overlay-level-fill"
                  style={{ width: `${Math.max(3, Math.round(audioLevel * 100))}%` }}
                />
              </div>
              <p>{chunkCount} chunks captured</p>
            </div>
          </article>
          <button
            type="button"
            className="live-overlay-stop"
            onClick={stopRecording}
          >
            Stop Recording ({recordingSeconds.toFixed(1)}s)
          </button>
        </section>
      ) : null}

      <main className={`chat-shell ${settings.showStatusPanel ? "" : "single-pane"}`}>
        {settings.showStatusPanel ? (
          <aside className="control-rail">
            <div className="status-row">
              <p>{statusText}</p>
              <p>
                Session is unsaved by default. Use Save to export or keep conversation.
              </p>
            </div>
            <section className="debug-summary">
              <p>WS: {backendWsStatus}</p>
              <p>Session: {backendSessionId}</p>
              <p>Event: {backendEvent}</p>
              <p>STT: {sttStatus}</p>
              <p>LLM: {llmStatus}</p>
              <p>TTS: {ttsStatus}</p>
              <p>Live: {settings.showLiveTranscript ? "on" : "off"}</p>
              <button
                className="ghost-button"
                type="button"
                onClick={() => setIsDebugOpen((current) => !current)}
              >
                {isDebugOpen ? "Hide Debug" : "Show Debug"}
              </button>
            </section>
            {isDebugOpen ? (
              <section className="mic-debug">
                <p>Mic support: {micSupported ? "Supported" : "Unsupported"}</p>
                <p>Permission: {micPermission}</p>
                <p>Duration: {recordingSeconds.toFixed(1)}s</p>
                <p>Chunks: {chunkCount}</p>
                <p>Last audio: {formatBytes(lastAudioBytes)}</p>
                <p>Mime: {lastMimeType}</p>
                <p>WS status: {backendWsStatus}</p>
                <p>WS session: {backendSessionId}</p>
                <p>Server chunks: {backendChunks}</p>
                <p>Server bytes: {formatBytes(backendBytes)}</p>
                <p>STT status: {sttStatus}</p>
                <p>STT latency: {sttLatencyMs ? `${sttLatencyMs} ms` : "-"}</p>
                <p>LLM status: {llmStatus}</p>
                <p>LLM latency: {llmLatencyMs ? `${llmLatencyMs} ms` : "-"}</p>
                <p className="mic-debug-span">LLM model: {llmModelName}</p>
                <p>TTS status: {ttsStatus}</p>
                <p>TTS latency: {ttsLatencyMs ? `${ttsLatencyMs} ms` : "-"}</p>
                <p className="mic-debug-span">TTS provider: {ttsProvider}</p>
                <p className="mic-debug-span">TTS voice: {ttsVoice}</p>
                <p className="mic-debug-span">Last backend event: {backendEvent}</p>
                <p className="mic-debug-span">Last transcript: {lastTranscript}</p>
                <p className="mic-debug-span">Live transcript: {liveTranscript || "-"}</p>
                <div className="mic-debug-action">
                  <button className="ghost-button" type="button" onClick={reconnectWebSocket}>
                    Reconnect WS
                  </button>
                </div>
                <div className="audio-meter">
                  <span>Input level</span>
                  <div className="audio-meter-track">
                    <div
                      className="audio-meter-fill"
                      style={{ width: `${Math.max(2, Math.round(audioLevel * 100))}%` }}
                    />
                  </div>
                </div>
                {audioPreviewUrl ? (
                  <div className="audio-preview">
                    <span>Last capture preview</span>
                    <audio controls src={audioPreviewUrl} />
                  </div>
                ) : null}
                {micError ? <p className="mic-error">{micError}</p> : null}
              </section>
            ) : null}
          </aside>
        ) : null}

        <section className="conversation-pane">
          {saveStatus ? <p className="save-status">{saveStatus}</p> : null}
          {micError && (!settings.showStatusPanel || !isDebugOpen) ? (
            <p className="mic-error-banner">{micError}</p>
          ) : null}

          <section className="thread" aria-live="polite" ref={threadRef}>
            {messages.map((message) => (
              <article
                key={message.id}
                className={`bubble ${message.role === "user" ? "user" : "assistant"}`}
              >
                {message.kind === "voice" && message.audioUrl ? (
                  <>
                    <div className="voice-row">
                      <button
                        type="button"
                        className="voice-play-button"
                        onClick={() => togglePlayMessageAudio(message.id)}
                      >
                        {playingMessageId === message.id ? "■" : "▶"}
                      </button>
                      <div className="voice-text">
                        <p>{message.text}</p>
                        {message.audioDurationSec ? (
                          <p className="voice-meta">
                            Voice · {formatDuration(message.audioDurationSec)}
                          </p>
                        ) : null}
                      </div>
                    </div>
                    {message.sttText && message.sttText.trim() !== message.text.trim() ? (
                      <p className="voice-stt-line">STT: {message.sttText}</p>
                    ) : null}
                    <audio
                      ref={(element) => {
                        messageAudioElementMapRef.current[message.id] = element;
                      }}
                      src={message.audioUrl}
                      onLoadedMetadata={(event) => {
                        const duration = event.currentTarget.duration;
                        if (!Number.isFinite(duration) || duration <= 0) {
                          return;
                        }
                        setMessages((current) =>
                          current.map((item) =>
                            item.id === message.id && !item.audioDurationSec
                              ? {
                                  ...item,
                                  audioDurationSec: duration
                                }
                              : item
                          )
                        );
                      }}
                      onEnded={() => {
                        if (playingMessageId === message.id) {
                          setPlayingMessageId(null);
                        }
                      }}
                      onPause={() => {
                        if (playingMessageId === message.id) {
                          setPlayingMessageId(null);
                        }
                      }}
                      preload="metadata"
                    />
                  </>
                ) : (
                  <p>{message.text}</p>
                )}
              </article>
            ))}
            {isAssistantTyping ? <p className="typing">Assistant is drafting...</p> : null}
          </section>

          {hasPendingVoiceDecision ? (
            <section className="post-record-panel" aria-live="polite">
              <p className="post-record-text">
                Voice captured ({formatDuration(pendingVoiceCapture?.durationSec ?? 0)}). Choose:
                Re-record, edit text, or send.
              </p>
              <div className="post-record-actions">
                <button
                  type="button"
                  className="ghost-button"
                  onClick={reRecordAfterCapture}
                  disabled={isMicBusy || isAssistantTyping}
                >
                  Re-record
                </button>
                <button type="button" className="ghost-button" onClick={focusComposerInput}>
                  Edit text
                </button>
                <button
                  type="button"
                  onClick={() => {
                    void submitComposerMessage();
                  }}
                  disabled={!canSendPendingCapture}
                >
                  Send now
                </button>
              </div>
            </section>
          ) : null}

          <form className="composer" onSubmit={sendMessage}>
            <button
              type="button"
              className={`record-button ${isRecording ? "active" : ""}`}
              disabled={isMicBusy}
              onClick={() => {
                if (isRecording) {
                  stopRecording();
                } else {
                  void startRecording();
                }
              }}
            >
              {isMicBusy
                ? "Starting..."
                : isRecording
                  ? `Stop Recording (${recordingSeconds.toFixed(1)}s)`
                  : micSupported
                    ? "Start Recording"
                    : "Mic Unavailable"}
            </button>
            <input
              ref={composerInputRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Type Japanese or English..."
              aria-label="Message"
              disabled={isAssistantTyping}
            />
            <button type="submit" disabled={isAssistantTyping}>
              {isAssistantTyping ? "Thinking..." : "Send"}
            </button>
          </form>
        </section>
      </main>

      <SettingsPanel
        isOpen={isSettingsOpen}
        settings={settings}
        onClose={() => setIsSettingsOpen(false)}
        onChange={setSettings}
      />

      {isHistoryOpen ? (
        <div
          className="modal-backdrop"
          role="presentation"
          onClick={() => setIsHistoryOpen(false)}
        >
          <section
            className="save-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Saved conversations"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="panel-head">
              <h2>Saved Conversations</h2>
              <button className="ghost-button" onClick={() => setIsHistoryOpen(false)}>
                Close
              </button>
            </div>
            <button
              className="ghost-button"
              onClick={() => {
                void loadHistoryList(false);
              }}
              disabled={historyLoading}
            >
              Refresh
            </button>
            {historyError ? <p className="auth-error">{historyError}</p> : null}
            {historyLoading ? <p>Loading...</p> : null}
            {!historyLoading && historyItems.length === 0 ? (
              <p>No saved conversations yet.</p>
            ) : null}
            {historyItems.map((item) => (
              <article key={item.id} className="history-item">
                <div className="history-meta">
                  <p className="history-title">{displayHistoryTitle(item)}</p>
                  <p className="history-subtitle">
                    {new Date(item.created_at).toLocaleString()} · {item.message_count} msgs
                  </p>
                </div>
                <div className="history-actions">
                  <button
                    className="ghost-button history-open-btn"
                    onClick={() => {
                      setSaveStatus("Opening saved conversation...");
                      void loadHistoryConversation(item.id);
                    }}
                  >
                    Open
                  </button>
                  <button
                    className="ghost-button history-delete"
                    onClick={() => {
                      if (!window.confirm("Delete this saved conversation?")) {
                        return;
                      }
                      void deleteHistoryConversation(item.id);
                    }}
                  >
                    Delete
                  </button>
                </div>
              </article>
            ))}
          </section>
        </div>
      ) : null}

      {isSaveOpen ? (
        <div className="modal-backdrop" role="presentation" onClick={() => setIsSaveOpen(false)}>
          <section
            className="save-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Save conversation options"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="panel-head">
              <h2>Save Conversation</h2>
              <button className="ghost-button" onClick={() => setIsSaveOpen(false)}>
                Close
              </button>
            </div>
            <p>Select one option:</p>
            <button onClick={saveAudioPackage}>1. Save as audio package</button>
            <button onClick={exportConversationAsText}>2. Save as text file (.txt)</button>
            <button onClick={saveConversationToHistoryPool}>
              3. Save on server history
            </button>
          </section>
        </div>
      ) : null}
    </div>
  );
}
