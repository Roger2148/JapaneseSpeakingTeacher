from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import wave
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib import error as urllib_error
from urllib import request as urllib_request

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.websockets import WebSocketDisconnect

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - dependency may be installed later by user.
    WhisperModel = None  # type: ignore[assignment]


@dataclass
class AudioSessionState:
    session_id: str
    connected_at: float
    total_chunks: int = 0
    total_bytes: int = 0
    recording_chunks: int = 0
    recording_bytes: int = 0
    recording_active: bool = False
    mime_type: str = "unknown"
    recording_audio_chunks: list[bytes] = field(default_factory=list)


class TutorSettings(BaseModel):
    tutorStyle: Literal["casual", "balanced", "strict"] = "balanced"
    replyLanguage: Literal["jp", "jp_en", "en"] = "jp_en"
    correctionIntensity: Literal["light", "medium", "heavy"] = "medium"
    responseLength: Literal["very_short", "short", "detailed"] = "short"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(default="", max_length=4000)


class ChatRequest(BaseModel):
    user_text: str = Field(min_length=1, max_length=4000)
    settings: TutorSettings = Field(default_factory=TutorSettings)
    history: list[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    text: str
    model: str
    latency_ms: float


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    reply_language: Literal["jp", "jp_en", "en"] = "jp_en"
    voice: str | None = None


class TtsResponse(BaseModel):
    audio_url: str
    provider: str
    voice: str
    mime_type: str
    latency_ms: float


class AuthLoginRequest(BaseModel):
    username: str = Field(min_length=2, max_length=40)


class AuthUserResponse(BaseModel):
    username: str
    created: bool = False


class HistorySaveMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=0, max_length=6000)
    kind: Literal["text", "voice"] | None = None
    sttText: str | None = None
    createdAt: str | None = None


class HistorySaveRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    messages: list[HistorySaveMessage] = Field(default_factory=list)


class HistoryItemSummary(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int


class HistoryItemResponse(BaseModel):
    id: str
    title: str
    created_at: str
    messages: list[HistorySaveMessage]


class AudioExportMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(min_length=0, max_length=8000)
    audio_b64: str | None = None
    audio_mime: str | None = None


class AudioExportRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    messages: list[AudioExportMessage] = Field(default_factory=list)


DEFAULT_PROMPT_PROFILE: dict[str, Any] = {
    "style_instruction": {
        "casual": "Friendly conversation partner. Keep tone relaxed, warm, and natural.",
        "balanced": "Conversation-first tutor mode. Respond naturally, then give short useful correction when needed.",
        "strict": "Teacher mode. Keep conversation flowing but prioritize precise grammar and natural phrasing corrections.",
    },
    "language_instruction": {
        "jp": "Reply only in Japanese.",
        "jp_en": "Reply primarily in Japanese and add short English support only when useful.",
        "en": "Reply in English first, with Japanese examples when useful.",
    },
    "correction_instruction": {
        "light": "Only correct major mistakes with one short fix.",
        "medium": "Correct clear mistakes and provide one natural rewrite.",
        "heavy": "Give stronger correction with natural rewrite and brief reason, but still keep a spoken tone.",
    },
    "length_instruction": {
        "very_short": "Keep answer to 1-2 short sentences.",
        "short": "Keep answer to 2-4 sentences.",
        "detailed": "Give a detailed but focused answer with examples.",
    },
    "global_rules": [
        "Use spoken, chat-like language, not textbook style.",
        "Default behavior: continue the conversation.",
        "React to what the learner said and ask one follow-up question to keep them talking.",
        "If correction is needed, keep it short and natural. Prefer inline correction or one short line like '自然な言い方: ...'.",
        "Do not output headings, bullet lists, tables, markdown separators, or code blocks unless the user explicitly asks for that format.",
        "Never reveal chain-of-thought. Keep output clean and learner-friendly.",
    ],
    "disable_correction_for_styles": ["casual"],
    "follow_up_question": {
        "jp": "ちなみに、どんな味でしたか？",
        "jp_en": "ちなみに、どんな味でしたか？ (How did it taste?)",
        "en": "By the way, how did it taste?",
    },
}


def contains_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text))


class JsonStateStore:
    def __init__(self) -> None:
        data_root = Path(__file__).resolve().parent / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        self.state_path = data_root / "state.json"
        self._lock = threading.RLock()
        self.default_state: dict[str, Any] = {
            "users": {},
            "sessions": {},
            "history": {},
        }

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return json.loads(json.dumps(self.default_state))
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
        except Exception:
            return json.loads(json.dumps(self.default_state))
        if not isinstance(parsed, dict):
            return json.loads(json.dumps(self.default_state))
        for key, value in self.default_state.items():
            if key not in parsed or not isinstance(parsed[key], type(value)):
                parsed[key] = json.loads(json.dumps(value))
        return parsed

    def _write_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(state, ensure_ascii=False, indent=2)
            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.state_path.parent),
                prefix="state.",
                suffix=".tmp",
                delete=False,
            )
            temp_path = Path(temp_file.name)
            try:
                with temp_file:
                    temp_file.write(payload)
                    temp_file.flush()
                temp_path.replace(self.state_path)
            finally:
                temp_path.unlink(missing_ok=True)

    def create_or_get_user(self, username: str) -> tuple[str, bool]:
        normalized = username.strip()
        if not normalized:
            raise RuntimeError("Username cannot be empty.")
        if len(normalized) > 40:
            raise RuntimeError("Username is too long.")
        if not re.fullmatch(r"[A-Za-z0-9._\- ]{2,40}", normalized):
            raise RuntimeError(
                "Username can only include letters, numbers, space, dot, underscore, and hyphen."
            )
        key = normalized.casefold()
        with self._lock:
            state = self._read_state()
            users = state["users"]
            created = False
            if key not in users:
                users[key] = {
                    "username": normalized,
                    "created_at": time.time(),
                }
                created = True
            self._write_state(state)
            return users[key]["username"], created

    def create_session(self, username: str, session_days: int) -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            state = self._read_state()
            expires_at = time.time() + (session_days * 24 * 3600)
            state["sessions"][session_id] = {
                "username_key": username.casefold(),
                "created_at": time.time(),
                "expires_at": expires_at,
                "last_seen_at": time.time(),
            }
            self._write_state(state)
            return session_id

    def destroy_session(self, session_id: str) -> None:
        with self._lock:
            state = self._read_state()
            state["sessions"].pop(session_id, None)
            self._write_state(state)

    def user_from_session(self, session_id: str) -> str | None:
        if not session_id:
            return None
        with self._lock:
            state = self._read_state()
            session = state["sessions"].get(session_id)
            if not isinstance(session, dict):
                return None
            if float(session.get("expires_at", 0)) < time.time():
                state["sessions"].pop(session_id, None)
                self._write_state(state)
                return None
            username_key = str(session.get("username_key", "")).casefold()
            user = state["users"].get(username_key)
            if not isinstance(user, dict):
                return None
            # Avoid a state write on every /auth/me poll from frontend.
            return str(user.get("username", "")).strip() or None

    def list_history(self, username: str) -> list[dict[str, Any]]:
        state = self._read_state()
        entries = state["history"].get(username.casefold(), [])
        if not isinstance(entries, list):
            return []
        sorted_entries = sorted(
            [entry for entry in entries if isinstance(entry, dict)],
            key=lambda item: str(item.get("created_at", "")),
            reverse=True,
        )
        return sorted_entries

    def get_history_item(self, username: str, item_id: str) -> dict[str, Any] | None:
        for item in self.list_history(username):
            if str(item.get("id")) == item_id:
                return item
        return None

    def save_history(
        self, username: str, title: str | None, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        with self._lock:
            state = self._read_state()
            key = username.casefold()
            history_bucket = state["history"].setdefault(key, [])
            if not isinstance(history_bucket, list):
                history_bucket = []
                state["history"][key] = history_bucket

            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            item_id = uuid.uuid4().hex[:16]
            auto_title = title.strip() if title else ""
            if not auto_title:
                first_user = next(
                    (m.get("text", "").strip() for m in messages if m.get("role") == "user"),
                    "",
                )
                auto_title = (first_user[:40] + "...") if len(first_user) > 40 else first_user
                if not auto_title:
                    auto_title = f"Conversation {created_at[:10]}"
            item = {
                "id": item_id,
                "title": auto_title,
                "created_at": created_at,
                "messages": messages,
            }
            history_bucket.append(item)
            # Keep last 200 saved conversations per user.
            state["history"][key] = history_bucket[-200:]
            self._write_state(state)
            return item


class LocalWhisperTranscriber:
    def __init__(self) -> None:
        self.model_name = os.getenv("STT_MODEL_NAME", "small")
        self.device = os.getenv("STT_DEVICE", "auto")
        self.compute_type = os.getenv("STT_COMPUTE_TYPE", "int8")
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed. Please install it in your environment."
            )
        if self._model is None:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe_file(self, file_path: str) -> dict[str, Any]:
        model = self._ensure_model()
        segments, info = model.transcribe(
            file_path,
            beam_size=1,
            best_of=1,
            vad_filter=True,
            temperature=0.0,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
        }


class LocalOllamaTutor:
    def __init__(self) -> None:
        self.base_url = os.getenv("LLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip(
            "/"
        )
        self.model = os.getenv("LLM_MODEL", "qwen3:8b")
        self.timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "90"))
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.5"))
        self.max_history_turns = int(os.getenv("LLM_MAX_HISTORY_TURNS", "8"))
        self.enable_thinking = (
            os.getenv("LLM_ENABLE_THINKING", "false").strip().lower() == "true"
        )
        default_profile_path = Path(__file__).resolve().parent / "prompt_profile.json"
        configured_profile_path = os.getenv("LLM_PROMPT_PROFILE_PATH", str(default_profile_path))
        profile_path = Path(configured_profile_path)
        if not profile_path.is_absolute():
            profile_path = (Path(__file__).resolve().parent / profile_path).resolve()
        self.prompt_profile_path = profile_path
        self._prompt_profile_cache: dict[str, Any] | None = None
        self._prompt_profile_mtime_ns: int | None = None

    def _deep_copy_default_profile(self) -> dict[str, Any]:
        # JSON round-trip keeps this file dependency-free while preventing accidental mutation.
        return json.loads(json.dumps(DEFAULT_PROMPT_PROFILE))

    def _merge_prompt_profile(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = self._deep_copy_default_profile()
        merged.update(base)
        for section in (
            "style_instruction",
            "language_instruction",
            "correction_instruction",
            "length_instruction",
            "follow_up_question",
        ):
            if isinstance(override.get(section), dict):
                section_dict = {
                    str(key): str(value)
                    for key, value in override[section].items()
                    if isinstance(key, str) and isinstance(value, str)
                }
                merged[section] = {**merged.get(section, {}), **section_dict}

        if isinstance(override.get("global_rules"), list):
            rules = [str(item) for item in override["global_rules"] if str(item).strip()]
            if rules:
                merged["global_rules"] = rules

        return merged

    def _load_prompt_profile_from_disk(self) -> dict[str, Any]:
        if not self.prompt_profile_path.exists():
            return self._deep_copy_default_profile()

        try:
            raw = self.prompt_profile_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
        except Exception:
            return self._deep_copy_default_profile()

        if not isinstance(parsed, dict):
            return self._deep_copy_default_profile()
        return self._merge_prompt_profile(DEFAULT_PROMPT_PROFILE, parsed)

    def _prompt_profile(self) -> dict[str, Any]:
        try:
            current_mtime = self.prompt_profile_path.stat().st_mtime_ns
        except FileNotFoundError:
            current_mtime = -1

        if (
            self._prompt_profile_cache is None
            or self._prompt_profile_mtime_ns != current_mtime
        ):
            self._prompt_profile_cache = self._load_prompt_profile_from_disk()
            self._prompt_profile_mtime_ns = current_mtime
        return self._prompt_profile_cache

    def _build_system_prompt(self, settings: TutorSettings) -> str:
        profile = self._prompt_profile()
        style_instruction = profile.get("style_instruction", {}).get(
            settings.tutorStyle, DEFAULT_PROMPT_PROFILE["style_instruction"]["balanced"]
        )
        language_instruction = profile.get("language_instruction", {}).get(
            settings.replyLanguage, DEFAULT_PROMPT_PROFILE["language_instruction"]["jp_en"]
        )
        correction_instruction = profile.get("correction_instruction", {}).get(
            settings.correctionIntensity,
            DEFAULT_PROMPT_PROFILE["correction_instruction"]["medium"],
        )
        disabled_styles = profile.get("disable_correction_for_styles", [])
        if isinstance(disabled_styles, list) and settings.tutorStyle in {
            str(item) for item in disabled_styles
        }:
            correction_instruction = (
                "Do not provide corrections by default. "
                "Only correct if the learner explicitly asks for correction."
            )
        length_instruction = profile.get("length_instruction", {}).get(
            settings.responseLength, DEFAULT_PROMPT_PROFILE["length_instruction"]["short"]
        )
        global_rules = profile.get("global_rules")
        if not isinstance(global_rules, list) or not global_rules:
            global_rules = DEFAULT_PROMPT_PROFILE["global_rules"]
        global_rules_text = " ".join(str(rule).strip() for rule in global_rules if str(rule).strip())
        return (
            "You are a Japanese speaking tutor for conversation practice. "
            f"{style_instruction} {language_instruction} "
            f"{correction_instruction} {length_instruction} "
            f"{global_rules_text}"
        )

    def _num_predict_for_length(self, response_length: str) -> int:
        mapping = {
            "very_short": 80,
            "short": 180,
            "detailed": 360,
        }
        return mapping.get(response_length, 180)

    def _prepare_messages(
        self, history: list[ChatTurn], user_text: str, settings: TutorSettings
    ) -> list[dict[str, str]]:
        cleaned_history: list[dict[str, str]] = []
        for turn in history:
            text = turn.text.strip()
            if not text:
                continue
            cleaned_history.append({"role": turn.role, "content": text})
        trimmed_history = cleaned_history[-self.max_history_turns :]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt(settings)}
        ]
        messages.extend(trimmed_history)
        messages.append({"role": "user", "content": user_text.strip()})
        return messages

    def _clean_assistant_text(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        # Qwen models may include hidden reasoning in <think>...</think> blocks.
        # Remove these blocks before returning text to UI.
        no_think = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
        # Light cleanup for markdown-heavy answers in chat UI.
        no_hr = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", no_think)
        no_heading = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", no_hr)
        no_bold = no_heading.replace("**", "")
        return no_bold.strip()

    def _strip_correction_lines_for_casual(self, text: str) -> str:
        # Enforce "conversation only" mode robustness for casual style.
        stripped = re.sub(
            r"(自然な言い方|より自然な言い方|言い換えると|訂正)\s*[:：]\s*[^\n。!?！？]+[。!?！？]?",
            "",
            text,
            flags=re.IGNORECASE,
        )
        stripped = re.sub(r"\s{2,}", " ", stripped)
        stripped = re.sub(r"\n{2,}", "\n", stripped)
        return stripped.strip()

    def _ensure_follow_up_question(self, text: str, settings: TutorSettings) -> str:
        if "?" in text or "？" in text:
            return text
        follow_up_map = self._prompt_profile().get("follow_up_question", {})
        follow_up = str(
            follow_up_map.get(
                settings.replyLanguage,
                DEFAULT_PROMPT_PROFILE["follow_up_question"]["jp"],
            )
        ).strip()
        if not follow_up:
            follow_up = DEFAULT_PROMPT_PROFILE["follow_up_question"]["jp"]
        return f"{text} {follow_up}".strip()

    def _chat_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib_request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_sec) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore").strip()
            detail_head = detail[:300] if detail else exc.reason
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail_head}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(
                "Cannot reach Ollama. Start Ollama and run: ollama run qwen3:8b"
            ) from exc

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid response from Ollama /api/chat") from exc
        return body

    def draft_reply(
        self, user_text: str, settings: TutorSettings, history: list[ChatTurn]
    ) -> dict[str, str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": self._prepare_messages(history, user_text, settings),
            "think": self.enable_thinking,
            "options": {
                "temperature": self.temperature,
                "num_predict": self._num_predict_for_length(settings.responseLength),
            },
        }

        body = self._chat_once(payload)
        message = body.get("message")
        text = ""
        if isinstance(message, dict):
            text = self._clean_assistant_text(str(message.get("content", "")))

        # Some Qwen/Ollama combinations occasionally return an empty final content.
        # Retry once with explicit non-thinking mode to recover.
        if not text:
            retry_payload = dict(payload)
            retry_payload["think"] = False
            body = self._chat_once(retry_payload)
            message = body.get("message")
            if isinstance(message, dict):
                text = self._clean_assistant_text(str(message.get("content", "")))

        if not text:
            raise RuntimeError("Ollama returned an empty reply after retry.")
        if settings.tutorStyle == "casual":
            text = self._strip_correction_lines_for_casual(text)
        text = self._ensure_follow_up_question(text, settings)

        model_name = str(body.get("model") or self.model)
        return {"text": text, "model": model_name}


class LocalSpeechSynthesizer:
    def __init__(self) -> None:
        self.provider = os.getenv("TTS_PROVIDER", "auto").strip().lower()
        self.piper_model = os.getenv("TTS_PIPER_MODEL", "").strip()
        self.voice_jp = os.getenv("TTS_VOICE_JP", "Kyoko").strip() or "Kyoko"
        self.voice_en = os.getenv("TTS_VOICE_EN", "Samantha").strip() or "Samantha"
        self.say_rate = os.getenv("TTS_SAY_RATE", "").strip()
        self.say_mixed_language = (
            os.getenv("TTS_SAY_MIXED_LANGUAGE", "true").strip().lower() == "true"
        )
        self.force_japanese_when_jp_text = (
            os.getenv("TTS_FORCE_JAPANESE_WHEN_JP_TEXT", "true").strip().lower()
            == "true"
        )
        self.max_keep = int(os.getenv("TTS_MAX_FILES", "200"))
        self.max_age_sec = int(os.getenv("TTS_MAX_AGE_SEC", "86400"))
        self.output_dir = Path(__file__).resolve().parent / "generated_audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._say_voices_cache: set[str] | None = None

    def _cleanup_old_files(self) -> None:
        try:
            files = sorted(
                [path for path in self.output_dir.glob("*") if path.is_file()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return

        now = time.time()
        for index, path in enumerate(files):
            should_delete = False
            if index >= self.max_keep:
                should_delete = True
            else:
                age_sec = now - path.stat().st_mtime
                should_delete = age_sec > self.max_age_sec
            if should_delete:
                path.unlink(missing_ok=True)

    def _clean_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            raise RuntimeError("TTS text is empty.")
        return normalized

    def _target_language(self, text: str, reply_language: str) -> str:
        # Japanese script should always prefer Japanese TTS voice.
        if contains_japanese(text):
            return "jp"
        if reply_language == "jp":
            return "jp"
        return "en"

    def _available_say_voices(self) -> set[str]:
        if self._say_voices_cache is not None:
            return self._say_voices_cache
        if not shutil.which("say"):
            self._say_voices_cache = set()
            return self._say_voices_cache
        try:
            result = subprocess.run(
                ["say", "-v", "?"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except subprocess.SubprocessError:
            self._say_voices_cache = set()
            return self._say_voices_cache

        voices: set[str] = set()
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if parts:
                voices.add(parts[0])
        self._say_voices_cache = voices
        return voices

    def _pick_say_voice(self, language: str, requested_voice: str | None) -> str:
        voices = self._available_say_voices()
        if requested_voice:
            if not voices or requested_voice in voices:
                return requested_voice
        preferred = self.voice_jp if language == "jp" else self.voice_en
        if not voices or preferred in voices:
            return preferred
        if language == "jp":
            for candidate in ("Kyoko", "Otoya"):
                if candidate in voices:
                    return candidate
        fallback_order = ("Samantha", "Alex", "Daniel")
        for candidate in fallback_order:
            if candidate in voices:
                return candidate
        return preferred

    def _has_english_letters(self, text: str) -> bool:
        return bool(re.search(r"[A-Za-z]", text))

    def _char_script(self, char: str) -> str:
        if re.match(r"[\u3040-\u30ff\u3400-\u9fff]", char):
            return "jp"
        if re.match(r"[A-Za-z0-9]", char):
            return "en"
        return "neutral"

    def _split_mixed_say_segments(
        self, text: str, default_language: str
    ) -> list[tuple[str, str]]:
        if not text.strip():
            return []
        segments: list[tuple[str, str]] = []
        current_language = default_language
        current_chars: list[str] = []

        for char in text:
            script = self._char_script(char)
            if script == "neutral":
                current_chars.append(char)
                continue
            if not current_chars:
                current_language = script
                current_chars.append(char)
                continue
            if script == current_language:
                current_chars.append(char)
                continue

            segment_text = "".join(current_chars).strip()
            if segment_text:
                segments.append((current_language, segment_text))
            current_language = script
            current_chars = [char]

        tail_text = "".join(current_chars).strip()
        if tail_text:
            segments.append((current_language, tail_text))

        if not segments:
            return [(default_language, text)]
        return segments

    def _run_piper(self, text: str, output_path: Path) -> dict[str, str]:
        if not shutil.which("piper"):
            raise RuntimeError("`piper` command not found.")
        if not self.piper_model:
            raise RuntimeError("TTS_PIPER_MODEL is not configured.")
        model_path = Path(self.piper_model).expanduser()
        if not model_path.exists():
            raise RuntimeError(f"TTS_PIPER_MODEL not found: {model_path}")

        command = [
            "piper",
            "--model",
            str(model_path),
            "--output_file",
            str(output_path),
        ]
        subprocess.run(
            command,
            input=text.encode("utf-8"),
            check=True,
            timeout=120,
        )
        return {
            "provider": "piper",
            "voice": model_path.stem,
            "mime_type": "audio/wav",
        }

    def _run_say(
        self,
        text: str,
        output_path: Path,
        voice: str,
    ) -> dict[str, str]:
        if not shutil.which("say"):
            raise RuntimeError("`say` command not found.")
        command = [
            "say",
            "-v",
            voice,
            "-o",
            str(output_path),
            "--file-format=WAVE",
            "--data-format=LEI16",
        ]
        if self.say_rate:
            command.extend(["-r", self.say_rate])
        command.append(text)
        subprocess.run(command, check=True, timeout=120)
        return {
            "provider": "say",
            "voice": voice,
            "mime_type": "audio/wav",
        }

    def _merge_wav_segments(self, segment_paths: list[Path], output_path: Path) -> None:
        if not segment_paths:
            raise RuntimeError("No WAV segments generated.")
        with wave.open(str(output_path), "wb") as merged:
            expected: tuple[int, int, int] | None = None
            for segment in segment_paths:
                with wave.open(str(segment), "rb") as source:
                    params = (source.getnchannels(), source.getsampwidth(), source.getframerate())
                    if expected is None:
                        expected = params
                        merged.setnchannels(params[0])
                        merged.setsampwidth(params[1])
                        merged.setframerate(params[2])
                    elif params != expected:
                        raise RuntimeError("Incompatible WAV params while merging segments.")
                    merged.writeframes(source.readframes(source.getnframes()))

    def _run_say_mixed(
        self,
        text: str,
        output_path: Path,
        default_language: str,
        requested_voice: str | None,
    ) -> dict[str, str]:
        segments = self._split_mixed_say_segments(text, default_language)
        if len(segments) <= 1:
            language = segments[0][0] if segments else default_language
            voice = self._pick_say_voice(language, requested_voice)
            return self._run_say(text, output_path, voice)

        generated_paths: list[Path] = []
        voices_used: list[str] = []
        try:
            for index, (language, segment_text) in enumerate(segments):
                voice = self._pick_say_voice(language, requested_voice if index == 0 else None)
                voices_used.append(voice)
                segment_path = output_path.with_name(f"{output_path.stem}_seg{index}.wav")
                self._run_say(segment_text, segment_path, voice)
                generated_paths.append(segment_path)
            self._merge_wav_segments(generated_paths, output_path)
        finally:
            for path in generated_paths:
                path.unlink(missing_ok=True)

        unique_voices: list[str] = []
        for voice in voices_used:
            if voice not in unique_voices:
                unique_voices.append(voice)
        return {
            "provider": "say",
            "voice": "mixed:" + ",".join(unique_voices),
            "mime_type": "audio/wav",
        }

    def _run_espeak(self, text: str, output_path: Path, language: str) -> dict[str, str]:
        if not shutil.which("espeak"):
            raise RuntimeError("`espeak` command not found.")
        command = ["espeak", "-w", str(output_path)]
        if language == "jp":
            command.extend(["-v", "ja"])
        else:
            command.extend(["-v", "en"])
        command.append(text)
        subprocess.run(command, check=True, timeout=120)
        return {
            "provider": "espeak",
            "voice": "ja" if language == "jp" else "en",
            "mime_type": "audio/wav",
        }

    def synthesize(
        self, text: str, reply_language: str, requested_voice: str | None = None
    ) -> dict[str, str]:
        self._cleanup_old_files()
        cleaned = self._clean_text(text)
        has_jp = contains_japanese(cleaned)
        language = self._target_language(cleaned, reply_language)
        if self.force_japanese_when_jp_text and has_jp:
            language = "jp"
        file_token = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

        provider_preferences = (
            [self.provider] if self.provider not in ("", "auto") else ["piper", "say", "espeak"]
        )
        if self.provider == "local_piper":
            provider_preferences = ["piper", "say", "espeak"]
        if self.provider == "local_system":
            provider_preferences = ["say", "espeak"]

        errors: list[str] = []
        for provider_name in provider_preferences:
            try:
                if provider_name == "piper":
                    output = self.output_dir / f"tts_{file_token}.wav"
                    meta = self._run_piper(cleaned, output)
                elif provider_name == "say":
                    output = self.output_dir / f"tts_{file_token}.wav"
                    use_mixed_say = (
                        self.say_mixed_language
                        and not (self.force_japanese_when_jp_text and has_jp)
                        and has_jp
                        and self._has_english_letters(cleaned)
                    )
                    if use_mixed_say:
                        meta = self._run_say_mixed(
                            cleaned,
                            output,
                            default_language=language,
                            requested_voice=requested_voice,
                        )
                    else:
                        voice = self._pick_say_voice(language, requested_voice)
                        meta = self._run_say(cleaned, output, voice)
                elif provider_name == "espeak":
                    output = self.output_dir / f"tts_{file_token}.wav"
                    meta = self._run_espeak(cleaned, output, language)
                else:
                    continue
            except (subprocess.SubprocessError, RuntimeError) as exc:
                errors.append(f"{provider_name}: {exc}")
                continue

            if output.exists() and output.stat().st_size > 0:
                return {
                    "file_name": output.name,
                    "provider": meta["provider"],
                    "voice": meta["voice"],
                    "mime_type": meta["mime_type"],
                }
            errors.append(f"{provider_name}: empty output file")

        detail = "; ".join(errors) if errors else "No provider attempted."
        raise RuntimeError(
            "No TTS engine available. Install piper or use system TTS. "
            f"Details: {detail}"
        )


app = FastAPI(title="Japanese Speaking Teacher API", version="0.1.0")
transcriber = LocalWhisperTranscriber()
tutor = LocalOllamaTutor()
speech = LocalSpeechSynthesizer()
store = JsonStateStore()
app.mount("/generated_audio", StaticFiles(directory=str(speech.output_dir)), name="generated_audio")

COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "jst_session")
SESSION_DAYS = int(os.getenv("AUTH_SESSION_DAYS", "30"))
COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").strip().lower() == "true"

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
cors_origin_regex = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[0-1])\.\d+\.\d+)(:\d+)?$",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

allowed_hosts = [
    host.strip()
    for host in os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if host.strip()
]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts or ["*"])


def suffix_from_mime(mime_type: str) -> str:
    lowered = mime_type.lower()
    if "webm" in lowered:
        return ".webm"
    if "ogg" in lowered:
        return ".ogg"
    if "mp4" in lowered or "m4a" in lowered:
        return ".m4a"
    if "wav" in lowered:
        return ".wav"
    return ".audio"


def transcribe_audio_bytes(audio_bytes: bytes, mime_type: str) -> dict[str, Any]:
    suffix = suffix_from_mime(mime_type)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = Path(temp_file.name)
        return transcriber.transcribe_file(str(temp_path))
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "japanese-speaking-teacher-api",
        "llm_model": tutor.model,
        "timestamp": time.time(),
    }


def effective_secure_cookie(request: Request) -> bool:
    if COOKIE_SECURE:
        return True
    return request.url.scheme == "https"


def current_username_or_none(request: Request) -> str | None:
    session_id = request.cookies.get(COOKIE_NAME, "").strip()
    if not session_id:
        return None
    return store.user_from_session(session_id)


def require_username(request: Request) -> str:
    username = current_username_or_none(request)
    if not username:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return username


def set_session_cookie(response: Response, request: Request, session_id: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        max_age=SESSION_DAYS * 24 * 3600,
        httponly=True,
        samesite="lax",
        secure=effective_secure_cookie(request),
        path="/",
    )


def clear_session_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        secure=effective_secure_cookie(request),
        samesite="lax",
    )


def username_from_websocket(websocket: WebSocket) -> str | None:
    cookie_header = websocket.headers.get("cookie", "")
    if not cookie_header:
        return None
    cookies: dict[str, str] = {}
    for part in cookie_header.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        cookies[key.strip()] = value.strip()
    session_id = cookies.get(COOKIE_NAME, "")
    if not session_id:
        return None
    return store.user_from_session(session_id)


@app.post("/auth/login", response_model=AuthUserResponse)
def auth_login(payload: AuthLoginRequest, request: Request, response: Response) -> AuthUserResponse:
    try:
        username, created = store.create_or_get_user(payload.username)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session_id = store.create_session(username, SESSION_DAYS)
    set_session_cookie(response, request, session_id)
    return AuthUserResponse(username=username, created=created)


@app.post("/auth/logout")
def auth_logout(request: Request, response: Response) -> dict[str, str]:
    session_id = request.cookies.get(COOKIE_NAME, "").strip()
    if session_id:
        store.destroy_session(session_id)
    clear_session_cookie(response, request)
    return {"status": "ok"}


@app.get("/auth/me", response_model=AuthUserResponse)
def auth_me(username: str = Depends(require_username)) -> AuthUserResponse:
    return AuthUserResponse(username=username, created=False)


@app.get("/history/list", response_model=list[HistoryItemSummary])
def history_list(username: str = Depends(require_username)) -> list[HistoryItemSummary]:
    items = store.list_history(username)
    return [
        HistoryItemSummary(
            id=str(item.get("id", "")),
            title=str(item.get("title", "Untitled")),
            created_at=str(item.get("created_at", "")),
            message_count=len(item.get("messages", []))
            if isinstance(item.get("messages"), list)
            else 0,
        )
        for item in items
    ]


@app.get("/history/{item_id}", response_model=HistoryItemResponse)
def history_get(item_id: str, username: str = Depends(require_username)) -> HistoryItemResponse:
    item = store.get_history_item(username, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    raw_messages = item.get("messages", [])
    messages: list[HistorySaveMessage] = []
    if isinstance(raw_messages, list):
        for entry in raw_messages:
            if not isinstance(entry, dict):
                continue
            try:
                messages.append(HistorySaveMessage(**entry))
            except Exception:
                continue
    return HistoryItemResponse(
        id=str(item.get("id", "")),
        title=str(item.get("title", "Untitled")),
        created_at=str(item.get("created_at", "")),
        messages=messages,
    )


@app.post("/history/save", response_model=HistoryItemSummary)
def history_save(
    payload: HistorySaveRequest, username: str = Depends(require_username)
) -> HistoryItemSummary:
    normalized_messages = [
        {
            "role": msg.role,
            "text": msg.text,
            "kind": msg.kind,
            "sttText": msg.sttText,
            "createdAt": msg.createdAt,
        }
        for msg in payload.messages
    ]
    item = store.save_history(username, payload.title, normalized_messages)
    return HistoryItemSummary(
        id=item["id"],
        title=item["title"],
        created_at=item["created_at"],
        message_count=len(normalized_messages),
    )


def file_extension_from_mime(mime_type: str | None) -> str:
    if not mime_type:
        return "bin"
    lowered = mime_type.lower()
    if "wav" in lowered:
        return "wav"
    if "webm" in lowered:
        return "webm"
    if "ogg" in lowered:
        return "ogg"
    if "mpeg" in lowered or "mp3" in lowered:
        return "mp3"
    if "mp4" in lowered or "m4a" in lowered or "aac" in lowered:
        return "m4a"
    if "aiff" in lowered or "aif" in lowered:
        return "aiff"
    return "bin"


@app.post("/export/audio-package")
def export_audio_package(
    payload: AudioExportRequest,
    username: str = Depends(require_username),
) -> StreamingResponse:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    safe_title = (payload.title or "conversation").strip().replace("/", "-")
    transcript_lines: list[str] = []
    bundle = BytesIO()
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for idx, message in enumerate(payload.messages, start=1):
            transcript_lines.append(f"{idx:02d}. {message.role.upper()}: {message.text}")
            if not message.audio_b64:
                continue
            try:
                raw_audio = base64.b64decode(message.audio_b64, validate=True)
            except (ValueError, binascii.Error):
                continue
            ext = file_extension_from_mime(message.audio_mime)
            archive.writestr(f"audio/{idx:02d}_{message.role}.{ext}", raw_audio)
        transcript_content = "\n".join(transcript_lines) + "\n"
        archive.writestr("transcript.txt", transcript_content)
        metadata = {
            "username": username,
            "title": safe_title,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message_count": len(payload.messages),
        }
        archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    bundle.seek(0)
    filename = f"{safe_title}-{stamp}.zip"
    return StreamingResponse(
        bundle,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, _: str = Depends(require_username)) -> ChatResponse:
    started_at = time.time()
    try:
        result = await asyncio.to_thread(
            tutor.draft_reply,
            payload.user_text,
            payload.settings,
            payload.history,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    latency_ms = round((time.time() - started_at) * 1000, 2)
    return ChatResponse(
        text=result["text"],
        model=result["model"],
        latency_ms=latency_ms,
    )


@app.post("/tts", response_model=TtsResponse)
async def tts(payload: TtsRequest, _: str = Depends(require_username)) -> TtsResponse:
    started_at = time.time()
    try:
        result = await asyncio.to_thread(
            speech.synthesize,
            payload.text,
            payload.reply_language,
            payload.voice,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    latency_ms = round((time.time() - started_at) * 1000, 2)
    return TtsResponse(
        audio_url=f"/generated_audio/{result['file_name']}",
        provider=result["provider"],
        voice=result["voice"],
        mime_type=result["mime_type"],
        latency_ms=latency_ms,
    )


async def send_event(websocket: WebSocket, payload: dict[str, object]) -> bool:
    try:
        await websocket.send_text(json.dumps(payload))
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError:
        return False


@app.websocket("/ws/audio")
async def audio_stream(websocket: WebSocket) -> None:
    username = username_from_websocket(websocket)
    if not username:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    state = AudioSessionState(session_id=uuid.uuid4().hex[:12], connected_at=time.time())

    sent = await send_event(
        websocket,
        {
            "type": "server_ready",
            "session_id": state.session_id,
            "username": username,
            "connected_at": state.connected_at,
        },
    )
    if not sent:
        return

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            text_payload = message.get("text")
            if text_payload is not None:
                try:
                    payload = json.loads(text_payload)
                except json.JSONDecodeError:
                    sent = await send_event(
                        websocket,
                        {"type": "error", "detail": "Invalid JSON payload"},
                    )
                    if not sent:
                        return
                    continue

                event_type = payload.get("type")
                if event_type == "recording_started":
                    state.recording_active = True
                    state.recording_chunks = 0
                    state.recording_bytes = 0
                    state.recording_audio_chunks = []
                    state.mime_type = str(payload.get("mime_type", "unknown"))
                    sent = await send_event(
                        websocket,
                        {
                            "type": "recording_started_ack",
                            "session_id": state.session_id,
                            "mime_type": state.mime_type,
                        },
                    )
                    if not sent:
                        return
                elif event_type == "recording_stopped":
                    state.recording_active = False
                    sent = await send_event(
                        websocket,
                        {
                            "type": "recording_summary",
                            "session_id": state.session_id,
                            "recording_chunks": state.recording_chunks,
                            "recording_bytes": state.recording_bytes,
                            "total_chunks": state.total_chunks,
                            "total_bytes": state.total_bytes,
                            "client_chunk_count": payload.get("client_chunk_count"),
                            "client_bytes": payload.get("client_bytes"),
                            "duration_sec": payload.get("duration_sec"),
                        },
                    )
                    if not sent:
                        return

                    audio_bytes = b"".join(state.recording_audio_chunks)
                    if not audio_bytes:
                        sent = await send_event(
                            websocket,
                            {
                                "type": "transcription_error",
                                "detail": "No audio chunks received for transcription.",
                            },
                        )
                        if not sent:
                            return
                        continue

                    sent = await send_event(
                        websocket,
                        {
                            "type": "transcription_started",
                            "audio_bytes": len(audio_bytes),
                        },
                    )
                    if not sent:
                        return

                    started_at = time.time()
                    try:
                        result = await asyncio.to_thread(
                            transcribe_audio_bytes,
                            audio_bytes,
                            state.mime_type,
                        )
                    except Exception as exc:
                        sent = await send_event(
                            websocket,
                            {
                                "type": "transcription_error",
                                "detail": str(exc),
                            },
                        )
                        if not sent:
                            return
                        continue

                    latency_ms = round((time.time() - started_at) * 1000, 2)
                    sent = await send_event(
                        websocket,
                        {
                            "type": "transcription_result",
                            "text": result.get("text", ""),
                            "language": result.get("language"),
                            "language_probability": result.get("language_probability"),
                            "latency_ms": latency_ms,
                        },
                    )
                    if not sent:
                        return
                elif event_type == "transcription_partial_request":
                    request_id = payload.get("request_id")
                    if not state.recording_active:
                        sent = await send_event(
                            websocket,
                            {
                                "type": "partial_transcription_result",
                                "request_id": request_id,
                                "text": "",
                                "recording_active": False,
                            },
                        )
                        if not sent:
                            return
                        continue

                    audio_bytes = b"".join(state.recording_audio_chunks)
                    if len(audio_bytes) < 2048:
                        sent = await send_event(
                            websocket,
                            {
                                "type": "partial_transcription_result",
                                "request_id": request_id,
                                "text": "",
                                "recording_active": True,
                            },
                        )
                        if not sent:
                            return
                        continue

                    started_at = time.time()
                    try:
                        result = await asyncio.to_thread(
                            transcribe_audio_bytes,
                            audio_bytes,
                            state.mime_type,
                        )
                    except Exception as exc:
                        sent = await send_event(
                            websocket,
                            {
                                "type": "partial_transcription_error",
                                "request_id": request_id,
                                "detail": str(exc),
                            },
                        )
                        if not sent:
                            return
                        continue

                    latency_ms = round((time.time() - started_at) * 1000, 2)
                    sent = await send_event(
                        websocket,
                        {
                            "type": "partial_transcription_result",
                            "request_id": request_id,
                            "text": result.get("text", ""),
                            "language": result.get("language"),
                            "language_probability": result.get("language_probability"),
                            "latency_ms": latency_ms,
                            "recording_active": True,
                        },
                    )
                    if not sent:
                        return
                elif event_type == "audio_chunk":
                    chunk_b64 = payload.get("chunk_b64")
                    if not isinstance(chunk_b64, str):
                        sent = await send_event(
                            websocket,
                            {"type": "error", "detail": "audio_chunk missing chunk_b64"},
                        )
                        if not sent:
                            return
                        continue

                    try:
                        chunk = base64.b64decode(chunk_b64, validate=True)
                    except (binascii.Error, ValueError):
                        sent = await send_event(
                            websocket,
                            {"type": "error", "detail": "Invalid base64 in audio_chunk"},
                        )
                        if not sent:
                            return
                        continue

                    size = len(chunk)
                    state.total_chunks += 1
                    state.total_bytes += size
                    if state.recording_active:
                        state.recording_chunks += 1
                        state.recording_bytes += size
                        state.recording_audio_chunks.append(chunk)

                    sent = await send_event(
                        websocket,
                        {
                            "type": "chunk_received",
                            "session_id": state.session_id,
                            "chunk_size": size,
                            "recording_chunks": state.recording_chunks,
                            "recording_bytes": state.recording_bytes,
                            "total_chunks": state.total_chunks,
                            "total_bytes": state.total_bytes,
                        },
                    )
                    if not sent:
                        return
                elif event_type == "ping":
                    sent = await send_event(websocket, {"type": "pong", "ts": time.time()})
                    if not sent:
                        return
                else:
                    sent = await send_event(
                        websocket,
                        {"type": "ignored_event", "event": str(event_type)},
                    )
                    if not sent:
                        return

                continue

            binary_payload = message.get("bytes")
            if binary_payload is not None:
                size = len(binary_payload)
                state.total_chunks += 1
                state.total_bytes += size
                if state.recording_active:
                    state.recording_chunks += 1
                    state.recording_bytes += size
                    state.recording_audio_chunks.append(binary_payload)

                sent = await send_event(
                    websocket,
                    {
                        "type": "chunk_received",
                        "session_id": state.session_id,
                        "chunk_size": size,
                        "recording_chunks": state.recording_chunks,
                        "recording_bytes": state.recording_bytes,
                        "total_chunks": state.total_chunks,
                        "total_bytes": state.total_bytes,
                    },
                )
                if not sent:
                    return
    except WebSocketDisconnect:
        return
