"""Conversation memory: per-session message history with bounded length."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Session:
    session_id: str
    customer_id: str
    messages: list[dict] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    escalated: bool = False


class SessionStore:
    """Thread-safe in-memory session store (swap for Redis/DB in production)."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._sessions: dict[str, Session] = {}
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None, customer_id: str) -> Session:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            sid = session_id or uuid.uuid4().hex
            session = Session(session_id=sid, customer_id=str(customer_id))
            self._sessions[sid] = session
            return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            return self._locks[session_id]

    def append(self, session: Session, role: str, content: str) -> None:
        session.messages.append({"role": role, "content": content})
        # Keep the last N user/assistant turns (2 messages per turn).
        overflow = len(session.messages) - self.max_turns * 2
        if overflow > 0:
            del session.messages[:overflow]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            self._locks.pop(session_id, None)
            return self._sessions.pop(session_id, None) is not None

    def __len__(self) -> int:
        return len(self._sessions)
