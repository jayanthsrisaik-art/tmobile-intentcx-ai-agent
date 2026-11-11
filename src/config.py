"""Central configuration, driven by environment variables (optionally via .env)."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = field(default_factory=lambda: os.getenv("INTENTCX_MODEL", "gpt-4o-mini"))
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("INTENTCX_DATA_DIR", PROJECT_ROOT / "data"))
    )
    max_tool_rounds: int = field(default_factory=lambda: int(os.getenv("INTENTCX_MAX_TOOL_ROUNDS", "5")))
    max_history_turns: int = field(default_factory=lambda: int(os.getenv("INTENTCX_MAX_HISTORY", "10")))
    # Use the LLM to classify intent when keyword routing is low-confidence.
    llm_routing: bool = field(default_factory=lambda: _bool("INTENTCX_LLM_ROUTING", True))
    # Force offline (rule-based) mode even if an API key is present.
    offline: bool = field(default_factory=lambda: _bool("INTENTCX_OFFLINE", False))

    @property
    def llm_enabled(self) -> bool:
        return not self.offline and bool(os.getenv("OPENAI_API_KEY"))


def get_settings() -> Settings:
    return Settings()
