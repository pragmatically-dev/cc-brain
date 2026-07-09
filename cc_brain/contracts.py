from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    kind: str = "code"
    include: str = r"\.(py|js|jsx|ts|tsx|md|json|yaml|yml|toml|rs|go|java|kt|sql|sh|ps1|html|css)$"
    exclude: str = r"(^|[\\/])(\.git|node_modules|dist|build|target|__pycache__|\.venv|venv|coverage)[\\/]"
    project: str = ""
    trust: float = 1.0


@dataclass(frozen=True)
class Chunk:
    id: int | None
    source: str
    path: Path
    loc: str
    title: str
    text: str
    project: str = ""
    mtime: float = 0.0
    trust: float = 1.0


@dataclass(frozen=True)
class SearchHit:
    id: int
    score: float
    source: str
    path: str
    loc: str
    title: str
    text: str
    project: str = ""


@dataclass(frozen=True)
class HookPayload:
    event: str
    cwd: Path
    transcript_path: Path | None = None
    prompt: str = ""
    source: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_response: object = None
    session_id: str = ""


@dataclass(frozen=True)
class CaptureResult:
    l0_path: Path
    l1_path: Path
    project: str
    title: str


class Indexer(Protocol):
    def index(self, rebuild: bool = False) -> str: ...
    def search(self, query: str, k: int = 6, source: str = "", project: str = "", lex: bool = False) -> list[SearchHit]: ...
