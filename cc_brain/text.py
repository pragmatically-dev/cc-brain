from __future__ import annotations

import re
from pathlib import Path

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(
        r"(AQ\.[\w\-]{10,}|nvapi-[\w\-]{10,}|AIza[\w\-]{20,}|sk-[A-Za-z0-9\-_]{20,}|"
        r"gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[\w\-]{10,}|AKIA[0-9A-Z]{16})"
    ),
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b(\s*[=:]\s*)([\"']?)[^\s\"',;]{8,}\3"),
    re.compile(r"\b([a-z][a-z0-9+.\-]*://[^\s:@/]+):([^\s@/]+)@"),
)
SECRET_RE = SECRET_PATTERNS[1]
WORD_RE = re.compile(r"[A-Za-z0-9_]{2,}")
URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")
MAX_CHUNK = 1800
CODE_WINDOW = 80
CODE_OVERLAP = 20


def scrub(text: str) -> str:
    out = text or ""
    out = SECRET_PATTERNS[0].sub("[REDACTED-KEY-BLOCK]", out)
    out = SECRET_PATTERNS[1].sub("[REDACTED]", out)
    out = SECRET_PATTERNS[2].sub("[REDACTED-JWT]", out)
    out = SECRET_PATTERNS[3].sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", out)
    out = SECRET_PATTERNS[4].sub(lambda m: f"{m.group(1)}:[REDACTED]@", out)
    return out


def tokenize(text: str) -> list[str]:
    raw = WORD_RE.findall(text or "")
    toks = [t.lower() for t in raw]
    extra: list[str] = []
    for token in raw:
        parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", token)
        if len(parts) > 1:
            extra.extend(p.lower() for p in parts)
    return toks + extra


def split_long(text: str, limit: int = MAX_CHUNK) -> list[str]:
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    cur: list[str] = []
    size = 0
    for para in text.split("\n\n"):
        if cur and size + len(para) > limit:
            out.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(para)
        size += len(para) + 2
    if cur:
        out.append("\n\n".join(cur))
    return out


def chunk_markdown(text: str):
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    cur: list[str] = []
    trail = ["(top)"]
    for line in lines:
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            if cur and any(x.strip() for x in cur):
                sections.append((" > ".join(trail), "\n".join(cur)))
            depth = len(m.group(1))
            trail = trail[: max(0, depth - 1)] or []
            trail = (trail + [m.group(2).strip()]) if trail else [m.group(2).strip()]
            cur = [line]
        else:
            cur.append(line)
    if cur and any(x.strip() for x in cur):
        sections.append((" > ".join(trail), "\n".join(cur)))
    for title, body in sections:
        for i, piece in enumerate(split_long(body)):
            yield (title if i == 0 else f"{title} (cont {i})", title, piece)


def chunk_code(text: str):
    lines = text.split("\n")
    step = CODE_WINDOW - CODE_OVERLAP
    start = 0
    while start < len(lines):
        end = min(start + CODE_WINDOW, len(lines))
        body = "\n".join(lines[start:end])
        if body.strip():
            yield (f"L{start + 1}-{end}", f"L{start + 1}", body)
        if end >= len(lines):
            break
        start += step


def chunk_file(path: Path, text: str):
    if path.suffix.lower() in (".md", ".qmd", ".rst", ".adoc"):
        yield from chunk_markdown(text)
    else:
        yield from chunk_code(text)
