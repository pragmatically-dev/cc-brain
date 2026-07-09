from __future__ import annotations

import hashlib
import html as _html
import re
import urllib.request
from pathlib import Path

from .config import ensure_dirs, paths, slug
from .store import mark_dirty
from .text import scrub

_TAG_BLOCKS = re.compile(r"<(script|style|noscript|svg|head)\b.*?</\1>", re.S | re.I)
_TAGS = re.compile(r"<[^>]+>")
_BLANKS = re.compile(r"\n{3,}")


def _html_to_text(body: str) -> str:
    if "<" not in body[:2000].lower():
        return body
    if not re.search(r"<(html|body|div|p|br|span|h[1-6])\b", body[:4000], re.I):
        return body
    out = _TAG_BLOCKS.sub(" ", body)
    out = re.sub(r"<(br|/p|/div|/h[1-6]|/li|/tr)\b[^>]*>", "\n", out, flags=re.I)
    out = _TAGS.sub(" ", out)
    out = _html.unescape(out)
    out = "\n".join(" ".join(line.split()) for line in out.splitlines())
    return _BLANKS.sub("\n\n", out).strip()


def capture_web(url: str, content: str = "", project: str = "") -> Path:
    p = ensure_dirs(paths())
    digest = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()[:12]
    host = re.sub(r"^https?://", "", url).split("/", 1)[0]
    name = f"{slug(host, 'web')}__{digest}.md"
    path = p.web / name
    body = (content or "").strip()
    if not body:
        req = urllib.request.Request(url, headers={"User-Agent": "cc-brain/0.2 (+local knowledge vault)"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read(700_000)
        except OSError as exc:
            raise RuntimeError(f"fetch failed for {url}: {exc}") from exc
        body = raw.decode("utf-8", errors="replace")
    body = _html_to_text(body)
    md = f"---\nurl: {url}\nproject: {project}\nprovenance: web\n---\n\n# {url}\n\n{scrub(body)[:250000]}\n"
    path.write_text(md, encoding="utf-8", newline="\n")
    mark_dirty("web", path, p)
    return path
