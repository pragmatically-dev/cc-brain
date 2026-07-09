from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path

from .config import ensure_dirs, paths, slug, upsert_source
from .contracts import SourceSpec
from .store import mark_dirty
from .text import scrub


def capture_web(url: str, content: str = "", project: str = "") -> Path:
    p = ensure_dirs(paths())
    digest = hashlib.sha1(url.encode("utf-8", errors="ignore")).hexdigest()[:12]
    host = re.sub(r"^https?://", "", url).split("/", 1)[0]
    name = f"{slug(host, 'web')}__{digest}.md"
    path = p.web / name
    body = content.strip()
    if not body:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw = resp.read(700_000)
        body = raw.decode("utf-8", errors="replace")
    md = f"---\nurl: {url}\nproject: {project}\nprovenance: web\n---\n\n# {url}\n\n{scrub(body)[:250000]}\n"
    path.write_text(md, encoding="utf-8", newline="\n")
    upsert_source(SourceSpec("web", p.web, kind="md", include=r"\.md$", exclude="", project=project, trust=1.05), p)
    mark_dirty("web", path, p)
    return path
