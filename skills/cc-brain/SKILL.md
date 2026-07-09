---
name: cc-brain
description: Use when working in a Claude Code environment with cc-brain installed, especially before reading large repos/web docs, resuming a project, adding durable knowledge, or debugging recall/index freshness.
---

# cc-brain Skill

Use the brain before bulk reading.

## First Moves

1. For current state: call MCP `project_state(project)`.
2. For exact identifiers: call MCP `search(query, lex=True)`.
3. For semantic recall: call MCP `search(query)` and then `get(ids)` only for the chunks needed.
4. If recall looks stale: call MCP `doctor()`, then `index()`.

## Growth Rule

Before analyzing a new repository or web source:

- Local repo: `add_repo(path, project=...)`, then `index()`.
- Web page: `add_web(url, project=...)`, then search the captured content.
- Durable decision/gotcha: `note(name, content)`.

The goal is a growing turbovec brain, not repeated raw reads.

## Constraints

- Turbovec is required. A BM25-only brain is unhealthy.
- Runtime mode is `CC_BRAIN_DEVICE=auto|gpu|cpu`; auto prefers GPU.
- If GPU DLLs are missing on Windows, run `cc-brain bootstrap-gpu` or let auto
  vendoring populate `vendor/python`.
- Do not use usage-monitor behavior.
- Do not route through a worker/farm fleet.
- Keep injected context bounded; use MCP drill-down for detail.
