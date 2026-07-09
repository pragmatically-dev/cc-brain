# Architecture

`cc-brain` is contract-driven: application code depends on contracts, while
SQLite, hooks, MCP and optional vector search are infrastructure adapters.

## Contracts

- `SourceSpec`: a root to index, with include/exclude rules and project label.
- `Chunk`: an indexed unit with source, path, location and text.
- `SearchHit`: retrieval result returned by CLI/MCP/hook recall.
- `HookPayload`: normalized Claude Code hook event.
- `CaptureResult`: files written by a transcript capture pass.

## Layers

- L0: deterministic transcript facts in JSON.
- L1: compact Markdown session atom with facts and next step.
- Vault chunks: project files, notes, commits, web pages and L1 memories.
- Recall: short hook context and explicit MCP drill-down.

## Hooks

`SessionStart`, `UserPromptSubmit`, `SessionEnd`, `PreCompact` and
`PostToolUse` are all fail-safe (exit 0 on any error). `PreCompact` dispatches
to the same transcript capture as `SessionEnd`, so context compaction does not
lose a session's facts. Hooks only mark the vault dirty when something
genuinely changed (e.g. a newly registered repo), not on every prompt, so MCP
tools are not forced into a full reindex on every turn.

## Freshness

When an MCP tool observes the vault is dirty, it kicks off a non-blocking
background index refresh (`mcp_server._index_if_dirty`) and returns
immediately with a note that results may be a few seconds stale, rather than
blocking a `search` call on a multi-minute synchronous reindex.
`project_state` is deterministic: it reads the newest session atom and recent
commit log directly instead of relying on lexical search to surface them.

## Design Rules

- Facts are extracted, never invented.
- The MCP server never calls a local worker fleet.
- Hooks mark dirty and capture evidence; indexing can run from MCP/CLI.
- Search is hybrid BM25 + turbovec. BM25 is a lexical lane, not a fallback
  product mode. Missing turbovec is a failed health check.
- Filters (`source`, `project`) are applied before ranking narrows the
  candidate pool, not after, so a filtered search cannot be starved by an
  unrelated majority of matches.
- Every high-level memory must retain a path back to its evidence.
- Embedding model selection is versioned: the DB records which model/dim
  produced its embeddings, and a model change triggers a full re-embed rather
  than silently mixing vector spaces.
- `remove_source(name)` and `uninstall()` are explicit, reversible-by-reinstall
  ways to shrink the brain or the hook footprint.
