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

## Design Rules

- Facts are extracted, never invented.
- The MCP server never calls a local worker fleet.
- Hooks mark dirty and capture evidence; indexing can run from MCP/CLI.
- Search is hybrid BM25 + turbovec. BM25 is a lexical lane, not a fallback
  product mode. Missing turbovec is a failed health check.
- Every high-level memory must retain a path back to its evidence.
