# TencentDB-Agent-Memory Notes

Source reviewed: `https://github.com/TencentCloud/TencentDB-Agent-Memory`.

Ideas adopted:

- Layered memory beats a flat vector pile.
- Progressive disclosure keeps injected context small while preserving evidence.
- Human-readable Markdown/JSONL artifacts make memory debuggable.
- Hybrid recall is the default path: keyword plus vector retrieval.
- Config should work with sensible defaults and expose deeper knobs only when
  needed.

Ideas not adopted in this version:

- OpenClaw/Hermes-specific plugin architecture.
- SQLite-vec/Tencent Cloud Vector DB backend.
- Context offload/Mermaid compression for tool logs.
- Persona generation with an LLM pipeline.

Future candidates:

- L2 scenario files derived from repeated L1 atoms.
- L3 persona/preferences with conflict detection.
- A Mermaid task canvas for long sessions, still linked to raw refs.
