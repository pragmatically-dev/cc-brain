# cc-brain Rules

1. Turbovec is mandatory. BM25 is only the lexical lane in hybrid retrieval.
2. Runtime must support `auto`, `gpu`, and `cpu`; `auto` prefers GPU.
3. Missing Windows CUDA DLLs are vendored locally into `vendor/python` when GPU
   mode needs them, unless `CC_BRAIN_AUTO_VENDOR=0`.
4. Facts are extracted from transcripts, git and files; no LLM writes facts.
5. Every memory abstraction must point to evidence under `memories/l0` or a
   source file path.
6. Hooks must be fail-safe and exit `0` on errors.
7. Any repo or web page consulted for work should be registered/captured before
   deep analysis, then indexed.
8. User data stays local under `CC_BRAIN_HOME` or `~/.cc-brain`.
9. MCP tools must be bounded and robust; no farm fleet, no usage monitor.
