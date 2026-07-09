# cc-brain

**A local-first, turbovec-powered memory and knowledge brain for Claude Code.**

`cc-brain` turns Claude Code from a stateless coding assistant into an agent with
durable project memory. It captures sessions, indexes repositories, remembers
commits, stores web pages you consult, and exposes everything back to Claude
through a compact MCP server.

The goal is simple: clone the repo, install it, point it at your projects, and
let Claude build a useful private brain over time.

## Why cc-brain Exists

Claude is powerful, but every new session normally starts cold. You repeat the
same context, re-open the same files, re-explain the same architecture, and lose
the trail of decisions made across days or weeks.

`cc-brain` fixes that by giving Claude Code a persistent local memory layer:

- It remembers where each project left off.
- It indexes source code and docs into a searchable local vault.
- It captures session summaries with links back to raw evidence.
- It recalls only small, relevant snippets instead of flooding the context.
- It keeps proprietary/project data local by default.

## Core Features

| Capability | What it does |
| --- | --- |
| Mandatory turbovec search | Semantic retrieval over your local brain, fused with BM25 for exact identifiers. |
| Claude Code hooks | Keeps memory fresh automatically on session start, prompt submit, session end and tool use. |
| MCP server | Gives Claude native tools for search, get, note, index, source management and health checks. |
| Layered memory | Stores L0 raw facts and L1 Markdown memory atoms for traceable recall. |
| Repo indexing | Index any local repository with source-aware chunking and project labels. |
| Private ingest | Paste proprietary folders into `ingest/`; they are ignored by git but indexed locally. |
| Web ingest | Capture web pages you consult so external knowledge becomes searchable later. |
| Commit memory | Records successful commits deterministically from git output. |
| CPU/GPU runtime | Defaults to GPU when available, supports forced CPU, and vendors missing Windows CUDA DLLs. |
| Privacy controls | Documents Claude Code data policy implications and prints safe opt-out env vars. |
| No usage monitor | No invasive quota/context monitoring. |
| No worker farm | No local LLM fleet dependency; the product is memory/retrieval only. |

## Architecture

`cc-brain` is contract-driven and local-first.

```text
Claude Code hooks
  -> transcript parser / repo tracker / web capture / commit capture
  -> local evidence + memory files
  -> SQLite BM25 + turbovec vector index
  -> MCP tools + bounded hook recall
  -> Claude gets the right context at the right time
```

Memory is layered:

- **L0 conversation evidence**: deterministic JSON extracted from Claude Code transcripts.
- **L1 session atoms**: compact Markdown memories with facts, files, tools, commits and next step.
- **Knowledge vault**: repositories, notes, commits, web pages and L1 memories indexed with turbovec.
- **Recall layer**: bounded hook injection plus explicit MCP drill-down via `search` and `get`.

This adopts the best lesson from layered memory systems like
TencentDB-Agent-Memory: do not dump everything into a flat vector pile. Keep a
small useful top layer, and preserve a drill-down path to evidence.

## Quick Start

```powershell
git clone <your-fork-url> cc-brain
cd cc-brain
python -m pip install -e .
cc-brain doctor
cc-brain install
claude mcp add --scope user cc-brain -- python -m cc_brain mcp
```

Then index your first project:

```powershell
cc-brain add-repo C:\path\to\your-project --project your-project
cc-brain index
cc-brain search "current status next step" --project your-project
```

## Turbovec Is Required

`cc-brain` is built around turbovec. BM25 is present, but only as the lexical
lane for exact identifiers, filenames, hashes and error strings. A BM25-only
brain is considered unhealthy.

## Embeddings

`cc-brain` auto-selects an embedding model, preferring a multilingual model
(`intfloat/multilingual-e5-large` when the installed `fastembed` supports it —
important since a lot of session content is Spanish), falling back to an
English-only model if no multilingual model is available. Override the choice
with `CC_BRAIN_EMBED_MODEL` (and optionally `CC_BRAIN_EMBED_DIM`). Changing the
model — whether by upgrading `fastembed` or setting the env var — is detected
automatically: the next `index()` wipes stored embeddings and does a full
re-embed of every chunk, so the turbovec index never mixes vectors from two
different models.

Required runtime dependencies are installed by the package:

- `turbovec`
- `fastembed`
- `numpy`
- `mcp`

Check health with:

```powershell
cc-brain doctor
```

## CPU And GPU Support

Default mode is `auto`: prefer NVIDIA GPU when available, otherwise use CPU.

```powershell
$env:CC_BRAIN_DEVICE = "auto" # default, prefer GPU
$env:CC_BRAIN_DEVICE = "gpu"  # strict GPU, fail if CUDA cannot load
$env:CC_BRAIN_DEVICE = "cpu"  # force CPU embeddings
```

On Windows, if GPU mode needs missing CUDA/cuDNN/cuBLAS DLLs, `cc-brain` can
vendor the redistributable Python wheels into the repo-local `vendor/python`
directory and preload DLLs from there.

Manual bootstrap:

```powershell
cc-brain bootstrap-gpu
```

Useful environment variables:

```powershell
$env:CC_BRAIN_AUTO_VENDOR = "0"       # disable automatic GPU runtime downloads
$env:CC_BRAIN_VENDOR_DIR = "D:\vendor\cc-brain-python"
$env:CC_BRAIN_CPU = "1"               # legacy hard CPU override
```

`vendor/python/` is ignored by git. Each clone can self-bootstrap without
committing binary DLLs.

## Claude Code Hooks

`cc-brain install` updates `~/.claude/settings.json` with fail-safe hooks.

| Hook | Behavior |
| --- | --- |
| `SessionStart` | Registers the current repo and injects the last relevant project memory. |
| `UserPromptSubmit` | Adds bounded lexical recall for the current prompt. |
| `SessionEnd` | Captures transcript facts into L0/L1 memory and marks the vault dirty. |
| `PreCompact` | Captures transcript facts before context compaction, same as `SessionEnd`. |
| `PostToolUse` | Tracks edits, records successful commits, captures WebFetch outputs and marks repos dirty. |

Hook failures exit `0`, so a bug in `cc-brain` should not break Claude Code.

## MCP Tools

Run the server with:

```powershell
python -m cc_brain mcp
```

Register it with Claude Code:

```powershell
claude mcp add --scope user cc-brain -- python -m cc_brain mcp
```

Available MCP tools:

| Tool | Purpose |
| --- | --- |
| `search(query, k=6, source='', project='', lex=False)` | Search the brain with turbovec+BM25 hybrid retrieval. |
| `get(ids)` | Expand selected chunk ids to full text (capped at 24 ids). |
| `note(name, content)` | Save durable curated knowledge. |
| `index(rebuild=False)` | Index all registered sources. |
| `sources()` | List registered sources. |
| `add_repo(path, name='', project='')` | Register a local repo or folder. |
| `add_web(url, project='')` | Fetch and store a web page into the local brain. |
| `project_state(project, k=8)` | Deterministic freshest project memory: newest session atom + recent commits + related chunks, not a hope-the-search-finds-it lookup. |
| `stats()` | Brain size and freshness: chunks per source/project, embedding model, last index time. |
| `recent(project='', limit=10)` | Most recently indexed files, newest first — a quick freshness probe. |
| `remove_source(name)` | Unregister a source and delete its chunks from the brain. |
| `doctor()` / `health()` | Inspect sources, embeddings, providers and recommendations. |
| `ping()` | Fast liveness check. |

## CLI Commands

```powershell
cc-brain doctor
cc-brain sources
cc-brain index [--rebuild]
cc-brain search "query" [-k 6] [--project name] [--source name] [--lex]
cc-brain get 123 124
cc-brain add-repo C:\repo --name repo-name --project project-name
cc-brain add-web https://example.com/doc --project project-name
cc-brain note decision-name "Important decision text"
cc-brain capture C:\path\to\transcript.jsonl --cwd C:\repo
cc-brain stats
cc-brain recent [--project name] [-n 10]
cc-brain remove-source source-name
cc-brain notes
cc-brain project-state project-name
cc-brain bootstrap-gpu
cc-brain privacy-env
cc-brain install
cc-brain uninstall
cc-brain mcp-command
```

## Private Proprietary Code

For closed-source or client code, use the repo-local `ingest/` folder.

```text
cc-brain/
  ingest/
    ClientA-backend/
    internal-sdk/
    design-docs/
```

Then run:

```powershell
cc-brain index
cc-brain search "auth retry policy" --project ClientA-backend
```

`ingest/**` is ignored by git, but indexed by `cc-brain` as source
`private-ingest`. The first folder under `ingest/` becomes the project label.

Important: the index is local, but if Claude Code reads or reasons about private
code, selected snippets can become prompt content sent to your configured model
provider. See `docs/PRIVACY.md`.

## Privacy And Training Controls

`cc-brain` does not upload your index, source files, notes or embeddings. It is a
local tool.

Claude Code itself still sends prompts and model outputs to the configured model
provider to run the model. Current Claude Code docs say:

- Consumer Free/Pro/Max users can choose whether their data is used for model
  improvement. If enabled, Claude Code sessions from those accounts can be used
  for training.
- Commercial users such as Team, Enterprise, API, third-party platforms and
  Claude Gov are not used to train generative models under commercial terms
  unless explicitly opted in.
- `/feedback` can include code/session history if the user sends it.

Print recommended nonessential traffic/feedback opt-outs:

```powershell
cc-brain privacy-env
```

It prints:

```powershell
$env:DISABLE_TELEMETRY = "1"
$env:DISABLE_ERROR_REPORTING = "1"
$env:DISABLE_FEEDBACK_COMMAND = "1"
$env:CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY = "1"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
```

For strict proprietary work, prefer commercial Claude plans, API/Bedrock/Vertex
style providers, or zero-data-retention arrangements where available.

## Data Layout

Default data home:

```text
~/.cc-brain
```

Override it with:

```powershell
$env:CC_BRAIN_HOME = "D:\brain"
```

Important directories:

| Path | Purpose |
| --- | --- |
| `~/.cc-brain/data` | SQLite DB, turbovec index and id map. |
| `~/.cc-brain/memories/l0/conversations` | Raw deterministic transcript facts. |
| `~/.cc-brain/memories/l1/atoms` | Markdown memory atoms. |
| `~/.cc-brain/notes` | Curated user/Claude notes. |
| `~/.cc-brain/commits` | Deterministic commit logs. |
| `~/.cc-brain/ingest/web` | Captured web pages. |
| `cc-brain/ingest` | Repo-local private proprietary ingest. |
| `cc-brain/vendor/python` | Optional repo-local GPU runtime wheels/DLLs. |

## Recommended Workflow

1. Install and register MCP.
2. Add your active repo with `add-repo` or let hooks register it.
3. Run `cc-brain index`.
4. Ask Claude to use `project_state` before starting work.
5. Search before reading large trees.
6. Use `get(ids)` only for chunks that matter.
7. Save important decisions with `note`.
8. Put private code under `ingest/` when it should stay out of git.

## What cc-brain Deliberately Does Not Do

- No usage monitor or quota monitor.
- No local worker/farm fleet.
- No cloud database.
- No hidden remote sync.
- No automatic commit of private data.
- No claim that local indexing prevents prompt content from reaching your model
  provider when Claude is asked to reason about that content.

## Project Status

This repository is the clean successor to our earlier internal brain stack. It
vendors the useful ideas into a smaller, installable, contract-driven tool:

- Clean Python package.
- MCP-first interface.
- Hook-driven freshness.
- Mandatory turbovec retrieval.
- CPU/GPU runtime selection.
- Private ingest and privacy guidance.
