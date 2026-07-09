# LLM Install Guide for cc-brain

This file is written for Claude Code or another coding agent installing
`cc-brain` on behalf of a user.

Your job is to install it, verify it, configure hooks/MCP, and enforce the hard
usage rules so the brain stays useful and private.

## Hard Rules

1. `cc-brain` requires turbovec. Do not accept a BM25-only install as healthy.
2. Prefer GPU automatically, but support CPU. Use `CC_BRAIN_DEVICE=auto` unless
   the user explicitly asks for `gpu` or `cpu`.
3. Do not enable or recreate usage-monitor behavior.
4. Do not add a worker/farm fleet dependency.
5. Do not commit private data from `ingest/`, `vendor/python/`, `~/.cc-brain`, or
   generated indexes.
6. Before deep-reading a repo or web source, register/capture it and index it.
7. For proprietary code, use repo-local `ingest/` and explain provider privacy
   limits honestly.
8. Hooks must be fail-safe. If hook install fails, fix config; do not leave a
   half-working setup without telling the user.

## Preflight

Run from the `cc-brain` repository root.

```powershell
python --version
git status --short
python -m pip --version
```

If Python is missing or too old, stop and ask the user to install Python 3.11+.

If the worktree is dirty, continue without reverting user changes.

## Install Package

```powershell
python -m pip install -e .
```

Then verify the CLI exists:

```powershell
python -m cc_brain doctor
cc-brain doctor
```

Expected: `vector_available: true`. If false, fix dependencies. Turbovec is not
optional.

## CPU/GPU Runtime

Default runtime is automatic:

```powershell
$env:CC_BRAIN_DEVICE = "auto"
```

Modes:

```powershell
$env:CC_BRAIN_DEVICE = "auto" # prefer GPU when available
$env:CC_BRAIN_DEVICE = "gpu"  # strict GPU, fail if CUDA cannot load
$env:CC_BRAIN_DEVICE = "cpu"  # force CPU embeddings
```

If GPU mode has missing Windows CUDA/cuDNN/cuBLAS DLLs, bootstrap repo-local
runtime files:

```powershell
cc-brain bootstrap-gpu
```

This vendors redistributable Python wheels into `vendor/python/`, which is
gitignored. Do not commit it.

If the user forbids network downloads:

```powershell
$env:CC_BRAIN_AUTO_VENDOR = "0"
```

## Embedding Model

The default model is auto-selected, preferring multilingual
(`intfloat/multilingual-e5-large` when the installed fastembed supports it).
This matters for non-English sessions: an English-only model degrades semantic
recall over Spanish (or other) transcripts and notes.

Override model or dimension explicitly:

```powershell
$env:CC_BRAIN_EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # example: force old English default
$env:CC_BRAIN_EMBED_DIM = "768"                       # only if the model needs it
```

Two things to tell the user up front:

1. The first `cc-brain index` downloads the embedding model (the multilingual
   default is large). If downloads are forbidden, set a smaller/pinned model
   first.
2. Changing the model later is safe: cc-brain records the model in its index
   metadata and the next `index()` automatically re-embeds everything. `doctor`
   reports a warning while model and index are out of sync. Never mix — do not
   hand-edit the database to skip the re-embed.
3. Upgrading from a pre-v0.2 database is also safe: if embeddings exist but the
   model metadata is missing (old versions never recorded it), or the stored
   vector byte-length contradicts the current model dimension (e.g. a crashed
   half-migration), the next `index()` treats it as a model change and re-embeds
   everything. Expect the first index after an upgrade to take longer; that is
   the migration, not a hang.

## Install Hooks

Run:

```powershell
cc-brain install
```

This updates `~/.claude/settings.json` with:

- `SessionStart`
- `UserPromptSubmit`
- `SessionEnd`
- `PreCompact`
- `PostToolUse` for shell/edit/write/web tools

Re-running `install` is idempotent: it updates an existing cc-brain hook entry
in place (even one pointing at a stale Python path) instead of duplicating it.
Run `cc-brain uninstall` to remove only cc-brain's own hook entries, leaving
any unrelated hooks untouched.

Verify the settings file exists and includes commands that run:

```powershell
python -m cc_brain.hooks
```

No stdin is fine; it should exit successfully.

## Register MCP Server

Print the exact command:

```powershell
cc-brain mcp-command
```

Then register it:

```powershell
claude mcp add --scope user cc-brain -- python -m cc_brain mcp
```

If `claude` is unavailable, tell the user to run the printed command in a Claude
Code terminal.

Expected MCP tools:

- `ping`
- `health`
- `doctor`
- `index`
- `search`
- `get`
- `note`
- `sources`
- `add_repo`
- `add_web`
- `project_state`
- `stats`
- `recent`
- `remove_source`

## Privacy Setup For Proprietary Code

Show the user these opt-out variables when they work with closed-source code:

```powershell
cc-brain privacy-env
```

Recommended environment:

```powershell
$env:DISABLE_TELEMETRY = "1"
$env:DISABLE_ERROR_REPORTING = "1"
$env:DISABLE_FEEDBACK_COMMAND = "1"
$env:CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY = "1"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
```

Also tell consumer-plan users to disable model improvement at:

```text
https://claude.ai/settings/data-privacy-controls
```

Secret scrubbing: every indexed source (repos, `ingest/`, transcripts, web
captures) is passed through the secret scrubber before chunks are persisted —
API keys, tokens, JWTs, `password=`/`api_key=` assignments and URL credentials
are stored as `[REDACTED]` and are not searchable. This is a safety net, not a
license to index secrets: rotate anything that was committed to a repo.

Important wording: `cc-brain` keeps its index local, but if Claude reads private
code, selected snippets can still be sent to the configured model provider as
prompt content. For strict proprietary-code requirements, prefer Team,
Enterprise, API, Bedrock, Vertex/Google Cloud Agent Platform, Microsoft Foundry,
or zero-data-retention arrangements where available.

## Private Ingest Folder

For proprietary folders, instruct the user to paste them here:

```text
cc-brain/ingest/
```

Example:

```text
ingest/
  ClientA-backend/
  internal-sdk/
```

Then index:

```powershell
cc-brain index
```

The source appears as `private-ingest`. The first folder under `ingest/` becomes
the project label.

## Add Current Project

If installing from inside a user project or if the user names a project, register
it explicitly:

```powershell
cc-brain add-repo "C:\path\to\repo" --project project-name
cc-brain index
```

If the user has not specified a project, do not guess beyond the current working
directory name.

## First Verification

Run:

```powershell
cc-brain sources
cc-brain index
cc-brain doctor
cc-brain stats
cc-brain search "current status next step" -k 5
cc-brain project-state <project-name>
```

Healthy `doctor` should show:

- `vector_available: true`
- `recommendation: ok`
- `warnings: []` (a sidecar/embedding-count mismatch, a stale embed model,
  a pre-v0.2 DB pending re-embed, or a failed background refresh shows up
  here with the exact fix to run)
- `last_refresh_error: ""` (non-empty means the last background reindex died —
  run `index()` synchronously to reproduce and fix)
- nonzero `chunks` after indexing
- nonzero `embeddings` after indexing
- `CUDAExecutionProvider` when GPU is available and mode is `auto` or `gpu`
- `CPUExecutionProvider` when mode is `cpu`

## Brain Maintenance Commands

- `cc-brain stats` — chunks per source/project, index size, embed model, last
  index time. First stop when recall looks off.
- `cc-brain recent [--project name] [-n 10]` — most recently indexed files;
  freshness probe.
- `cc-brain remove-source <name>` — unregister a source and delete its chunks.
  Use it to prune stale auto-registered repos (every cwd Claude runs in becomes
  a `repo-<name>` source); do not let dead sources accumulate.
- `cc-brain notes` — list saved notes (note names overwrite silently; check
  before reusing a name).
- `cc-brain project-state <project>` — deterministic resume context: newest
  session atom + recent commits + related chunks.
- `cc-brain uninstall` — remove only cc-brain's hooks from
  `~/.claude/settings.json`.

Note on freshness: when the vault is dirty, the MCP `search`/`project_state`
tools trigger the reindex in the background and prepend a
`(vault dirty: index refresh started in background...)` note. That is normal —
do not treat it as an error; re-query a few seconds later if the result looks
stale, or run `index()` explicitly for a synchronous refresh.

If a background refresh FAILS (model download, CUDA, corrupt source, DB error),
the failure is recorded and surfaced instead of being swallowed: the next
`search()`/`project_state()` prepends a
`(warning: last background index refresh FAILED: ...)` line, and
`doctor()`/`health()` report it under `last_refresh_error` and `warnings`.
When you see it: run `index()` synchronously to get the full error, fix the
cause, and confirm the warning clears on the next successful index.

## Hard Usage Rules For Future Claude Sessions

After installation, tell Claude/agents to follow these rules:

1. Start by calling MCP `project_state(project)` when resuming a project.
2. Search with MCP `search()` before large file reads.
3. Use `search(..., lex=True)` for exact identifiers, file names, hashes and
   error strings.
4. Use `get(ids)` only for chunks that matter.
5. When consulting a new repo, call `add_repo(path)` and `index()`.
6. When consulting a useful web page, call `add_web(url)` and search the captured
   source.
7. Save durable decisions, recipes and gotchas with `note(name, content)`.
8. Do not dump huge files into context when a search/get drill-down will do.
9. Do not rely on unindexed private folders; index `ingest/` first.
10. If recall looks stale, run `doctor()` then `index()`.
11. Prefer `project_state` (deterministic) at session resume over hoping a
    lexical search happens to surface the right memory; use `stats()`/`recent()`
    to check freshness before assuming the brain is stale.

## Completion Criteria

Do not declare installation complete until all are true:

- Package imports and CLI work.
- `cc-brain doctor` reports turbovec/vector availability.
- Hooks are installed or a clear blocker is reported.
- MCP registration command has been run or handed to the user.
- At least one source is indexed.
- A search returns results.
- Privacy guidance was shown if proprietary code is involved.

## Troubleshooting

### `vector_available` is false

Reinstall package dependencies:

```powershell
python -m pip install -e .
```

Do not continue as if BM25-only is acceptable.

### GPU exists but CUDA provider is not active

Run:

```powershell
cc-brain bootstrap-gpu
cc-brain doctor
```

If still failing, use CPU mode temporarily:

```powershell
$env:CC_BRAIN_DEVICE = "cpu"
cc-brain doctor
```

Tell the user GPU setup remains unresolved.

### Hooks break Claude Code

Hooks should fail safe. Inspect `~/.claude/settings.json`, verify commands point
to the active Python, and run:

```powershell
python -m cc_brain.hooks
```

Fix paths; do not remove unrelated user hooks. Re-running `cc-brain install`
repairs a stale Python path in place.

### Hooks run but nothing is captured

Enable payload logging and reproduce:

```powershell
$env:CC_BRAIN_DEBUG = "1"
```

Every hook invocation then appends its raw payload to
`~/.cc-brain/logs/hooks.jsonl`. Check that events arrive and that
`transcript_path`/`tool_response` contain what capture expects. Unset the
variable afterwards.

### MCP not visible

Run:

```powershell
cc-brain mcp-command
```

Register with `claude mcp add --scope user ...`, then restart or reconnect MCP
inside Claude Code.
