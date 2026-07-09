# Private Ingest

Paste proprietary/local-only repositories or folders here when you want
`cc-brain` to index them without committing them to git.

Example:

```text
ingest/
  ClientA-backend/
  ClosedSourceSDK/
  internal-design-docs/
```

Then run:

```powershell
cc-brain index
cc-brain search "auth retry policy" --project ClientA-backend
```

This directory is ignored by git except for this README and `.gitkeep`.
The index stays local under `CC_BRAIN_HOME` or `~/.cc-brain`.

Important: indexing is local, but if you ask Claude Code to inspect or reason
about proprietary code, that selected content can be sent to your configured LLM
provider as part of the prompt. See `docs/PRIVACY.md`.
