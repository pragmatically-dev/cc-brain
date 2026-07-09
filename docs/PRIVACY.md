# Privacy For Proprietary Code

`cc-brain` can keep proprietary code local at rest, but it cannot change the data
policy of the model provider you use through Claude Code.

## Local Controls In cc-brain

- Put private folders under repo-local `ingest/`.
- `ingest/**` is gitignored by default.
- The brain index is local under `CC_BRAIN_HOME` or `~/.cc-brain`.
- `cc-brain` does not upload its index, source files, notes or embeddings.
- MCP search returns compact snippets; `get(ids)` expands only requested chunks.

## Claude Code Training And Retention

Current Claude Code docs say:

- Consumer Free/Pro/Max users can choose whether Claude data is used to improve
  future models. If the setting is on, Claude Code sessions from those accounts
  can be used for training.
- Commercial users such as Team, Enterprise, API, third-party platforms and
  Claude Gov are not used to train generative models under commercial terms,
  unless the customer explicitly opts in to model improvement.
- Claude Code sends user prompts and model outputs to the configured provider to
  run the model. Local code that Claude is asked to read can become prompt
  content.
- `/feedback` can include code/session history if the user sends it. Disable it
  for proprietary work.

Sources reviewed:

- `https://docs.anthropic.com/en/docs/claude-code/data-usage`
- `https://privacy.anthropic.com/en/articles/10023548-how-long-do-you-store-my-data`
- `https://privacy.anthropic.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data`

## Recommended No-Degradation Settings

These disable nonessential uploads/telemetry/feedback paths while preserving core
Claude Code model functionality:

```powershell
$env:DISABLE_TELEMETRY = "1"
$env:DISABLE_ERROR_REPORTING = "1"
$env:DISABLE_FEEDBACK_COMMAND = "1"
$env:CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY = "1"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
```

For consumer Claude plans, also turn off model improvement in:

```text
https://claude.ai/settings/data-privacy-controls
```

For organizations with strict proprietary-code requirements, prefer Team,
Enterprise, API, Bedrock, Vertex/Google Cloud Agent Platform, Microsoft Foundry,
or a zero-data-retention agreement where available.

## Practical Rule

Index private code locally in `ingest/`, search first, and expand only the chunks
needed. Do not paste full proprietary files into chat unless your account/provider
policy allows that exposure.
