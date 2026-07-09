# Install

## Local editable install

```powershell
cd C:\path\to\cc-brain
python -m pip install -e .
cc-brain doctor
cc-brain install
claude mcp add --scope user cc-brain -- python -m cc_brain mcp
```

## Contract

`cc-brain` requires turbovec. If `doctor` reports missing vector dependencies,
fix the Python environment before using the MCP server.

## CPU/GPU

Default: `CC_BRAIN_DEVICE=auto`.

- `auto`: prefer NVIDIA GPU, fallback to CPU when no GPU exists.
- `gpu`: strict GPU, fails if CUDA provider cannot load.
- `cpu`: force CPU embeddings.

Windows GPU bootstrap vendors missing redistributable DLL wheels into
`vendor/python` inside the repo:

```powershell
cc-brain bootstrap-gpu
```

Automatic vendoring is enabled by default when GPU mode needs missing CUDA DLLs.
Disable with `CC_BRAIN_AUTO_VENDOR=0`.

## First index

```powershell
cc-brain add-repo C:\path\to\your-project --project your-project
cc-brain index
cc-brain search "current status next step" --project your-project
```

## Hook-driven growth

After `cc-brain install`, hooks keep the brain current:

- Session start registers the current repo.
- Session end writes L0/L1 memory and marks the vault dirty.
- Edits and commits mark the project dirty.
- WebFetch captures consulted pages into `ingest/web`.
- MCP search indexes dirty sources before recall.
