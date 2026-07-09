from __future__ import annotations

import argparse
import json

from . import __version__
from .config import load_sources, paths
from .indexer import (
    doctor,
    find_near_duplicates,
    get,
    index,
    project_snapshot,
    recent,
    register_repo,
    remove_source,
    search,
    stats,
)
from .installer import install, mcp_command, uninstall
from .memory import capture_transcript, write_note
from .runtime import status as runtime_status
from .runtime import vendor_gpu_runtime
from .store import connect
from .web import capture_web


def _print_hits(hits) -> None:
    if not hits:
        print("(no hits)")
        return
    for h in hits:
        snippet = " ".join(h.text.split())[:180]
        print(f"[{h.id}] {h.score:.3f} {h.source}:{h.path}#{h.loc} project={h.project}")
        print(f"    {snippet}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cc-brain")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor")
    sub.add_parser("sources")

    p_index = sub.add_parser("index")
    p_index.add_argument("--rebuild", action="store_true")

    p_search = sub.add_parser("search")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=6)
    p_search.add_argument("--source", default="")
    p_search.add_argument("--project", default="")
    p_search.add_argument("--lex", action="store_true")

    p_get = sub.add_parser("get")
    p_get.add_argument("ids", nargs="+", type=int)

    p_repo = sub.add_parser("add-repo")
    p_repo.add_argument("path")
    p_repo.add_argument("--name", default="")
    p_repo.add_argument("--project", default="")

    p_web = sub.add_parser("add-web")
    p_web.add_argument("url")
    p_web.add_argument("--project", default="")

    p_note = sub.add_parser("note")
    p_note.add_argument("name")
    p_note.add_argument("content")

    p_cap = sub.add_parser("capture")
    p_cap.add_argument("transcript")
    p_cap.add_argument("--cwd", default="")

    sub.add_parser("install")
    sub.add_parser("bootstrap-gpu")
    sub.add_parser("privacy-env")
    sub.add_parser("mcp-command")
    sub.add_parser("mcp")

    sub.add_parser("stats")
    sub.add_parser("uninstall")

    p_recent = sub.add_parser("recent")
    p_recent.add_argument("--project", default="")
    p_recent.add_argument("-n", type=int, default=10)

    p_rm = sub.add_parser("remove-source")
    p_rm.add_argument("name")

    sub.add_parser("notes")

    p_state = sub.add_parser("project-state")
    p_state.add_argument("project")

    p_dedupe = sub.add_parser("dedupe")
    p_dedupe.add_argument("--threshold", type=float, default=0.95)
    p_dedupe.add_argument("--limit", type=int, default=50)

    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        print(json.dumps(doctor(), indent=2, ensure_ascii=False))
    elif args.cmd == "sources":
        for src in load_sources(paths()):
            print(f"{src.name}: {src.root} kind={src.kind} project={src.project or '-'}")
    elif args.cmd == "index":
        print(index(rebuild=args.rebuild))
    elif args.cmd == "search":
        _print_hits(search(args.query, k=args.k, source=args.source, project=args.project, lex=args.lex))
    elif args.cmd == "get":
        rows = get(args.ids)
        print("\n\n".join(f"===== [{h.id}] {h.source}:{h.path}#{h.loc} =====\n{h.text}" for h in rows))
    elif args.cmd == "add-repo":
        spec = register_repo(args.path, args.name, args.project)
        print(f"registered {spec.name}: {spec.root} project={spec.project}")
    elif args.cmd == "add-web":
        print(capture_web(args.url, project=args.project))
    elif args.cmd == "note":
        print(write_note(args.name, args.content))
    elif args.cmd == "capture":
        print(capture_transcript(args.transcript, args.cwd))
    elif args.cmd == "install":
        print(f"updated {install()}")
        print(mcp_command())
    elif args.cmd == "bootstrap-gpu":
        path = vendor_gpu_runtime()
        rt = runtime_status()
        print(f"vendored GPU runtime into {path}")
        print(json.dumps(rt.__dict__, indent=2, ensure_ascii=False))
    elif args.cmd == "privacy-env":
        print("$env:DISABLE_TELEMETRY = \"1\"")
        print("$env:DISABLE_ERROR_REPORTING = \"1\"")
        print("$env:DISABLE_FEEDBACK_COMMAND = \"1\"")
        print("$env:CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY = \"1\"")
        print("$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = \"1\"")
    elif args.cmd == "mcp-command":
        print(mcp_command())
    elif args.cmd == "mcp":
        from .mcp_server import main as mcp_main
        mcp_main()
    elif args.cmd == "stats":
        print(json.dumps(stats(), indent=2, ensure_ascii=False))
    elif args.cmd == "recent":
        for row in recent(args.project, args.n):
            print(f"{row['mtime']:.0f}  {row['source']:14s} {row['project']:20s} {row['path']}")
    elif args.cmd == "remove-source":
        print(remove_source(args.name))
    elif args.cmd == "notes":
        for f in sorted(paths().notes.glob("*.md")):
            first = f.read_text(encoding="utf-8", errors="replace").strip().splitlines()
            print(f"{f.stem}: {first[0][:100] if first else ''}")
    elif args.cmd == "project-state":
        print(project_snapshot(args.project))
    elif args.cmd == "dedupe":
        con = connect(paths())
        total_embedded = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        if total_embedded == 0:
            print("(no embeddings indexed yet -- run cc-brain index first)")
        else:
            pairs = find_near_duplicates(threshold=args.threshold, limit=args.limit, con=con)
            for cid_a, path_a, cid_b, path_b, cos in pairs:
                print(f"[{cid_a}] {path_a} ~ [{cid_b}] {path_b} ({cos:.3f})")
            print(f"{len(pairs)} candidate duplicate pairs (threshold {args.threshold})")
    elif args.cmd == "uninstall":
        print(f"removed cc-brain hooks from {uninstall()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
