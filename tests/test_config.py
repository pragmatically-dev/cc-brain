from cc_brain.config import load_sources, save_sources
from cc_brain.contracts import SourceSpec


def test_default_sources_auto_added(brain_home):
    sources = load_sources(brain_home)
    names = {s.name for s in sources}
    assert {"notes", "commits", "sessions", "web", "private-ingest"} <= names


def test_round_trip_save_load(brain_home):
    sources = load_sources(brain_home)
    save_sources(sources, brain_home)
    reloaded = load_sources(brain_home)
    assert {s.name for s in sources} == {s.name for s in reloaded}
    assert len(reloaded) == len(sources)


def test_upsert_source_replaces_by_name(brain_home):
    from cc_brain.config import upsert_source

    upsert_source(SourceSpec("custom", brain_home.home, trust=2.0), brain_home)
    upsert_source(SourceSpec("custom", brain_home.home, trust=5.0), brain_home)
    sources = load_sources(brain_home)
    matches = [s for s in sources if s.name == "custom"]
    assert len(matches) == 1
    assert matches[0].trust == 5.0
