from cc_brain.config import load_sources
from cc_brain.web import _html_to_text, capture_web


def test_html_to_text_strips_tags():
    html = (
        "<html><head><style>x{}</style></head><body><h1>Title</h1>"
        "<p>Hello <b>world</b></p><script>evil()</script></body></html>"
    )
    out = _html_to_text(html)
    assert "Title" in out and "Hello world" in out
    assert "<" not in out and "evil" not in out


def test_capture_web_does_not_poison_source_project(brain_home):
    capture_web("https://example.com/a", content="plain text", project="proj-a")
    web = [s for s in load_sources(brain_home) if s.name == "web"]
    assert web and web[0].project == ""


def test_capture_web_frontmatter_has_project(brain_home):
    path = capture_web("https://example.com/b", content="hola", project="proj-b")
    assert "project: proj-b" in path.read_text(encoding="utf-8")
