from cc_brain.text import chunk_code, scrub


def test_scrub_pem_private_key_block():
    pem = "-----BEGIN PRIVATE KEY-----\nMIIBVQIBADANBgkqhkiG9w0BAQEFAASCAT8wggE7\n-----END PRIVATE KEY-----"
    out = scrub(f"before {pem} after")
    assert "REDACTED-KEY-BLOCK" in out
    assert "MIIBVQ" not in out


def test_scrub_akia_key():
    text = "aws key AKIAIOSFODNN7EXAMPLE in config"
    out = scrub(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "[REDACTED]" in out


def test_scrub_jwt():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    out = scrub(f"token={jwt}")
    assert jwt not in out
    assert "REDACTED-JWT" in out


def test_scrub_password_assignment():
    out = scrub('password = "hunter2secret"')
    assert "hunter2secret" not in out
    assert "[REDACTED]" in out
    assert out.startswith("password")


def test_scrub_url_credentials():
    out = scrub("connect to postgres://user:pass@host:5432/db")
    assert "pass@" not in out
    assert "postgres://user:[REDACTED]@host" in out


def test_scrub_leaves_plain_code_and_prose_untouched():
    code = 'sk_test_value = "not-a-real-secret-marker"\nprint("hello world")'
    prose = "This is a normal sentence about the weather and nothing sensitive."
    assert scrub(code) == code
    assert scrub(prose) == prose


def test_chunk_code_overlap_first_and_second_chunk():
    lines = [f"line{i}" for i in range(1, 201)]
    text = "\n".join(lines)
    chunks = list(chunk_code(text))
    assert chunks[0][0] == "L1-80"
    assert chunks[1][0].startswith("L61")


def test_chunk_code_last_line_appears_once_per_containing_chunk():
    lines = [f"line{i}" for i in range(1, 201)]
    text = "\n".join(lines)
    chunks = list(chunk_code(text))
    last_line = lines[-1]
    containing = [body for _, _, body in chunks if last_line in body.split("\n")]
    assert len(containing) == 1
    assert containing[0].split("\n").count(last_line) == 1
