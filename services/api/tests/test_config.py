from app.config import _optional_secret


def test_empty_secret_is_treated_as_unset(monkeypatch):
    # Regression test for a real bug found running docker-compose: a blank
    # .env entry is passed into the container as "", and os.getenv()
    # returned that empty string as-is — enabling auth with an empty
    # secret, so every request without a key got a 401.
    monkeypatch.setenv("KTP_API_KEY", "")
    assert _optional_secret("KTP_API_KEY") is None


def test_missing_secret_is_unset(monkeypatch):
    monkeypatch.delenv("KTP_API_KEY", raising=False)
    assert _optional_secret("KTP_API_KEY") is None


def test_real_secret_is_kept(monkeypatch):
    monkeypatch.setenv("KTP_API_KEY", "a-real-key")
    assert _optional_secret("KTP_API_KEY") == "a-real-key"
