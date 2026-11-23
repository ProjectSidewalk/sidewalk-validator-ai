"""Shared test setup."""
import os
import tempfile

import pytest

REQUIRE_MLCROISSANT_ENV = "REQUIRE_MLCROISSANT"

# mlcroissant creates its cache (default ~/.cache/croissant) when it is first imported, which staging does when the
# package is installed; point it at a temporary directory before any test can import it.
os.environ.setdefault("CROISSANT_CACHE", tempfile.mkdtemp(prefix="croissant-cache-"))


@pytest.fixture(autouse=True)
def no_experts_from_the_shell(monkeypatch):
    """data_generation.main() reads EXPERTS from the environment; a developer's own value must not reach the tests."""
    monkeypatch.delenv("EXPERTS", raising=False)


def require_mlcroissant():
    """Import and return ``mlcroissant`` for a test that needs it. Without the package the test is skipped, or fails
    when ``REQUIRE_MLCROISSANT=1`` (CI sets it, so the Croissant checks cannot skip there unnoticed).

    ``CROISSANT_CACHE`` already points at a temporary directory (set when this module loads).

    Example::

        def test_something(tmp_path):
            mlc = require_mlcroissant()
            mlc.Dataset(jsonld=path)
    """
    try:
        import mlcroissant
    except ImportError:
        if os.environ.get(REQUIRE_MLCROISSANT_ENV) == "1":
            pytest.fail(f"{REQUIRE_MLCROISSANT_ENV}=1 but mlcroissant is not installed")
        pytest.skip("mlcroissant not installed")
    return mlcroissant
