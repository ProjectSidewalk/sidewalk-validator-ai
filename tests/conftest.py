"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def no_experts_from_the_shell(monkeypatch):
    """data_generation.main() reads EXPERTS from the environment; a developer's own value must not reach the tests."""
    monkeypatch.delenv("EXPERTS", raising=False)
