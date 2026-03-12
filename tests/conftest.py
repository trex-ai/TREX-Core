"""Shared pytest fixtures for TREX-Core tests."""

import pytest


@pytest.fixture
def sample_participant_config() -> dict:
    """Minimal participant configuration for testing."""
    return {
        "participant_id": "test_participant_001",
        "type": "residential",
        "trader": "basic_trader",
        "storage": None,
    }


@pytest.fixture
def sample_market_config() -> dict:
    """Minimal market configuration for testing."""
    return {
        "market_id": "test_market",
        "type": "MicroTE4",
        "grid": {"price_buy": 0.10, "price_sell": 0.05},
    }
