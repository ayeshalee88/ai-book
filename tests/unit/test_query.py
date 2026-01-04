"""Unit tests for Query model."""

import pytest
from src.models.query import Query


class TestQueryModel:
    """Test cases for the Query model."""

    def test_query_creation_valid(self):
        """Test creating a valid Query instance."""
        query = Query(text="test query", top_k=5)
        assert query.text == "test query"
        assert query.top_k == 5

    def test_query_creation_with_defaults(self):
        """Test creating a Query instance with default values."""
        query = Query(text="test query")
        assert query.text == "test query"
        assert query.top_k == 5  # default value

    def test_query_top_k_validation(self):
        """Test that top_k is validated correctly."""
        # Valid top_k values
        Query(text="test", top_k=1)  # minimum
        Query(text="test", top_k=100)  # maximum

        # Invalid top_k values should raise validation error
        with pytest.raises(ValueError):
            Query(text="test", top_k=0)

        with pytest.raises(ValueError):
            Query(text="test", top_k=101)

    def test_query_with_filters(self):
        """Test creating a Query with filters."""
        filters = {"category": "science", "year": 2023}
        query = Query(text="test query", top_k=5, filters=filters)
        assert query.filters == filters

    def test_query_with_vector(self):
        """Test creating a Query with a pre-computed vector."""
        vector = [0.1, 0.2, 0.3]
        query = Query(text="test query", top_k=5, query_vector=vector)
        assert query.query_vector == vector