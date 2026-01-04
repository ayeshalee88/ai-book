"""Unit tests for RetrievedChunk model."""

import pytest
from datetime import datetime
from src.models.retrieved_chunk import RetrievedChunk


class TestRetrievedChunkModel:
    """Test cases for the RetrievedChunk model."""

    def test_retrieved_chunk_creation_valid(self):
        """Test creating a valid RetrievedChunk instance."""
        chunk = RetrievedChunk(
            id="chunk_1",
            text="test content",
            score=0.85,
            position=0
        )
        assert chunk.id == "chunk_1"
        assert chunk.text == "test content"
        assert chunk.score == 0.85
        assert chunk.position == 0
        assert isinstance(chunk.timestamp, datetime)

    def test_retrieved_chunk_with_metadata(self):
        """Test creating a RetrievedChunk with metadata."""
        metadata = {"author": "test", "source": "document.pdf"}
        chunk = RetrievedChunk(
            id="chunk_1",
            text="test content",
            score=0.85,
            position=0,
            metadata=metadata
        )
        assert chunk.metadata == metadata

    def test_retrieved_chunk_score_validation(self):
        """Test that score is validated correctly."""
        # Valid scores
        RetrievedChunk(id="chunk_1", text="test", score=0.0, position=0)  # minimum
        RetrievedChunk(id="chunk_1", text="test", score=1.0, position=0)  # maximum

        # Invalid scores should raise validation error
        with pytest.raises(ValueError):
            RetrievedChunk(id="chunk_1", text="test", score=-0.1, position=0)

        with pytest.raises(ValueError):
            RetrievedChunk(id="chunk_1", text="test", score=1.1, position=0)

    def test_retrieved_chunk_position_validation(self):
        """Test that position is validated correctly."""
        # Valid positions
        RetrievedChunk(id="chunk_1", text="test", score=0.5, position=0)  # minimum
        RetrievedChunk(id="chunk_1", text="test", score=0.5, position=10)  # valid

        # Invalid position should raise validation error
        with pytest.raises(ValueError):
            RetrievedChunk(id="chunk_1", text="test", score=0.5, position=-1)

    def test_retrieved_chunk_optional_fields(self):
        """Test optional fields in RetrievedChunk."""
        chunk = RetrievedChunk(
            id="chunk_1",
            text="test content",
            score=0.85,
            position=0,
            collection_name="test_collection"
        )
        assert chunk.collection_name == "test_collection"