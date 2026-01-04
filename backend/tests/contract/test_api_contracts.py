"""
Contract tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from backend.src.api.main import app
from backend.src.models.qa import QARequest, QAResponse, HealthCheckResponse


class TestAPIContracts:
    """
    Contract tests to verify API endpoints match their expected contracts
    """

    def setup_method(self):
        """
        Setup method to create test client
        """
        self.client = TestClient(app)

    def test_qa_endpoint_contract(self):
        """
        Test that the /qa endpoint matches the expected contract
        """
        # Test with a sample request
        response = self.client.post(
            "/api/v1/qa",
            json={
                "question": "What is the capital of France?",
                "top_k": 3
            }
        )

        # Verify status code
        assert response.status_code == 200

        # Parse response
        data = response.json()

        # Verify response structure matches QAResponse model
        assert "question" in data
        assert "answer" in data
        assert "sources" in data
        assert "retrieved_chunks_count" in data

        # Verify data types
        assert isinstance(data["question"], str)
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["retrieved_chunks_count"], int)

        # Verify optional fields
        if "error" in data:
            assert isinstance(data["error"], str) or data["error"] is None

        # Verify sources structure
        for source in data["sources"]:
            assert "content_snippet" in source
            assert "score" in source
            assert "source_metadata" in source
            assert isinstance(source["content_snippet"], str)
            assert isinstance(source["score"], (int, float))
            assert isinstance(source["source_metadata"], dict)

    def test_qa_endpoint_request_validation(self):
        """
        Test that the /qa endpoint properly validates requests
        """
        # Test with invalid request (missing required field)
        response = self.client.post(
            "/api/v1/qa",
            json={
                # Missing required "question" field
                "top_k": 3
            }
        )

        # Should return validation error
        assert response.status_code == 422

        # Test with invalid request (wrong data type)
        response = self.client.post(
            "/api/v1/qa",
            json={
                "question": 123,  # Should be string
                "top_k": "invalid"  # Should be integer
            }
        )

        # Should return validation error
        assert response.status_code == 422

    def test_health_endpoint_contract(self):
        """
        Test that the /health endpoint matches the expected contract
        """
        response = self.client.get("/api/v1/health")

        # Verify status code
        assert response.status_code == 200

        # Parse response
        data = response.json()

        # Verify response structure matches HealthCheckResponse model
        assert "status" in data
        assert "services" in data

        # Verify data types
        assert isinstance(data["status"], str)
        assert isinstance(data["services"], dict)

        # Verify status value
        assert data["status"] in ["healthy", "unhealthy"]

    def test_validate_endpoint_contract(self):
        """
        Test that the /validate endpoint matches the expected contract
        """
        response = self.client.get("/api/v1/validate")

        # Verify status code
        assert response.status_code == 200

        # Parse response
        data = response.json()

        # Verify required fields
        assert "validation_result" in data
        assert "message" in data

        # Verify data types
        assert isinstance(data["validation_result"], dict)
        assert isinstance(data["message"], str)

        # Verify validation_result structure
        validation_result = data["validation_result"]
        assert "llm_service" in validation_result
        assert "retriever_service" in validation_result
        assert "overall" in validation_result

        assert isinstance(validation_result["llm_service"], bool)
        assert isinstance(validation_result["retriever_service"], bool)
        assert isinstance(validation_result["overall"], bool)

    def test_root_endpoint_contract(self):
        """
        Test that the root endpoint matches the expected contract
        """
        response = self.client.get("/")

        # Verify status code
        assert response.status_code == 200

        # Parse response
        data = response.json()

        # Verify required fields
        assert "message" in data
        assert "endpoints" in data

        # Verify data types
        assert isinstance(data["message"], str)
        assert isinstance(data["endpoints"], list)

        # Verify endpoints structure
        for endpoint in data["endpoints"]:
            assert "path" in endpoint
            assert "method" in endpoint
            assert "description" in endpoint
            assert isinstance(endpoint["path"], str)
            assert isinstance(endpoint["method"], str)
            assert isinstance(endpoint["description"], str)

    def test_response_model_compliance(self):
        """
        Test that responses comply with the defined Pydantic models
        """
        # Test QA response model compliance
        response = self.client.post(
            "/api/v1/qa",
            json={
                "question": "Test question",
                "top_k": 1
            }
        )
        assert response.status_code == 200
        qa_data = response.json()

        # Validate against QAResponse model
        qa_response = QAResponse(**qa_data)
        assert qa_response.question == qa_data["question"]
        assert qa_response.answer == qa_data["answer"]
        assert qa_response.retrieved_chunks_count == qa_data["retrieved_chunks_count"]

        # Test Health response model compliance
        response = self.client.get("/api/v1/health")
        assert response.status_code == 200
        health_data = response.json()

        # Validate against HealthCheckResponse model
        health_response = HealthCheckResponse(**health_data)
        assert health_response.status == health_data["status"]
        assert health_response.services == health_data["services"]

    def test_request_model_compliance(self):
        """
        Test that the API properly handles requests that comply with QARequest model
        """
        # Valid request should work
        valid_request = QARequest(
            question="What is the capital of France?",
            top_k=3
        )

        response = self.client.post(
            "/api/v1/qa",
            json=valid_request.model_dump()
        )
        # Even if the service is not available, it should accept the request format
        # and return appropriate status (could be 200 or 500 depending on service availability)
        assert response.status_code in [200, 500]