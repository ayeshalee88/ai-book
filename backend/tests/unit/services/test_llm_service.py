"""
Unit tests for OpenRouter LLM Service
"""
import pytest
from unittest.mock import Mock, patch
from openai import APIError
from backend.src.services.llm_service import OpenRouterService


class TestOpenRouterService:
    """
    Test cases for OpenRouterService
    """

    @patch('backend.src.services.llm_service.openai.OpenAI')
    @patch('backend.src.services.llm_service.settings')
    def test_init_with_valid_config(self, mock_settings, mock_openai_client):
        """
        Test that OpenRouterService initializes correctly with valid configuration
        """
        # Arrange
        mock_settings.is_config_valid = True
        mock_settings.openrouter_api_key = "test-api-key"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_settings.openrouter_model = "google/gemini-pro"

        # Act
        service = OpenRouterService()

        # Assert
        assert service.model == "google/gemini-pro"
        mock_openai_client.assert_called_once()

    @patch('backend.src.services.llm_service.settings')
    def test_init_with_invalid_config(self, mock_settings):
        """
        Test that OpenRouterService raises ValueError with invalid configuration
        """
        # Arrange
        mock_settings.is_config_valid = False

        # Act & Assert
        with pytest.raises(ValueError, match="OpenRouter API key is not configured properly"):
            OpenRouterService()

    @patch('backend.src.services.llm_service.openai.OpenAI')
    @patch('backend.src.services.llm_service.settings')
    def test_generate_response_success(self, mock_settings, mock_openai_client):
        """
        Test that generate_response returns a proper response
        """
        # Arrange
        mock_settings.is_config_valid = True
        mock_settings.openrouter_api_key = "test-api-key"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_settings.openrouter_model = "google/gemini-pro"

        mock_client = Mock()
        mock_openai_client.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is the generated response."
        mock_client.chat.completions.create.return_value = mock_response

        service = OpenRouterService()

        # Act
        result = service.generate_response("What is the capital of France?")

        # Assert
        assert result == "This is the generated response."
        mock_client.chat.completions.create.assert_called_once()

    @patch('backend.src.services.llm_service.openai.OpenAI')
    @patch('backend.src.services.llm_service.settings')
    def test_generate_response_with_context_success(self, mock_settings, mock_openai_client):
        """
        Test that generate_response works with context
        """
        # Arrange
        mock_settings.is_config_valid = True
        mock_settings.openrouter_api_key = "test-api-key"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_settings.openrouter_model = "google/gemini-pro"

        mock_client = Mock()
        mock_openai_client.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is the generated response based on context."
        mock_client.chat.completions.create.return_value = mock_response

        service = OpenRouterService()

        # Act
        result = service.generate_response(
            "What is the capital of France?",
            context="France is a country in Europe. Paris is its capital."
        )

        # Assert
        assert result == "This is the generated response based on context."
        # Verify the call was made with the expected messages
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 3  # system message + context + user question

    @patch('backend.src.services.llm_service.openai.OpenAI')
    @patch('backend.src.services.llm_service.settings')
    def test_generate_response_api_error(self, mock_settings, mock_openai_client):
        """
        Test that generate_response handles API errors
        """
        # Arrange
        mock_settings.is_config_valid = True
        mock_settings.openrouter_api_key = "test-api-key"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_settings.openrouter_model = "google/gemini-pro"

        mock_client = Mock()
        mock_openai_client.return_value = mock_client

        mock_client.chat.completions.create.side_effect = APIError("API Error", Mock(), "Rate limit exceeded")

        service = OpenRouterService()

        # Act & Assert
        with pytest.raises(Exception, match="OpenRouter API error: API Error"):
            service.generate_response("What is the capital of France?")

    @patch('backend.src.services.llm_service.openai.OpenAI')
    @patch('backend.src.services.llm_service.settings')
    def test_validate_connection_success(self, mock_settings, mock_openai_client):
        """
        Test that validate_connection returns True when connection is successful
        """
        # Arrange
        mock_settings.is_config_valid = True
        mock_settings.openrouter_api_key = "test-api-key"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_settings.openrouter_model = "google/gemini-pro"

        mock_client = Mock()
        mock_openai_client.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello, I am available."
        mock_client.chat.completions.create.return_value = mock_response

        service = OpenRouterService()

        # Act
        result = service.validate_connection()

        # Assert
        assert result is True

    @patch('backend.src.services.llm_service.openai.OpenAI')
    @patch('backend.src.services.llm_service.settings')
    def test_validate_config(self, mock_settings, mock_openai_client):
        """
        Test that validate_config returns proper configuration status
        """
        # Arrange
        mock_settings.is_config_valid = True
        mock_settings.openrouter_api_key = "test-api-key"
        mock_settings.openrouter_base_url = "https://openrouter.ai/api/v1"
        mock_settings.openrouter_model = "google/gemini-pro"

        service = OpenRouterService()

        # Act
        result = service.validate_config()

        # Assert
        assert result["api_key_configured"] is True
        assert result["model_specified"] is True
        assert result["base_url_valid"] is True
        assert result["all_valid"] is True