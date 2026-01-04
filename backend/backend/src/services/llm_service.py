"""
LLM Service for OpenRouter integration with Gemini models
"""
from typing import List, Dict, Any, Optional
import openai
import logging
from ..config.settings import settings


class OpenRouterService:
    """
    Service class to handle OpenAI API interactions with Gemini models
    """

    def __init__(self):
        # Validate configuration before initializing
        if not settings.is_config_valid:
            raise ValueError("OpenAI API key is not configured properly")

        # Configure OpenAI client to use OpenRouter
        self.client = openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )
        self.model = settings.openai_model
        self.logger = logging.getLogger(__name__)

    def generate_response(self, prompt: str, context: Optional[str] = None, max_retries: int = 3) -> str:
        """
        Generate a response using the configured model

        Args:
            prompt: The user's question or prompt
            context: Optional context to ground the response
            max_retries: Maximum number of retry attempts (default 3)

        Returns:
            Generated response string
        """
        # Build the full message with context if provided
        messages = []

        # System message to ensure grounding
        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based only on the provided context. Do not make up information. If the context doesn't contain enough information to answer, say so."
        }
        messages.append(system_message)

        # Add context if provided
        if context:
            context_message = {
                "role": "system",
                "content": f"Context for answering the question:\n{context}"
            }
            messages.append(context_message)

        # Add the user's question
        user_message = {
            "role": "user",
            "content": prompt
        }
        messages.append(user_message)

        # Attempt to call the API with retries
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more consistent, fact-based responses
                    max_tokens=1000,
                )

                content = response.choices[0].message.content
                if content is None:
                    raise Exception("LLM returned null response content")

                return content

            except openai.APIError as e:
                last_exception = e
                self.logger.warning(f"OpenRouter API error (attempt {attempt + 1}/{max_retries}): {str(e)}")

                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                last_exception = e
                self.logger.warning(f"Error calling OpenRouter API (attempt {attempt + 1}/{max_retries}): {str(e)}")

                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff

        # If all retries failed, log the final error and return a fallback response
        self.logger.error(f"Failed to get response from LLM after {max_retries} attempts: {str(last_exception)}")
        
        return "I'm sorry, but I'm currently unable to process your request. The language model service is not responding. Please try again later."
    
        if context:
          return f"Based on the available context: {context[:500]}..."
        return "I'm sorry, but I'm currently unable to process your request..."

    def validate_connection(self) -> bool:
        """
        Validate that we can connect to OpenRouter with the provided credentials
        """
        try:
            # Simple test call
            test_response = self.generate_response("Hello, are you available?")
            return len(test_response) > 0
        except Exception as e:
            self.logger.error(f"OpenRouter connection validation failed: {str(e)}")
            return False

    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the configuration settings for OpenRouter
        """
        config_status = {
            "api_key_configured": settings.openai_api_key is not None and len(settings.openai_api_key) > 0,
            "model_specified": settings.openai_model is not None and len(settings.openai_model) > 0,
            "base_url_valid": settings.openai_base_url is not None and len(settings.openai_base_url) > 0
        }

        config_status["all_valid"] = all(config_status.values())
        return config_status