"""
RAG Agent implementation that uses Qdrant retrieval and OpenRouter LLM
"""
from typing import Dict, Any, Optional, List
from ..services.llm_service import OpenRouterService
from .tools.qdrant_retriever import QdrantRetrieverTool
from ..config.settings import settings


class RAGAgent:
    """
    Agent that implements a tool-based RAG approach:
    1. Retrieve relevant content using Qdrant
    2. Generate answer using LLM with retrieved context
    """

    def __init__(self):
        self.llm_service = OpenRouterService()
        self.retriever_tool = QdrantRetrieverTool(
            #host=settings.qdrant_host,
            
            #port=settings.qdrant_port,
            #collection_name=settings.qdrant_collection_name
        )

    def answer_question(self, question: str, top_k: int = 5, min_score_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Answer a question using the RAG approach

        Args:
            question: The user's question
            top_k: Number of relevant chunks to retrieve
            min_score_threshold: Minimum relevance score threshold

        Returns:
            Dictionary containing the answer and source information
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Starting to process question: {question[:50]}...")

        try:
            # Step 1: Retrieve relevant content from Qdrant with fallback strategies
            # First try to get content using the enhanced retrieval with fallback
            logger.debug(f"Attempting to retrieve content for question with top_k={top_k}, min_score_threshold={min_score_threshold}")
            results = self.retriever_tool.retrieve_with_fallback(question, top_k, min_score_threshold)

            logger.info(f"Retrieved {len(results)} results from Qdrant")

            # Format the retrieved content for the LLM
            if results:
                retrieved_content = self._format_retrieved_content_for_llm(results)
                logger.debug(f"Formatted retrieved content with {len(results)} chunks")
            else:
                retrieved_content = "No relevant content found in the knowledge base."
                logger.warning("No relevant content found in knowledge base")

            # Step 2: Generate answer using the LLM with the retrieved context
            logger.debug("Sending request to LLM service")
            answer = self.llm_service.generate_response(
                prompt=question,
                context=retrieved_content
            )
            logger.info("Successfully received response from LLM service")

            # Step 3: Extract source information from the retrieved results
            sources = self._extract_sources_from_results(results)
            logger.debug(f"Extracted {len(sources)} sources from retrieval results")

            # If no relevant sources were found, provide appropriate answer
            if not results:
                answer = "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing your question or ask about a different topic."
                logger.info("Providing no-content-found response to user")

            logger.info(f"Completed processing question: {question[:30]}..., retrieved {len(sources)} sources")

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "retrieved_chunks_count": len(sources)
            }

        except Exception as e:
            logger.error(f"Error processing question '{question[:30]}...': {str(e)}", exc_info=True)
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "retrieved_chunks_count": 0,
                "error": str(e)
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for monitoring
        """
        # In a real implementation, this would connect to a metrics system like Prometheus
        # For now, we'll return mock metrics based on recent activity
        import time
        return {
            "timestamp": time.time(),
            "metrics": {
                "total_questions_processed": getattr(self, '_questions_processed', 0),
                "avg_response_time": getattr(self, '_avg_response_time', 0.0),
                "successful_retrievals": getattr(self, '_successful_retrievals', 0),
                "failed_retrievals": getattr(self, '_failed_retrievals', 0),
                "avg_retrieval_time": getattr(self, '_avg_retrieval_time', 0.0),
                "successful_llm_calls": getattr(self, '_successful_llm_calls', 0),
                "failed_llm_calls": getattr(self, '_failed_llm_calls', 0),
                "avg_llm_response_time": getattr(self, '_avg_llm_response_time', 0.0),
            }
        }

    def _format_retrieved_content_for_llm(self, results: List[Dict[str, Any]]) -> str:
        """
        Format retrieved content for the LLM with proper grounding information
        """
        if not results:
            return "No relevant content found in the knowledge base."

        formatted_content = "Retrieved content for answering the question:\n\n"
        for i, result in enumerate(results, 1):
            formatted_content += f"Result {i}:\n"
            formatted_content += f"Content: {result['content']}\n"
            formatted_content += f"Relevance Score: {result['score']:.3f}\n"
            formatted_content += f"Metadata: {result['metadata']}\n\n"

        return formatted_content

    def _extract_sources_from_results(self, results: List[Dict[str, Any]]) -> list:
        """
        Extract source information from retrieval results
        """
        sources = []
        for result in results:
            # Extract detailed source information from metadata
            metadata = result.get('metadata', {})

            # Extract additional metadata fields that might be present
            additional_metadata = {k: v for k, v in metadata.items()
                                 if k not in ['source', 'section', 'url', 'title', 'page']}

            source_info = {
                "content_snippet": result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', ''),
                "score": result.get('score', 0),
                "source_metadata": metadata,
                "source_details": {
                    "source": metadata.get('source', 'Unknown'),
                    "section": metadata.get('section', 'Unknown'),
                    "url": metadata.get('url', 'Unknown'),
                    "title": metadata.get('title', 'Unknown'),
                    "page": metadata.get('page', 'Unknown'),
                    "full_content_length": len(result.get('content', '')),
                    "additional_metadata": additional_metadata
                }
            }
            sources.append(source_info)

        return sources

    def _extract_sources(self, question: str, top_k: int = 5) -> list:
        """
        Extract source information from the retrieved content
        """
        try:
            # Use the enhanced source information method from the retriever tool
            detailed_sources = self.retriever_tool.get_detailed_source_info(question, top_k)
            return detailed_sources

        except Exception as e:
            print(f"Error extracting sources: {str(e)}")
            # Fallback to basic source extraction
            try:
                raw_results = self.retriever_tool.retrieve(question, top_k)
                sources = []

                for result in raw_results:
                    metadata = result.get('metadata', {})
                    source_info = {
                        "content_snippet": result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', ''),
                        "score": result.get('score', 0),
                        "source_metadata": metadata,
                        "source_details": {
                            "source": metadata.get('source', 'Unknown'),
                            "section": metadata.get('section', 'Unknown'),
                            "url": metadata.get('url', 'Unknown'),
                            "title": metadata.get('title', 'Unknown'),
                            "page": str(metadata.get('page', 'Unknown')),  # Convert to string to match model
                            "full_content_length": len(result.get('content', ''))
                        }
                    }
                    sources.append(source_info)

                return sources
            except Exception:
                return []

    def validate_source_metadata(self, sources: list) -> Dict[str, Any]:
        """
        Validate source metadata for completeness and quality
        """
        validation_results = {
            "total_sources": len(sources),
            "valid_sources": 0,
            "invalid_sources": 0,
            "validation_details": [],
            "completeness_score": 0.0
        }

        total_possible_validations = 0
        total_passed_validations = 0

        for i, source in enumerate(sources):
            details = source.get('source_details', {})
            is_valid = True
            validation_issues = []
            validation_passed = []

            # Check if required fields exist
            if details.get('source', 'Unknown') == 'Unknown':
                is_valid = False
                validation_issues.append("Missing source identifier")
            else:
                validation_passed.append("Source identifier present")
                total_passed_validations += 1
            total_possible_validations += 1

            if details.get('title', 'Unknown') == 'Unknown':
                is_valid = False
                validation_issues.append("Missing title")
            else:
                validation_passed.append("Title present")
                total_passed_validations += 1
            total_possible_validations += 1

            if details.get('url', 'Unknown') == 'Unknown':
                validation_issues.append("Missing URL (not critical but recommended)")
            else:
                validation_passed.append("URL present")
                total_passed_validations += 1
            total_possible_validations += 1

            if details.get('section', 'Unknown') == 'Unknown':
                validation_issues.append("Missing section information")
            else:
                validation_passed.append("Section present")
                total_passed_validations += 1
            total_possible_validations += 1

            if details.get('page', 'Unknown') == 'Unknown':
                validation_issues.append("Missing page information")
            else:
                validation_passed.append("Page information present")
                total_passed_validations += 1
            total_possible_validations += 1

            # Validate content length
            content_length = details.get('full_content_length', 0)
            if content_length == 0:
                validation_issues.append("Content length is 0")
            else:
                validation_passed.append("Content length valid")
                total_passed_validations += 1
            total_possible_validations += 1

            validation_results["validation_details"].append({
                "source_index": i,
                "is_valid": is_valid,
                "issues": validation_issues,
                "passed_validations": validation_passed,
                "details": details
            })

            if is_valid:
                validation_results["valid_sources"] += 1
            else:
                validation_results["invalid_sources"] += 1

        # Calculate completeness score
        if total_possible_validations > 0:
            validation_results["completeness_score"] = total_passed_validations / total_possible_validations
        else:
            validation_results["completeness_score"] = 1.0

        return validation_results

    def format_source_citations(self, sources: list) -> str:
        """
        Format detailed source citations for response
        """
        if not sources:
            return "No sources available for this answer."

        formatted_citations = "Sources for this answer:\n\n"
        for i, source in enumerate(sources, 1):
            details = source.get('source_details', {})
            formatted_citations += f"Source {i}:\n"
            formatted_citations += f"  Title: {details.get('title', 'N/A')}\n"
            formatted_citations += f"  Source: {details.get('source', 'N/A')}\n"
            formatted_citations += f"  Section: {details.get('section', 'N/A')}\n"
            formatted_citations += f"  URL: {details.get('url', 'N/A')}\n"
            formatted_citations += f"  Page: {details.get('page', 'N/A')}\n"
            formatted_citations += f"  Relevance Score: {source.get('score', 0):.3f}\n"
            formatted_citations += f"  Content Length: {details.get('full_content_length', 0)} characters\n"

            # Add additional metadata if available
            additional_metadata = details.get('additional_metadata', {})
            if additional_metadata:
                formatted_citations += f"  Additional Metadata: {additional_metadata}\n"

            formatted_citations += "\n"

        return formatted_citations

    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that all required services are available
        """
        llm_valid = self.llm_service.validate_connection()
        retriever_valid = self.retriever_tool.validate_connection()

        return {
            "llm_service": llm_valid,
            "retriever_service": retriever_valid,
            "overall": llm_valid and retriever_valid
        }

    def verify_source_citation_accuracy(self, question: str, sources: list, answer: str) -> Dict[str, Any]:
        """
        Verify that the sources cited in the answer actually support the claims made
        """
        verification_results = {
            "question": question,
            "total_sources": len(sources),
            "sources_verified": 0,
            "sources_unverifiable": 0,
            "verification_details": [],
            "overall_confidence": 0.0
        }

        for i, source in enumerate(sources):
            content_snippet = source.get('content_snippet', '')
            source_details = source.get('source_details', {})

            # Check if the answer content can be found in the source
            # This is a basic check - in a real implementation, this would involve more sophisticated NLP
            content_matches = []
            if content_snippet and answer:
                # Check for overlap between answer and source content
                answer_words = set(answer.lower().split())
                source_words = set(content_snippet.lower().split())
                common_words = answer_words.intersection(source_words)

                if len(common_words) > 0:
                    overlap_ratio = len(common_words) / len(answer_words) if len(answer_words) > 0 else 0
                    content_matches.append({
                        "overlap_ratio": overlap_ratio,
                        "common_words_count": len(common_words),
                        "is_supporting": overlap_ratio > 0.1  # At least 10% overlap
                    })
                else:
                    content_matches.append({
                        "overlap_ratio": 0,
                        "common_words_count": 0,
                        "is_supporting": False
                    })

            verification_detail = {
                "source_index": i,
                "source_title": source_details.get('title', 'Unknown'),
                "content_matches": content_matches,
                "is_verified": len(content_matches) > 0 and content_matches[0]["is_supporting"],
                "confidence_score": content_matches[0]["overlap_ratio"] if content_matches else 0
            }

            verification_results["verification_details"].append(verification_detail)

            if verification_detail["is_verified"]:
                verification_results["sources_verified"] += 1
            else:
                verification_results["sources_unverifiable"] += 1

        # Calculate overall confidence
        if len(sources) > 0:
            total_confidence = sum(detail["confidence_score"] for detail in verification_results["verification_details"])
            verification_results["overall_confidence"] = total_confidence / len(sources)
        else:
            verification_results["overall_confidence"] = 0.0

        return verification_results

    def get_source_trustworthiness_score(self, sources: list) -> Dict[str, Any]:
        """
        Calculate a trustworthiness score for the provided sources
        """
        if not sources:
            return {
                "trustworthiness_score": 0.0,
                "factors": [],
                "message": "No sources provided"
            }

        scores = []
        factors = []

        for i, source in enumerate(sources):
            score = 0.0
            source_factors = []

            # Check source reliability factors
            details = source.get('source_details', {})

            # Check if source has proper identification
            if details.get('source', 'Unknown') != 'Unknown':
                score += 0.2
                source_factors.append("Source identifier present")

            # Check if title is meaningful
            title = details.get('title', 'Unknown')
            if title != 'Unknown' and len(title) > 5:
                score += 0.2
                source_factors.append("Meaningful title present")

            # Check if URL is provided and seems valid
            url = details.get('url', 'Unknown')
            if url != 'Unknown' and url.startswith(('http://', 'https://')):
                score += 0.2
                source_factors.append("Valid URL provided")

            # Check content length (longer content may be more reliable)
            content_length = details.get('full_content_length', 0)
            if content_length > 100:  # At least 100 characters
                score += 0.15
                source_factors.append("Sufficient content length")
            elif content_length > 0:
                score += 0.05  # Minimal score for non-zero content

            # Check relevance score
            relevance_score = source.get('score', 0)
            if relevance_score > 0.5:
                score += 0.25
                source_factors.append("High relevance score")
            elif relevance_score > 0.2:
                score += 0.1
                source_factors.append("Moderate relevance score")

            scores.append(score)
            factors.extend([f"Source {i+1}: {factor}" for factor in source_factors])

        # Calculate average trustworthiness
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "trustworthiness_score": avg_score,
            "factors": factors,
            "message": f"Average trustworthiness score: {avg_score:.2f}/1.0"
        }

    def test_source_citation_accuracy(self, test_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test source citation accuracy across different content types
        """
        test_results = {
            "total_tests": len(test_questions),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "accuracy_rate": 0.0
        }

        for i, test_case in enumerate(test_questions):
            question = test_case.get("question", "")
            expected_sources = test_case.get("expected_sources", [])
            content_type = test_case.get("content_type", "general")

            # Get the answer and sources from the RAG agent
            result = self.answer_question(question, top_k=3)
            answer = result.get("answer", "")
            sources = result.get("sources", [])

            # Verify the source citation accuracy
            verification = self.verify_source_citation_accuracy(question, sources, answer)

            test_detail = {
                "test_index": i,
                "question": question,
                "content_type": content_type,
                "sources_found": len(sources),
                "expected_sources_count": len(expected_sources),
                "verification_results": verification,
                "passed": verification.get("overall_confidence", 0) > 0.3  # Threshold for passing
            }

            test_results["test_details"].append(test_detail)

            if test_detail["passed"]:
                test_results["passed_tests"] += 1
            else:
                test_results["failed_tests"] += 1

        # Calculate accuracy rate
        if test_results["total_tests"] > 0:
            test_results["accuracy_rate"] = test_results["passed_tests"] / test_results["total_tests"]
        else:
            test_results["accuracy_rate"] = 0.0

        return test_results

    def run_source_verification_acceptance_test(self, question: str) -> Dict[str, Any]:
        """
        Run acceptance test for source verification functionality
        """
        # Get the answer and sources
        result = self.answer_question(question)
        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # Run various verification checks
        metadata_validation = self.validate_source_metadata(sources)
        citation_verification = self.verify_source_citation_accuracy(question, sources, answer)
        trustworthiness_score = self.get_source_trustworthiness_score(sources)

        # Determine if the acceptance test passes
        # Criteria: at least some sources with good metadata, reasonable citation accuracy, and decent trustworthiness
        metadata_pass = metadata_validation.get("completeness_score", 0) > 0.5
        citation_pass = citation_verification.get("overall_confidence", 0) > 0.3
        trustworthiness_pass = trustworthiness_score.get("trustworthiness_score", 0) > 0.3
        has_sources = len(sources) > 0

        acceptance_passed = metadata_pass and citation_pass and trustworthiness_pass and has_sources

        return {
            "acceptance_test_passed": acceptance_passed,
            "question": question,
            "answer_provided": bool(answer and answer.strip()),
            "sources_count": len(sources),
            "metadata_validation": metadata_validation,
            "citation_verification": citation_verification,
            "trustworthiness_score": trustworthiness_score,
            "criteria_met": {
                "metadata_validation_pass": metadata_pass,
                "citation_verification_pass": citation_pass,
                "trustworthiness_pass": trustworthiness_pass,
                "has_sources": has_sources
            }
        }

    def validate_response_format_compliance(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the response complies with the expected structured format
        """
        compliance_results = {
            "is_compliant": True,
            "missing_fields": [],
            "field_validations": {},
            "compliance_score": 0.0
        }

        # Required fields according to QAResponse model
        required_fields = ["question", "answer", "sources", "retrieved_chunks_count"]
        optional_fields = ["error"]

        # Check for required fields
        for field in required_fields:
            if field not in response:
                compliance_results["missing_fields"].append(field)
                compliance_results["is_compliant"] = False
                compliance_results["field_validations"][field] = {
                    "present": False,
                    "valid": False,
                    "error": "Field is missing"
                }
            else:
                # Validate field types and content
                value = response[field]
                is_valid = True
                error_msg = None

                if field == "question":
                    if not isinstance(value, str) or not value.strip():
                        is_valid = False
                        error_msg = "Question must be a non-empty string"
                elif field == "answer":
                    if not isinstance(value, str):
                        is_valid = False
                        error_msg = "Answer must be a string"
                elif field == "sources":
                    if not isinstance(value, list):
                        is_valid = False
                        error_msg = "Sources must be a list"
                    else:
                        # Validate each source in the list
                        for i, source in enumerate(value):
                            if not isinstance(source, dict):
                                is_valid = False
                                error_msg = f"Source {i} must be a dictionary"
                                break
                            if "content_snippet" not in source or "score" not in source:
                                is_valid = False
                                error_msg = f"Source {i} missing required fields (content_snippet, score)"
                                break
                elif field == "retrieved_chunks_count":
                    if not isinstance(value, int) or value < 0:
                        is_valid = False
                        error_msg = "Retrieved chunks count must be a non-negative integer"

                compliance_results["field_validations"][field] = {
                    "present": True,
                    "valid": is_valid,
                    "error": error_msg
                }

                if not is_valid:
                    compliance_results["is_compliant"] = False

        # Calculate compliance score (percentage of required fields that are valid)
        total_required = len(required_fields)
        valid_required = sum(1 for field, validation in compliance_results["field_validations"].items()
                           if field in required_fields and validation["valid"])
        compliance_results["compliance_score"] = valid_required / total_required if total_required > 0 else 1.0

        return compliance_results