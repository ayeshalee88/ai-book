"""Validation service for Qdrant retrieval validation."""

import time
from typing import List, Dict
from src.models.query import Query
from src.models.retrieved_chunk import RetrievedChunk
from src.models.validation_log import ValidationLog
from src.models.performance_metrics import PerformanceMetrics
from src.services.qdrant_service import QdrantService
from src.lib.logging import get_logger, log_validation_result


class ValidationService:
    """Service for validating Qdrant retrieval functionality."""

    def __init__(self, qdrant_service: QdrantService):
        """Initialize the validation service with a Qdrant service."""
        self.qdrant_service = qdrant_service
        self.logger = get_logger("validation_service")

    def validate_retrieval(self, query: Query, log_id: str) -> ValidationLog:
        """Validate retrieval by executing the query and creating a validation log."""
        start_time = time.time()

        try:
            # Execute the query against Qdrant
            retrieved_chunks = self.qdrant_service.search(query)

            # Calculate execution time in milliseconds
            execution_time_ms = (time.time() - start_time) * 1000

            # Create validation log for successful retrieval
            validation_log = ValidationLog(
                id=log_id,
                query=query,
                retrieved_chunks=retrieved_chunks,
                execution_time_ms=execution_time_ms,
                success=True
            )

            self.logger.info(
                f"Validation successful: {len(retrieved_chunks)} chunks retrieved in {execution_time_ms:.2f}ms"
            )

            return validation_log

        except Exception as e:
            # Calculate execution time even when there's an error
            execution_time_ms = (time.time() - start_time) * 1000

            # Create validation log for failed retrieval
            validation_log = ValidationLog(
                id=log_id,
                query=query,
                retrieved_chunks=[],
                execution_time_ms=execution_time_ms,
                success=False,
                error_message=str(e)
            )

            self.logger.error(f"Validation failed: {str(e)}")

            return validation_log

    def validate_top_k_retrieval(self, query: Query, expected_count: int) -> bool:
        """Validate that the correct number of results are returned."""
        # Temporarily modify the top_k to expected_count
        original_top_k = query.top_k
        query.top_k = expected_count

        try:
            validation_log = self.validate_retrieval(query, f"top_k_validation_{int(time.time())}")
            actual_count = len(validation_log.retrieved_chunks)
            return actual_count == expected_count
        finally:
            # Restore original top_k
            query.top_k = original_top_k

    def validate_relevance(self, query: Query, relevance_threshold: float = 0.5) -> bool:
        """Validate that retrieved results meet a minimum relevance threshold."""
        validation_log = self.validate_retrieval(query, f"relevance_validation_{int(time.time())}")

        if not validation_log.retrieved_chunks:
            return False

        # Check if the highest scoring result meets the threshold
        highest_score = max(chunk.score for chunk in validation_log.retrieved_chunks)
        return highest_score >= relevance_threshold

    def validate_configurable_retrieval(self, query: Query, log_id: str) -> ValidationLog:
        """Validate retrieval with configurable parameters."""
        # This method allows for more configurable validation scenarios
        return self.validate_retrieval(query, log_id)

    def validate_with_configurable_top_k(self, query_text: str, top_k_values: List[int]) -> Dict[int, ValidationLog]:
        """Validate retrieval with different top-k values."""
        results = {}
        for k in top_k_values:
            # Create a query with the specific top_k value and default other parameters
            query_copy = Query(
                text=query_text,
                top_k=k
                # filters and query_vector will be None by default
            )
            log_id = f"config_top_k_{k}_{int(time.time())}"
            results[k] = self.validate_retrieval(query_copy, log_id)
        return results

    def collect_performance_metrics(self, validation_logs: List[ValidationLog]) -> PerformanceMetrics:
        """Collect and calculate performance metrics from validation logs."""
        if not validation_logs:
            return PerformanceMetrics(
                id=f"metrics_{int(time.time())}",
                query_latency_ms=0.0,
                success_rate=0.0,
                total_queries=0
            )

        total_queries = len(validation_logs)
        successful_queries = sum(1 for log in validation_logs if log.success)
        success_rate = successful_queries / total_queries if total_queries > 0 else 0.0
        error_rate = 1.0 - success_rate

        # Calculate average query latency
        latencies = [log.execution_time_ms for log in validation_logs]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Calculate p95 and p99 latencies (simplified)
        sorted_latencies = sorted(latencies)
        p95_idx = int(0.95 * len(sorted_latencies)) - 1 if len(sorted_latencies) > 0 else 0
        p99_idx = int(0.99 * len(sorted_latencies)) - 1 if len(sorted_latencies) > 0 else 0
        p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0.0
        p99_latency = sorted_latencies[p99_idx] if sorted_latencies else 0.0

        return PerformanceMetrics(
            id=f"metrics_{int(time.time())}",
            query_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            success_rate=success_rate,
            error_rate=error_rate,
            total_queries=total_queries
        )

    def validate_with_performance_tracking(self, query: Query, log_id: str) -> tuple[ValidationLog, PerformanceMetrics]:
        """Validate retrieval and return both validation log and performance metrics."""
        validation_log = self.validate_retrieval(query, log_id)

        # Collect performance metrics (for now, just return metrics for single validation)
        metrics = self.collect_performance_metrics([validation_log])

        return validation_log, metrics

    def perform_comprehensive_pipeline_validation(self, validation_queries: List[Query]) -> 'PipelineReadiness':
        """Perform comprehensive pipeline validation with multiple checks."""
        from src.models.pipeline_readiness import PipelineReadiness, ValidationCheck

        checks = []
        validation_logs = []

        # Check 1: Health check
        try:
            health_ok = self.qdrant_service.health_check()
            checks.append(ValidationCheck(
                name="health_check",
                description="Check if Qdrant service is accessible and healthy",
                success=health_ok
            ))
        except Exception as e:
            checks.append(ValidationCheck(
                name="health_check",
                description="Check if Qdrant service is accessible and healthy",
                success=False,
                details={"error": str(e)}
            ))

        # Check 2: Basic retrieval for each query
        for i, query in enumerate(validation_queries):
            try:
                log_id = f"pipeline_validation_{i}_{int(time.time())}"
                validation_log = self.validate_retrieval(query, log_id)
                validation_logs.append(validation_log)

                checks.append(ValidationCheck(
                    name=f"retrieval_test_{i}",
                    description=f"Test retrieval with query: {query.text[:30]}...",
                    success=validation_log.success,
                    details={
                        "top_k": query.top_k,
                        "results_count": len(validation_log.retrieved_chunks),
                        "execution_time_ms": validation_log.execution_time_ms
                    },
                    execution_time_ms=validation_log.execution_time_ms
                ))
            except Exception as e:
                checks.append(ValidationCheck(
                    name=f"retrieval_test_{i}",
                    description=f"Test retrieval with query: {query.text[:30]}...",
                    success=False,
                    details={"error": str(e)},
                    execution_time_ms=0.0
                ))

        # Check 3: Top-k validation for the first query if we have queries
        if validation_queries:
            try:
                first_query = validation_queries[0]
                top_k_valid = self.validate_top_k_retrieval(first_query, first_query.top_k)
                checks.append(ValidationCheck(
                    name="top_k_validation",
                    description="Validate that correct number of results are returned",
                    success=top_k_valid,
                    details={"expected": first_query.top_k}
                ))
            except Exception as e:
                checks.append(ValidationCheck(
                    name="top_k_validation",
                    description="Validate that correct number of results are returned",
                    success=False,
                    details={"error": str(e)}
                ))

        # Check 4: Relevance validation for the first query if we have queries
        if validation_queries:
            try:
                first_query = validation_queries[0]
                relevance_ok = self.validate_relevance(first_query, relevance_threshold=0.1)
                checks.append(ValidationCheck(
                    name="relevance_check",
                    description="Validate that retrieved results meet minimum relevance threshold",
                    success=relevance_ok
                ))
            except Exception as e:
                checks.append(ValidationCheck(
                    name="relevance_check",
                    description="Validate that retrieved results meet minimum relevance threshold",
                    success=False,
                    details={"error": str(e)}
                ))

        # Calculate overall status
        passed_count = sum(1 for check in checks if check.success)
        total_count = len(checks)
        success_rate = passed_count / total_count if total_count > 0 else 0.0

        overall_status = "ready"
        if success_rate < 0.5:
            overall_status = "not_ready"
        elif success_rate < 1.0:
            overall_status = "partial"

        # Generate recommendations based on failed checks
        recommendations = []
        for check in checks:
            if not check.success:
                if "health" in check.name:
                    recommendations.append("Verify Qdrant service is running and accessible")
                elif "retrieval" in check.name:
                    recommendations.append("Check that Qdrant collection has data and is properly configured")
                elif "top_k" in check.name:
                    recommendations.append("Verify that the number of results matches the requested top_k")
                elif "relevance" in check.name:
                    recommendations.append("Check that documents have sufficient semantic similarity to queries")

        # Collect performance metrics
        performance_metrics = self.collect_performance_metrics(validation_logs) if validation_logs else None

        # Create pipeline readiness report
        readiness = PipelineReadiness(
            id=f"pipeline_readiness_{int(time.time())}",
            overall_status=overall_status,
            checks=checks,
            validation_logs=validation_logs,
            performance_metrics=performance_metrics,
            summary=f"Pipeline validation completed with {passed_count}/{total_count} checks passing ({success_rate:.1%})",
            recommendations=recommendations
        )

        return readiness