"""Command-line interface for Qdrant retrieval validation."""

import argparse
import sys
import json
from typing import Optional, List
from src.models.query import Query
from src.services.qdrant_service import QdrantService
from src.services.validation_service import ValidationService
from src.lib.config import QdrantConfig
from src.lib.logging import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Qdrant Retrieval Validation Tool"
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query text for semantic search"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)"
    )

    parser.add_argument(
        "--filters",
        type=str,
        help="JSON string of filters to apply to the query"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (optional)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Qdrant host address (default: localhost)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port number (default: 6333)"
    )

    parser.add_argument(
        "--collection",
        type=str,
        default="documents",
        help="Name of the Qdrant collection (default: documents)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--performance",
        action="store_true",
        help="Show performance metrics"
    )

    parser.add_argument(
        "--multiple-top-k",
        type=str,
        help="Comma-separated list of top-k values to test (e.g., '1,3,5,10')"
    )

    parser.add_argument(
        "--pipeline-readiness",
        action="store_true",
        help="Perform comprehensive pipeline readiness validation"
    )

    parser.add_argument(
        "--readiness-queries",
        type=str,
        help="Comma-separated list of queries to use for pipeline readiness validation (e.g., 'query1,query2,query3')"
    )

    return parser


def parse_filters(filter_str: Optional[str]) -> Optional[dict]:
    """Parse filter string as JSON."""
    if not filter_str:
        return None

    try:
        return json.loads(filter_str)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in filters: {filter_str}", file=sys.stderr)
        return None


def parse_top_k_values(top_k_str: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated top-k values."""
    if not top_k_str:
        return None

    try:
        values = [int(x.strip()) for x in top_k_str.split(',')]
        return values
    except ValueError:
        print(f"Error: Invalid top-k values: {top_k_str}", file=sys.stderr)
        return None


def parse_readiness_queries(queries_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated readiness queries."""
    if not queries_str:
        return None

    try:
        queries = [x.strip() for x in queries_str.split(',')]
        return queries
    except Exception:
        print(f"Error: Invalid readiness queries: {queries_str}", file=sys.stderr)
        return None


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)

    # Create configuration
    config = QdrantConfig(
        host=args.host,
        port=args.port,
        collection_name=args.collection
    )

    try:
        # Initialize services
        qdrant_service = QdrantService(config)
        validation_service = ValidationService(qdrant_service)

        # Parse filters if provided
        filters = parse_filters(args.filters)

        if args.pipeline_readiness:
            # Handle comprehensive pipeline readiness validation
            logger.info("Executing comprehensive pipeline readiness validation")

            # Determine queries to use for readiness validation
            readiness_queries = parse_readiness_queries(args.readiness_queries)
            if not readiness_queries:
                # Use the main query if no specific readiness queries provided
                readiness_queries = [args.query] if args.query else ["test query", "sample query", "validation query"]

            # Create Query objects for each readiness query
            validation_queries = [
                Query(text=q, top_k=args.top_k, filters=filters) for q in readiness_queries
            ]

            # Perform comprehensive pipeline validation
            readiness_report = validation_service.perform_comprehensive_pipeline_validation(validation_queries)

            # Output readiness report
            print(f"📋 Pipeline Readiness Report")
            print(f"📊 Overall Status: {readiness_report.overall_status.upper()}")
            print(f"📈 Success Rate: {readiness_report.success_rate:.1%} ({readiness_report.passed_checks_count}/{len(readiness_report.checks)} checks passed)")
            print(f"📝 Summary: {readiness_report.summary}")

            if readiness_report.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in readiness_report.recommendations:
                    print(f"  • {rec}")

            print(f"\n🔍 Detailed Check Results:")
            for check in readiness_report.checks:
                status = "✅" if check.success else "❌"
                print(f"  {status} {check.name}: {check.description}")
                if check.details:
                    print(f"      Details: {check.details}")

            if readiness_report.performance_metrics:
                print(f"\n⚡ Performance Metrics:")
                metrics = readiness_report.performance_metrics
                print(f"  Avg Latency: {metrics.query_latency_ms:.2f}ms")
                print(f"  Success Rate: {metrics.success_rate:.2f}")
                print(f"  Total Queries: {metrics.total_queries}")
                if metrics.p95_latency_ms:
                    print(f"  P95 Latency: {metrics.p95_latency_ms:.2f}ms")
                if metrics.p99_latency_ms:
                    print(f"  P99 Latency: {metrics.p99_latency_ms:.2f}ms")

        elif args.multiple_top_k:
            # Handle multiple top-k validation
            top_k_values = parse_top_k_values(args.multiple_top_k)
            if not top_k_values:
                print("❌ Invalid top-k values provided", file=sys.stderr)
                sys.exit(1)

            logger.info(f"Executing validation with multiple top-k values: {top_k_values}")

            # Create base query without top_k (will be set per validation)
            base_query = Query(
                text=args.query,
                filters=filters
            )

            results = validation_service.validate_with_configurable_top_k(args.query, top_k_values)

            print(f"📊 Results for multiple top-k validation:")
            for k in top_k_values:
                log = results[k]
                if log.success:
                    print(f"  top_k={k}: {len(log.retrieved_chunks)} results in {log.execution_time_ms:.2f}ms")
                    if args.performance:
                        # Calculate performance metrics for this specific result
                        metrics = validation_service.collect_performance_metrics([log])
                        print(f"    Performance: avg latency {metrics.query_latency_ms:.2f}ms, "
                              f"success rate {metrics.success_rate:.2f}")
                else:
                    print(f"  top_k={k}: Failed - {log.error_message}")
        else:
            # Handle single query validation
            # Create query object
            query = Query(
                text=args.query,
                top_k=args.top_k,
                filters=filters
            )

            logger.info(f"Executing validation query: '{query.text}' (top_k: {query.top_k})")

            if args.performance:
                # Perform validation with performance tracking
                log_id = f"cli_validation_{int(1000 * __import__('time').time())}"
                validation_log, performance_metrics = validation_service.validate_with_performance_tracking(query, log_id)

                # Output results with performance metrics
                if validation_log.success:
                    print(f"✅ Validation successful!")
                    print(f"⏱️  Execution time: {validation_log.execution_time_ms:.2f}ms")
                    print(f"📊 Retrieved {len(validation_log.retrieved_chunks)} chunks:")
                    print(f"📈 Performance: avg latency {performance_metrics.query_latency_ms:.2f}ms, "
                          f"success rate {performance_metrics.success_rate:.2f}, "
                          f"total queries {performance_metrics.total_queries}")

                    for i, chunk in enumerate(validation_log.retrieved_chunks, 1):
                        print(f"  {i}. Score: {chunk.score:.3f} | {chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}")
                        if chunk.metadata:
                            print(f"     Metadata: {chunk.metadata}")
                else:
                    print(f"❌ Validation failed: {validation_log.error_message}")
            else:
                # Perform standard validation
                log_id = f"cli_validation_{int(1000 * __import__('time').time())}"
                validation_log = validation_service.validate_retrieval(query, log_id)

                # Output results
                if validation_log.success:
                    print(f"✅ Validation successful!")
                    print(f"⏱️  Execution time: {validation_log.execution_time_ms:.2f}ms")
                    print(f"📊 Retrieved {len(validation_log.retrieved_chunks)} chunks:")

                    for i, chunk in enumerate(validation_log.retrieved_chunks, 1):
                        print(f"  {i}. Score: {chunk.score:.3f} | {chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}")
                        if chunk.metadata:
                            print(f"     Metadata: {chunk.metadata}")
                else:
                    print(f"❌ Validation failed: {validation_log.error_message}")

    except Exception as e:
        print(f"❌ Error during validation: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()