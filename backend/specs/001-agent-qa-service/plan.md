# Implementation Plan: RAG Chatbot – Agent-Based QA Service via OpenRouter

**Branch**: `001-agent-qa-service` | **Date**: 2025-12-19 | **Spec**: [specs/001-agent-qa-service/spec.md](./spec.md)
**Input**: Feature specification from `/specs/[001-agent-qa-service]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build an agent-based question answering service that uses Qdrant retrieval results and generates grounded answers using Gemini models accessed through OpenRouter. The system will implement a tool-based agent loop (retrieve → reason → answer) with clear separation of retrieval, reasoning, and model invocation, exposed via a FastAPI backend.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: FastAPI, OpenAI Agents SDK, Qdrant, OpenRouter API client
**Storage**: Qdrant vector database (external dependency)
**Testing**: pytest
**Target Platform**: Linux server
**Project Type**: web (API backend)
**Performance Goals**: <10 second response time for typical queries
**Constraints**: Answers must be grounded in retrieved content only (no hallucinations), stateless operation
**Scale/Scope**: Single-user API service for AI engineers

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

[No constitution file found, proceeding with standard approach]

## Project Structure

### Documentation (this feature)

```text
specs/001-agent-qa-service/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── rag_agent.py          # Main RAG agent implementation
│   │   └── tools/
│   │       ├── __init__.py
│   │       └── qdrant_retriever.py    # Qdrant retrieval tool
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       └── qa.py             # QA endpoint
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py           # Configuration and settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── qa.py                 # Request/response models
│   └── services/
│       ├── __init__.py
│       └── llm_service.py        # OpenRouter/Gemini integration
└── tests/
    ├── unit/
    │   ├── agents/
    │   ├── api/
    │   └── services/
    ├── integration/
    │   └── test_qa_endpoint.py
    └── contract/
        └── test_api_contracts.py
```

**Structure Decision**: Backend API structure selected to separate concerns between agents, API endpoints, configuration, data models, and services. The RAG agent will use a tool-based approach with Qdrant retriever as an explicit tool.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Custom OpenRouter adapter | OpenRouter uses OpenAI-compatible API but may need custom integration for optimal performance | Direct OpenAI API would limit to OpenAI models only, missing the requirement to use OpenRouter with Gemini |