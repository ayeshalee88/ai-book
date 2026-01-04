# Implementation Plan: RAG Chatbot UI for AI Book

**Branch**: `001-agent-qa-service` | **Date**: 2025-12-29 | **Spec**: [specs/001-agent-qa-service/spec.md](../../specs/001-agent-qa-service/spec.md)
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a professional, production-grade RAG chatbot UI that integrates seamlessly with the existing Docusaurus-based AI Book. The solution will be a client-side React component that captures text selections, sends queries to the existing backend RAG API, and displays responses with source citations. The UI will feature custom branding, loading states, and enterprise-grade design aesthetics.

## Technical Context

**Language/Version**: TypeScript/JavaScript (compatible with Docusaurus v3.0+)
**Primary Dependencies**: React 18+, ReactDOM, Docusaurus 3.0+ ecosystem
**Storage**: Browser localStorage for conversation history persistence
**Testing**: Jest, React Testing Library for component testing
**Target Platform**: Web browsers (Chrome, Firefox, Safari, Edge)
**Project Type**: Web frontend component (integrated with Docusaurus)
**Performance Goals**: <200ms response time for UI interactions, <5s for API responses
**Constraints**: Client-side only (no server-side rendering requirements), lightweight (minimize bundle size), compatible with Docusaurus
**Scale/Scope**: Single-page application component, supports multiple concurrent users per session

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- ✅ Production-Grade User Experience: Plan includes professional UI patterns and responsive design
- ✅ Seamless Backend Integration: Plan uses existing backend RAG API without modifications
- ✅ Claude CLI-Driven Development: Plan can be implemented using Claude CLI tools
- ✅ Zero-Configuration Deployment: Component will work within existing Docusaurus deployment
- ✅ Performance and Scalability Focus: Plan emphasizes lightweight implementation
- ✅ Security and Privacy First: Plan includes secure API communication practices

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

### Source Code (integrated with Docusaurus)

```text
src/
├── components/
│   └── Chatbot/
│       ├── ChatWidget.tsx          # Main container component
│       ├── ChatHeader.tsx          # Header with logo and title
│       ├── MessageList.tsx         # Container for message bubbles
│       ├── MessageBubble.tsx       # Individual message display
│       ├── ChatInput.tsx           # Input field with send button
│       ├── TextSelectionListener.tsx # Global text selection handler
│       └── LoadingIndicator.tsx    # Loading state component
├── services/
│   └── BackendClient.ts            # API abstraction layer
├── hooks/
│   ├── useTextSelection.ts         # Text selection logic
│   ├── useChatHistory.ts           # Conversation state management
│   └── useApi.ts                   # API communication hooks
└── styles/
    └── chatbot.css                 # Custom styling
```

**Structure Decision**: Integrated with existing Docusaurus structure as client-side React components. The chatbot will be implemented as a floating widget that can be included in Docusaurus layouts.

## Architecture Design

### Core Components

1. **ChatWidget** (container)
   - Main wrapper component that remains persistent across pages
   - Manages overall state and coordinates child components
   - Handles opening/closing the chat interface

2. **ChatHeader** (logo + title)
   - Displays custom chatbot logo in header
   - Shows title and status indicators
   - Contains controls for minimizing/maximizing

3. **MessageList**
   - Container for all conversation messages
   - Implements scrolling behavior
   - Maintains message ordering

4. **MessageBubble** (user / assistant)
   - Displays individual messages with sender differentiation
   - Handles assistant answers with proper text styling
   - Renders expandable sources section when present

5. **ChatInput**
   - Input field with send button
   - Supports Enter key submission
   - Handles text injection from selection

6. **TextSelectionListener** (global)
   - Global event listener for text selection
   - Captures selected text on mouse release
   - Injects text into ChatInput

7. **BackendClient** (API abstraction)
   - Handles communication with backend RAG API
   - Manages request/response formatting
   - Implements error handling

### Event Flow

1. User selects text in book
2. TextSelectionListener captures selection on mouse release
3. Selected text is injected into ChatInput
4. User presses Enter or Send button
5. POST request sent to backend via BackendClient
6. Loading indicator shows during processing
7. Response received from backend
8. Response rendered in chat with optional expandable sources
9. Conversation history updated

### Technology Stack

- **Frontend Framework**: React 18+ with TypeScript
- **Styling**: CSS Modules or Tailwind CSS for component styling
- **State Management**: React hooks (useState, useEffect, useContext)
- **API Communication**: Native fetch API for backend communication
- **Build Tool**: Webpack via Docusaurus (no additional build configuration needed)
- **Development Server**: Docusaurus development server

### Integration Points

- **Docusaurus Layout**: ChatWidget will be integrated into Docusaurus layout components
- **Existing API**: Uses existing backend RAG API endpoint without modifications
- **Asset Management**: Custom logos and branding assets stored in Docusaurus static folder

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Client-side only | Maintain compatibility with existing Docusaurus deployment | Server-side rendering would require backend changes which violates requirement |
