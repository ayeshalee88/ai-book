# Task Breakdown: RAG Chatbot UI for AI Book

**Feature**: RAG Chatbot UI
**Branch**: 001-agent-qa-service
**Spec**: [specs/001-agent-qa-service/spec.md](../spec.md)
**Plan**: [specs/001-agent-qa-service/plan.md](../plan.md)

## Task Categories

### 1. UI Construction
### 2. Selection Handling
### 3. Backend Wiring
### 4. Branding
### 5. Integration

---

## 1. UI Construction

### Task 1.1: Create Core Chat Widget Component
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 4-6 hours
- **Dependencies**: None

**Description**: Create the main ChatWidget component that serves as the container for the entire chat interface.

**Acceptance Criteria**:
- [ ] Component renders as a floating or docked widget
- [ ] Includes open/close functionality
- [ ] Responsive design works on different screen sizes
- [ ] Component state management for open/closed state

**Implementation Steps**:
1. Create `src/components/Chatbot/ChatWidget.tsx`
2. Implement floating/docked positioning
3. Add open/close toggle functionality
4. Implement responsive behavior for mobile/desktop
5. Add basic styling

**Test Scenarios**:
- [ ] Widget appears correctly positioned on page
- [ ] Open/close functionality works
- [ ] Responsive behavior works across screen sizes

---

### Task 1.2: Create Chat Header Component
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 2-3 hours
- **Dependencies**: Task 1.1

**Description**: Create the ChatHeader component with logo and title display.

**Acceptance Criteria**:
- [ ] Custom chatbot logo appears in header
- [ ] Title is displayed properly
- [ ] Header includes minimize/maximize controls
- [ ] Styling matches professional design requirements

**Implementation Steps**:
1. Create `src/components/Chatbot/ChatHeader.tsx`
2. Add logo placeholder with proper styling
3. Implement title display
4. Add minimize/maximize controls
5. Apply professional styling

**Test Scenarios**:
- [ ] Logo appears in header
- [ ] Title is displayed correctly
- [ ] Controls function properly

---

### Task 1.3: Create Message List Component
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 3-4 hours
- **Dependencies**: Task 1.1

**Description**: Create the MessageList component to contain conversation messages.

**Acceptance Criteria**:
- [ ] Container properly displays messages in chronological order
- [ ] Implements scrolling behavior for long conversations
- [ ] Supports both user and assistant messages
- [ ] Proper spacing and layout

**Implementation Steps**:
1. Create `src/components/Chatbot/MessageList.tsx`
2. Implement message container with proper styling
3. Add scrolling functionality
4. Implement message ordering
5. Add styling for proper spacing

**Test Scenarios**:
- [ ] Messages display in correct order
- [ ] Scrolling works for long conversations
- [ ] Different message types are visually distinct

---

### Task 1.4: Create Message Bubble Component
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 4-5 hours
- **Dependencies**: Task 1.3

**Description**: Create the MessageBubble component to display individual messages with proper styling and sources.

**Acceptance Criteria**:
- [ ] User messages styled differently from assistant messages
- [ ] Assistant answers rendered with proper text styling
- [ ] Optional expandable sources section implemented
- [ ] Custom logo appears as assistant avatar
- [ ] Proper text formatting and styling

**Implementation Steps**:
1. Create `src/components/Chatbot/MessageBubble.tsx`
2. Implement different styling for user vs assistant messages
3. Add proper text formatting for assistant answers
4. Implement expandable sources section
5. Add custom logo as assistant avatar
6. Add loading indicator support

**Test Scenarios**:
- [ ] User and assistant messages are visually distinct
- [ ] Expandable sources section works correctly
- [ ] Assistant avatar shows custom logo
- [ ] Loading indicators display properly

---

### Task 1.5: Create Chat Input Component
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 3-4 hours
- **Dependencies**: Task 1.1

**Description**: Create the ChatInput component with text field and submit functionality.

**Acceptance Criteria**:
- [ ] Text input field with proper styling
- [ ] Send button functionality
- [ ] Enter key submission works
- [ ] Text injection from selection works
- [ ] Proper validation and error handling

**Implementation Steps**:
1. Create `src/components/Chatbot/ChatInput.tsx`
2. Implement text input field with styling
3. Add send button with functionality
4. Implement Enter key submission
5. Add text injection capability
6. Add validation and error handling

**Test Scenarios**:
- [ ] Text input works correctly
- [ ] Send button submits messages
- [ ] Enter key submits messages
- [ ] Text injection works properly

---

### Task 1.6: Create Loading Indicator Component
- **Type**: Implementation
- **Priority**: P2
- **Estimate**: 2-3 hours
- **Dependencies**: Task 1.4

**Description**: Create a LoadingIndicator component to show during backend processing.

**Acceptance Criteria**:
- [ ] Visual indicator shows during API calls
- [ ] Proper styling matches professional design
- [ ] Disappears when response is received
- [ ] Accessible to screen readers

**Implementation Steps**:
1. Create `src/components/Chatbot/LoadingIndicator.tsx`
2. Implement visual loading indicator
3. Add proper styling
4. Ensure accessibility compliance
5. Add show/hide functionality

**Test Scenarios**:
- [ ] Loading indicator appears during API calls
- [ ] Disappears when response is received
- [ ] Styling matches design requirements

---

### Task 1.7: Implement Overall Styling
- **Type**: Implementation
- **Priority**: P2
- **Estimate**: 3-4 hours
- **Dependencies**: All UI components

**Description**: Implement overall styling to ensure professional, enterprise-grade design aesthetics.

**Acceptance Criteria**:
- [ ] All components follow consistent design language
- [ ] Professional color scheme applied
- [ ] Proper spacing and typography
- [ ] Responsive behavior across devices
- [ ] Clean, enterprise-grade appearance

**Implementation Steps**:
1. Create `src/styles/chatbot.css`
2. Define color scheme and design tokens
3. Apply consistent styling across all components
4. Implement responsive design
5. Test across different browsers

**Test Scenarios**:
- [ ] All components have consistent styling
- [ ] Design appears professional and clean
- [ ] Responsive behavior works correctly

---

## 2. Selection Handling

### Task 2.1: Create Text Selection Hook
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 4-5 hours
- **Dependencies**: None

**Description**: Create a custom hook to handle text selection globally across the book pages.

**Acceptance Criteria**:
- [ ] Global mouseup listener attached
- [ ] Selected text is captured safely
- [ ] Does not interfere with normal navigation
- [ ] Proper cleanup of event listeners
- [ ] Works across all book pages

**Implementation Steps**:
1. Create `src/hooks/useTextSelection.ts`
2. Implement global mouseup event listener
3. Add logic to capture selected text safely
4. Ensure no interference with normal navigation
5. Implement proper cleanup of event listeners
6. Add error handling

**Test Scenarios**:
- [ ] Text selection is captured on mouse release
- [ ] No interference with normal navigation
- [ ] Event listeners are properly cleaned up
- [ ] Works across different book pages

---

### Task 2.2: Create Text Selection Listener Component
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 2-3 hours
- **Dependencies**: Task 2.1

**Description**: Create a component that uses the text selection hook to capture and process selections.

**Acceptance Criteria**:
- [ ] Component initializes text selection hook
- [ ] Captured text is properly processed
- [ ] Text is injected into chat input
- [ ] No performance impact on page navigation

**Implementation Steps**:
1. Create `src/components/Chatbot/TextSelectionListener.tsx`
2. Use the text selection hook
3. Implement text injection logic
4. Add performance optimization
5. Test with various selection scenarios

**Test Scenarios**:
- [ ] Component initializes without errors
- [ ] Text selection is properly captured
- [ ] Selected text is injected into chat input
- [ ] No performance degradation

---

## 3. Backend Wiring

### Task 3.1: Create Backend Client Service
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 3-4 hours
- **Dependencies**: None

**Description**: Create a service to handle communication with the existing RAG API.

**Acceptance Criteria**:
- [ ] POST requests sent to backend endpoint
- [ ] Request formatted as JSON: {"question": "<string>"}
- [ ] Proper error handling implemented
- [ ] Response parsing implemented
- [ ] Timeout handling included

**Implementation Steps**:
1. Create `src/services/BackendClient.ts`
2. Implement POST request to backend endpoint
3. Format request as required JSON
4. Add error handling
5. Add timeout handling
6. Add response parsing

**Test Scenarios**:
- [ ] POST requests are sent correctly
- [ ] Request format matches requirements
- [ ] Error handling works properly
- [ ] Timeout handling works properly

---

### Task 3.2: Create API Hook
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 3-4 hours
- **Dependencies**: Task 3.1

**Description**: Create a custom hook to manage API communication state and handle loading/success/error states.

**Acceptance Criteria**:
- [ ] Manages loading state during API calls
- [ ] Handles success responses properly
- [ ] Handles error states gracefully
- [ ] Provides appropriate feedback to UI components
- [ ] Implements retry functionality if needed

**Implementation Steps**:
1. Create `src/hooks/useApi.ts`
2. Implement state management for API calls
3. Add loading state handling
4. Add success state handling
5. Add error state handling
6. Implement retry functionality
7. Add appropriate feedback mechanisms

**Test Scenarios**:
- [ ] Loading state is properly managed
- [ ] Success responses are handled correctly
- [ ] Error states are handled gracefully
- [ ] Retry functionality works when needed

---

### Task 3.3: Integrate API with Chat Components
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 4-5 hours
- **Dependencies**: Tasks 1.5, 3.1, 3.2

**Description**: Connect the chat input component with the backend API to send queries and receive responses.

**Acceptance Criteria**:
- [ ] Messages sent to backend API when submitted
- [ ] Loading indicators show during processing
- [ ] Responses displayed in chat interface
- [ ] Error states handled gracefully
- [ ] Source citations displayed when present

**Implementation Steps**:
1. Integrate API hook with ChatInput component
2. Implement message sending functionality
3. Add loading indicator integration
4. Implement response display
5. Add error handling
6. Implement source citation display
7. Test end-to-end flow

**Test Scenarios**:
- [ ] Messages are sent to backend when submitted
- [ ] Loading indicators appear during processing
- [ ] Responses are displayed correctly
- [ ] Error states are handled gracefully
- [ ] Source citations are displayed when present

---

## 4. Branding

### Task 4.1: Add Logo Assets
- **Type**: Implementation
- **Priority**: P2
- **Estimate**: 1-2 hours
- **Dependencies**: None

**Description**: Add custom chatbot logo assets to the project.

**Acceptance Criteria**:
- [ ] Logo assets added to appropriate directory
- [ ] Multiple formats provided (SVG, PNG)
- [ ] Appropriate sizes included
- [ ] Assets optimized for web use

**Implementation Steps**:
1. Create `static/img/chatbot-logo.*` directory
2. Add logo in SVG format
3. Add PNG fallbacks in appropriate sizes
4. Optimize assets for web use
5. Update documentation with asset locations

**Test Scenarios**:
- [ ] Logo assets are accessible
- [ ] Different formats load correctly
- [ ] Assets are properly optimized

---

### Task 4.2: Apply Logo to UI Components
- **Type**: Implementation
- **Priority**: P2
- **Estimate**: 2-3 hours
- **Dependencies**: Tasks 1.2, 1.4, 4.1

**Description**: Implement the custom logo in the chat header and assistant message avatar.

**Acceptance Criteria**:
- [ ] Custom logo appears in chat header
- [ ] Custom logo appears as assistant message avatar
- [ ] Logo displays correctly in all states
- [ ] Fallbacks work if logo fails to load

**Implementation Steps**:
1. Update ChatHeader component to use custom logo
2. Update MessageBubble component to use logo as avatar
3. Add fallback mechanisms
4. Test logo display in different contexts
5. Ensure accessibility for logo images

**Test Scenarios**:
- [ ] Logo appears in chat header
- [ ] Logo appears as assistant avatar
- [ ] Fallbacks work if primary logo fails
- [ ] Accessibility attributes are correct

---

## 5. Integration

### Task 5.1: Create Docusaurus Plugin
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 4-6 hours
- **Dependencies**: All UI components

**Description**: Create a Docusaurus plugin to integrate the chatbot globally across all book pages.

**Acceptance Criteria**:
- [ ] Chatbot appears on all book pages
- [ ] No impact on existing content
- [ ] Proper initialization on page load
- [ ] Works with Docusaurus routing
- [ ] Performance impact is minimal

**Implementation Steps**:
1. Create Docusaurus plugin in appropriate directory
2. Implement global mounting of chatbot component
3. Ensure compatibility with Docusaurus routing
4. Add performance optimizations
5. Test across different page types
6. Document integration process

**Test Scenarios**:
- [ ] Chatbot appears on all book pages
- [ ] Existing content is unaffected
- [ ] Component initializes properly
- [ ] Works with Docusaurus routing
- [ ] Performance impact is acceptable

---

### Task 5.2: Add Chatbot to Layout
- **Type**: Implementation
- **Priority**: P1
- **Estimate**: 2-3 hours
- **Dependencies**: Task 5.1

**Description**: Integrate the chatbot component into the Docusaurus layout.

**Acceptance Criteria**:
- [ ] Chatbot is included in Docusaurus layout
- [ ] Does not interfere with existing content
- [ ] Proper positioning relative to other elements
- [ ] Responsive behavior maintained

**Implementation Steps**:
1. Modify Docusaurus layout to include chatbot
2. Ensure proper positioning
3. Test with different page layouts
4. Verify no interference with existing content
5. Add CSS to prevent layout conflicts

**Test Scenarios**:
- [ ] Chatbot appears in layout correctly
- [ ] Existing content is not affected
- [ ] Positioning is appropriate
- [ ] Responsive behavior works correctly

---

### Task 5.3: Implement State Persistence
- **Type**: Implementation
- **Priority**: P2
- **Estimate**: 2-3 hours
- **Dependencies**: Task 3.3

**Description**: Implement conversation history persistence across page navigations.

**Acceptance Criteria**:
- [ ] Conversation history preserved when navigating between pages
- [ ] Data stored in localStorage
- [ ] Proper cleanup and limits implemented
- [ ] No performance impact on navigation

**Implementation Steps**:
1. Create `src/hooks/useChatHistory.ts`
2. Implement localStorage persistence
3. Add data limits and cleanup
4. Integrate with message components
5. Test across page navigations
6. Add error handling for storage issues

**Test Scenarios**:
- [ ] Conversation history is preserved across pages
- [ ] Data is properly stored in localStorage
- [ ] Limits and cleanup work correctly
- [ ] No performance impact on navigation

---

## Task Dependencies Summary

- **Critical Path**: Tasks 1.1, 1.5, 2.1, 3.1, 3.2, 3.3, 5.1, 5.2
- **Parallelizable**: UI components (Tasks 1.2-1.6) can be developed in parallel
- **Blocking**: Backend wiring (Section 3) requires UI components to be at least partially complete

## Success Criteria

- [ ] All UI components implemented and styled professionally
- [ ] Text selection functionality works without interfering with navigation
- [ ] Backend API integration handles all states properly
- [ ] Branding applied consistently throughout UI
- [ ] Chatbot integrated globally across all book pages
- [ ] Zero impact on existing content and performance
- [ ] All acceptance criteria met for individual tasks