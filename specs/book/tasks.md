---
description: "Task list for transforming Physical AI & Humanoid Robotics book into a structured 13-week course"
---

# Tasks: 13-Week Physical AI & Humanoid Robotics Course

**Input**: Design documents from `/specs/book/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/
**Feature**: Transform existing book content into structured 13-week course format

**Tests**: The examples below include test tasks. Tests are OPTIONAL - only include them if explicitly requested in the feature specification.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Documentation**: `docusaurus-book/docs/`, `docusaurus-book/src/` at repository root
- **Content**: `docusaurus-book/docs/` organized by sections
- **Components**: `docusaurus-book/src/components/`
- **Static assets**: `docusaurus-book/static/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure for course format

- [x] T001 Create course-overview directory structure in docusaurus-book/docs/course-overview/
- [x] T002 [P] Update docusaurus.config.js to support course navigation
- [x] T003 [P] Configure course-specific styling in src/css/custom.css
- [x] T004 Set up course metadata and frontmatter standards

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T005 [P] Create syllabus.md with 13-week breakdown and learning outcomes
- [x] T006 [P] Create learning-outcomes.md with detailed competency list
- [x] T007 [P] Create assessments.md with project and capstone details
- [x] T008 [P] Create content-alignment.md mapping existing content to weekly topics
- [x] T009 Update sidebars.js to place Course Overview as first category
- [x] T010 Update homepage hero to emphasize 13-week learning journey
- [x] T011 Add Learning Outcomes card to homepage with proper linking

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Course Overview Implementation (Priority: P1) 🎯 MVP

**Goal**: Implement the complete Course Overview section with syllabus, learning outcomes, and assessments

**Independent Test**: Verify the course structure renders correctly with proper navigation and all overview content is accessible

### Implementation for User Story 1

- [x] T012 [P] [US1] Enhance syllabus.md with Docusaurus admonitions (:::tip, :::note)
- [x] T013 [P] [US1] Enhance learning-outcomes.md with proper tables and formatting
- [x] T014 [P] [US1] Enhance assessments.md with comprehensive grading rubrics
- [x] T015 [US1] Add internal links from introduction.md to course syllabus
- [x] T016 [US1] Add internal links from quickstart.md to course overview
- [x] T017 [US1] Verify all course overview pages render correctly with proper styling
- [x] T018 [US1] Test navigation flow from homepage to course overview sections

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Weekly Content Structure (Priority: P2)

**Goal**: Organize existing book content into the 13-week structure with proper linking

**Independent Test**: Verify each week's content is properly organized and accessible through the course structure

### Implementation for User Story 2

- [x] T019 [P] [US2] Map Week 1-2 content to Introduction and Embodied AI sections
- [x] T020 [P] [US2] Map Week 3-4 content to Sensorimotor Systems and Perception
- [x] T021 [P] [US2] Map Week 5-6 content to Reinforcement Learning for Robot Control
- [x] T022 [P] [US2] Map Week 7-8 content to Sim-to-Real Transfer Techniques
- [x] T023 [P] [US2] Map Week 9-10 content to Humanoid Robot Development fundamentals
- [x] T024 [P] [US2] Map Week 11-12 content to Advanced Manipulation and Interaction
- [x] T025 [P] [US2] Map Week 13 content to Conversational Robotics
- [x] T026 [US2] Create cross-links between weekly content and existing materials
- [x] T027 [US2] Update navigation to support weekly progression

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Assessment Integration (Priority: P3)

**Goal**: Integrate assessment materials with the course structure and create project guidelines

**Independent Test**: Verify all assessment materials are properly linked and accessible from the course structure

### Implementation for User Story 3

- [x] T028 [P] [US3] Create ROS 2 package development project documentation
- [x] T029 [P] [US3] Create Gazebo simulation implementation guidelines
- [x] T030 [P] [US3] Create Isaac-based perception pipeline documentation
- [x] T031 [P] [US3] Create capstone project specifications for humanoid robot
- [x] T032 [US3] Link assessment projects to relevant weekly content
- [x] T033 [US3] Add submission guidelines and evaluation criteria
- [x] T034 [US3] Create project templates and starter code

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - Interactive Course Features (Priority: P2)

**Goal**: Enhance course with interactive elements specifically for the 13-week format

**Independent Test**: Verify interactive course elements work correctly and enhance learning experience

### Implementation for User Story 4

- [x] T035 [P] [US4] Create course progress tracking component
- [x] T036 [P] [US4] Create weekly milestone badges for student achievement
- [x] T037 [P] [US4] Update existing interactive components for course context
- [x] T038 [US4] Integrate progress tracking with existing interactive demos
- [x] T039 [US4] Add course-specific navigation and progress indicators
- [x] T040 [US4] Test interactive elements across different course sections

**Checkpoint**: Interactive elements should work independently and enhance course content

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T041 [P] Add cross-references between course sections and weekly content
- [x] T042 [P] Review and improve accessibility features across all course content
- [x] T043 [P] Optimize images and diagrams for course delivery
- [x] T044 [P] Add search functionality specifically for course navigation
- [x] T045 [P] Create course-specific index and glossary
- [x] T046 [P] Verify all code examples work within the course context
- [x] T047 [P] Add course-specific exercise solutions and resources
- [x] T048 Run complete course validation and update navigation as needed
- [x] T049 Test complete course functionality and navigation flow
- [x] T050 Deploy course to GitHub Pages for final review

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May reference US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May reference US1/US2 but should be independently testable
- **User Story 4 (P2)**: Can start after Foundational (Phase 2) - Can work with any course content

### Within Each User Story

- Core content before interactive elements
- Basic concepts before advanced applications
- Theory before practical examples
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- Content creation within each story can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
   - Developer D: User Story 4 (Interactive elements)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Ensure all code examples are tested and reproducible per constitution
- Verify accessibility compliance for all interactive elements