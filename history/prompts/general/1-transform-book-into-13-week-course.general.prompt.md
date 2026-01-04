---
id: 1
title: transform-book-into-13-week-course
stage: spec
date: '2025-12-16'
model: claude-sonnet-4-5-20250929
feature: course-overview
branch: main
user: Claude
command: /sp.specify
labels: ["course", "specification", "education"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files: []
tests: []
---

# Transform Book into 13-Week Course on Physical AI & Humanoid Robotics

## User Prompt
Update the book specification to transform it into a structured 13-week course on Physical AI & Humanoid Robotics, while keeping all existing technical depth.

Add a new top-level section: Course Overview with the following pages:
- syllabus.md: Full learning outcomes, weekly breakdown, and assessments.
- learning-outcomes.md: Detailed bullet list of what learners will master.
- assessments.md: Projects and capstone details.

Key content to include exactly:

Learning Outcomes
By the end of this course, you will:
- Understand Physical AI principles and embodied intelligence
- d perception and manipulation
  Reinforcement learning for robot control
  Sim-to-real transfer techniques
- Weeks 11-12: Humanoid Robot Development
  Humanoid robot kinematics and dynamics
  Bipedal locomotion and balance control
  Manipulation and grasping with humanoid hands
  Natural human-robot interaction design
- Week 13: Conversational Robotics
  Integrating GPT models for conversational AI in robots
  Speech recognition and natural language understanding
  Multi-modal interaction: speech, gesture, vision

Assessments
- ROS 2 package development project
- Gazebo simulation implementation
- Isaac-based perception pipeline
- Capstone: Simulated humanoid robot with conversational AI

Place "Course Overview" as the first category in the sidebar.
Update the homepage hero and description to emphasize this is a comprehensive 13-week learning journey.
Add a prominent "Learning Outcomes" card on the homepage linking to learning-outcomes.md.
Ensure all existing chapters align with the weekly topics (cross-reference where possible).
Use clean Markdown tables for the weekly breakdown and assessments.

## Response Summary
Successfully transformed the book specification into a structured 13-week course on Physical AI & Humanoid Robotics while preserving all existing technical depth. Completed the following:

1. Created new Course Overview section with:
   - syllabus.md with weekly breakdown and assessments
   - learning-outcomes.md with detailed learning objectives
   - assessments.md with project details and rubrics
   - content-alignment.md mapping existing chapters to weekly topics

2. Updated homepage to emphasize the 13-week learning journey:
   - Modified hero section to highlight the 13-week course
   - Updated call-to-action to "Start the Course - 13 Weeks"
   - Added Learning Outcomes card to homepage with link to learning-outcomes.md

3. Reorganized sidebar structure:
   - Placed "Course Overview" as the first category in the navigation
   - Maintained all existing content under appropriate categories

4. Created content alignment mapping to ensure all existing technical content is preserved and properly organized within the new course structure.

The transformation maintains all existing technical depth while organizing the material in a pedagogically sound 13-week course format with clear learning outcomes, assessments, and structured progression.