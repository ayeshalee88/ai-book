// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Course Overview',
      items: [
        'course-overview/syllabus',
        'course-overview/learning-outcomes',
        'course-overview/assessments',
        'course-overview/content-alignment'
      ],
      collapsed: false,
    },
    'intro',
    {
      type: 'category',
      label: 'Embodied AI',
      items: [
        'embodied-ai/introduction',
        'embodied-ai/fundamentals',
        'embodied-ai/sensorimotor-loops'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Humanoid Robotics',
      items: [
        'humanoid-robotics/design-principles',
        'humanoid-robotics/kinematics',
        'humanoid-robotics/control-systems'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'AI Integration',
      items: [
        'ai-integration/ml-locomotion',
        'ai-integration/rl-applications',
        'ai-integration/cv-interaction'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Case Studies',
      items: [
        'case-studies/boston-dynamics',
        'case-studies/tesla-optimus',
        'case-studies/open-source-projects'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Tutorials',
      items: [
        'tutorials/simulation-environments',
        'tutorials/hardware-integration'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Challenges & Ethics',
      items: [
        'challenges-ethics/safety-considerations',
        'challenges-ethics/human-robot-interaction',
        'challenges-ethics/societal-impact'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Deployment',
      items: [
        'deployment/testing-strategies',
        'deployment/real-world-deployment'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Optimization',
      items: [
        'optimization/image-optimization'
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Navigation',
      items: [
        'navigation/navigation-guide'
      ],
      collapsed: true,
    },
    {
      type: 'category',
      label: 'Index',
      items: [
        'index/concepts-index'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Resources',
      items: [
        'quickstart',
        'resources/exercise-solutions'
      ],
      collapsed: false,
    },
    {
      type: 'category',
      label: 'Advanced Modules',
      items: [
        'module-1-ros2',
        'module-2-digital-twin',
        'module-3-ai-robot-brain',
        'module-4-vision-language-action'
      ],
      collapsed: false,
    },
  ],
};

module.exports = sidebars;