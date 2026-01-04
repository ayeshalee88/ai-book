# ADR-1: Documentation Platform and Technology Stack

## Status
Accepted

## Date
2025-12-16

## Context
We need to create a comprehensive, interactive book on Physical AI & Humanoid Robotics that can effectively bridge digital AI systems with physical robotic platforms. The platform must support complex interactive elements, be accessible to intermediate+ audiences, and be easily maintainable for future updates.

## Decision
We will use the following integrated technology stack:
- **Framework**: Docusaurus for documentation generation and static site hosting
- **Content Format**: Markdown with MDX extensions for interactive components
- **Interactive Elements**: React components embedded in MDX for dynamic content
- **Deployment**: GitHub Pages for public accessibility
- **Programming Languages**: Python 3.8+ for code examples, JavaScript for interactive elements
- **Dependencies**: React for interactive components, Python libraries (numpy, matplotlib, ROS/ROS2)

## Alternatives Considered
- **Alternative 1**: Custom React application with Next.js + Contentful CMS
  - Pros: More flexibility in UI/UX, modern development experience
  - Cons: More complex setup, higher maintenance, requires backend services
- **Alternative 2**: Static site generator like Gatsby or Eleventy
  - Pros: Flexible templating, plugin ecosystem
  - Cons: Less documentation-focused, steeper learning curve for team
- **Alternative 3**: Jupyter Book with interactive notebooks
  - Pros: Built-in interactivity, strong for educational content
  - Cons: Limited styling options, less web-native experience

## Consequences
### Positive
- Docusaurus provides excellent search functionality out of the box
- Strong documentation features with versioning and internationalization support
- MDX allows seamless integration of interactive React components within Markdown
- GitHub Pages deployment is simple and cost-effective
- Strong community support and ecosystem
- Mobile-responsive by default

### Negative
- Learning curve for MDX and React components
- Docusaurus upgrade path may require careful migration
- Some styling customization limitations compared to fully custom solutions

## References
- plan.md: Technical Context section
- research.md: Docusaurus Capabilities for Interactive Content section