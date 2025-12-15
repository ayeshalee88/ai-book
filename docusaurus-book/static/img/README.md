# Images for Physical AI & Humanoid Robotics Book

This directory contains optimized images for the Physical AI & Humanoid Robotics book.

## Directory Structure

- `robotics/` - Robotics-specific diagrams and illustrations
- `diagrams/` - General technical diagrams

## Image Optimization Guidelines

### SVG Files
- Use SVG format for all technical diagrams and illustrations
- Include descriptive `<title>` and `<desc>` elements
- Remove unnecessary metadata and comments
- Use consistent color palette:
  - Primary: `#2e8555` (green for robot links)
  - Secondary: `#555` (gray for robot body)
  - Accent: `#d95757` (red for end effectors)

### File Naming
- Use descriptive, lowercase names with hyphens
- Include subject in directory name (e.g., `robotics/robot-arm.svg`)
- Version numbers if multiple iterations exist (e.g., `robot-arm-v2.svg`)

## Optimization Script

Run the optimization script to optimize all images:

```bash
npm run optimize-images
```

This script will:
- Optimize SVG files using SVGO
- Report file size reductions
- Maintain accessibility features

## Best Practices

1. **Accessibility**: Always include descriptive text in SVG files
2. **Performance**: Keep file sizes minimal while maintaining quality
3. **Consistency**: Use consistent styling across all diagrams
4. **Clarity**: Ensure diagrams are clear and informative