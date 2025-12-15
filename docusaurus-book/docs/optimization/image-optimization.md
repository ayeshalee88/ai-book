---
id: image-optimization
title: Image and Diagram Optimization
sidebar_label: Image Optimization
---

# Image and Diagram Optimization for Web Delivery

This guide outlines best practices for optimizing images and diagrams for the Physical AI & Humanoid Robotics book to ensure fast loading and optimal user experience.

## SVG for Diagrams

For technical diagrams, charts, and illustrations, SVG (Scalable Vector Graphics) is the preferred format because:

- **Scalable**: Maintains quality at any resolution
- **Lightweight**: Small file sizes for simple diagrams
- **Accessible**: Can include descriptive text and titles
- **Responsive**: Adapts to different screen sizes

### SVG Best Practices

1. **Include descriptive titles and descriptions**:
   ```svg
   <svg>
     <title>2-DOF Robot Arm Diagram</title>
     <desc>Simple 2-DOF robot arm with two links and joints</desc>
     <!-- Diagram content -->
   </svg>
   ```

2. **Minimize file size** by removing unnecessary metadata and comments

3. **Use appropriate dimensions** - Don't make SVGs unnecessarily large

## Raster Image Optimization

For photographs and complex images that can't be effectively represented as SVG:

### File Formats
- **WebP**: Best for photographs (smaller than JPEG/PNG with similar quality)
- **JPEG**: Good for photographs when WebP isn't supported
- **PNG**: Best for images with transparency or few colors
- **AVIF**: Emerging format with excellent compression (use as progressive enhancement)

### Compression Guidelines
- **Photographs**: 80-85% quality for good balance of size and quality
- **Screenshots**: 90-95% quality to preserve detail
- **Icons**: Use SVG when possible, PNG for complex icons with transparency

## Image Dimensions and Resizing

- **Match display size**: Don't load large images to display at small sizes
- **Responsive images**: Provide multiple sizes for different screen densities
- **Max width**: Limit images to the container width to prevent layout issues

### Example Responsive Image Implementation
```markdown
![Robot Arm Diagram](/img/robotics/robot_arm.svg)
*Caption: Simple 2-DOF robot arm with two links and joints*
```

## Performance Considerations

### Lazy Loading
- Images below the fold should be lazy-loaded
- Docusaurus handles this automatically for images in documentation

### Preloading Critical Images
- Preload images that are essential for the initial view
- Use appropriate loading strategies for different image types

## Tools for Optimization

### SVG Optimization
- [SVGOMG](https://jakearchibald.github.io/svgomg/): Web-based SVG optimizer
- [svgo](https://github.com/svg/svgo): Command-line SVG optimizer
- Manual cleanup by removing unnecessary metadata

### Raster Image Optimization
- [Squoosh](https://squoosh.app/): Web-based image compressor
- [ImageOptim](https://imageoptim.com/): Desktop image optimizer
- [TinyPNG](https://tinypng.com/): Web-based PNG/JPEG optimizer

## File Organization

```
static/
├── img/
│   ├── robotics/           # Subject-specific images
│   │   ├── robot_arm.svg
│   │   ├── quadruped_robot.svg
│   │   └── sensor_fusion.svg
│   ├── diagrams/           # General diagrams
│   └── logo.svg            # Site logo
```

## Accessibility Considerations

1. **Alt text**: Always provide meaningful alt text for images
2. **Captions**: Include descriptive captions when helpful
3. **Color contrast**: Ensure sufficient contrast in diagrams
4. **Text size**: Keep text in diagrams large enough to read

## Implementation Checklist

- [ ] Images are in the correct format (SVG for diagrams, optimized raster for photos)
- [ ] File sizes are minimized without sacrificing quality
- [ ] Alt text is descriptive and helpful
- [ ] Images are properly organized in the static directory
- [ ] Responsive behavior is tested across devices
- [ ] Accessibility requirements are met

## Example Implementation

```markdown
import Image from '@theme/IdealImage';

<Image
  img={require('/static/img/robotics/robot_arm.svg')}
  alt="2-DOF Robot Arm Diagram showing two links and joints"
  caption="Simple 2-DOF robot arm with two links and joints"
/>
```

By following these optimization practices, the Physical AI & Humanoid Robotics book will provide fast loading times and excellent user experience across all devices and network conditions.