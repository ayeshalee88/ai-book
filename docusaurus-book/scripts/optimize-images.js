/**
 * Image Optimization Script for Physical AI & Humanoid Robotics Book
 *
 * This script provides utilities for optimizing images for web delivery.
 * It can be run to optimize all images in the static/img directory.
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration for image optimization
const OPTIMIZATION_CONFIG = {
  svg: {
    // SVG optimization using SVGO
    command: 'npx svgo --config="{\\"precision\\": 3, \\"plugins\\": [\\"cleanupAttrs\\", \\"removeDoctype\\", \\"removeXMLProcInst\\", \\"removeComments\\", \\"removeMetadata\\", \\"removeTitle\\", \\"removeDesc\\", \\"removeUselessDefs\\", \\"removeXMLNS\\", \\"removeEditorsNSData\\", \\"removeEmptyAttrs\\", \\"removeHiddenElems\\", \\"removeEmptyText\\", \\"removeEmptyContainers\\", \\"cleanupEnableBackground\\", \\"minifyStyles\\", \\"convertPathData\\", \\"convertTransform\\", \\"removeUnknownsAndDefaults\\", \\"cleanupIDs\\", \\"cleanupNumericValues\\", \\"moveElemsAttrsToGroup\\", \\"moveGroupAttrsToElems\\", \\"collapseGroups\\", \\"removeRasterImages\\", \\"mergePaths\\", \\"convertShapeToPath\\", \\"sortAttrs\\", \\"removeDimensions\\", \\"removeAttrs\\", \\"removeElementsByAttr\\", \\"addClassesToSVGElement\\", \\"addAttributesToSVGElement\\", \\"removeStyleElement\\", \\"removeScriptElement\\"]}"'
  },
  jpeg: {
    // JPEG optimization using mozjpeg
    quality: 85
  },
  png: {
    // PNG optimization using optipng
    level: 6
  }
};

/**
 * Find all image files in a directory recursively
 */
function findImageFiles(dir, fileList = []) {
  const files = fs.readdirSync(dir);

  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      findImageFiles(filePath, fileList);
    } else if (isImageFile(file)) {
      fileList.push(filePath);
    }
  });

  return fileList;
}

/**
 * Check if a file is an image
 */
function isImageFile(filename) {
  const imageExtensions = ['.jpg', '.jpeg', '.png', '.svg', '.webp', '.gif'];
  const ext = path.extname(filename).toLowerCase();
  return imageExtensions.includes(ext);
}

/**
 * Optimize SVG files using SVGO
 */
function optimizeSVG(filePath) {
  console.log(`Optimizing SVG: ${filePath}`);

  try {
    execSync(`npx svgo --config="{\\"precision\\": 3, \\"plugins\\": [\\"cleanupAttrs\\", \\"removeDoctype\\", \\"removeXMLProcInst\\", \\"removeComments\\", \\"removeMetadata\\", \\"removeTitle\\", \\"removeDesc\\", \\"removeUselessDefs\\", \\"removeXMLNS\\", \\"removeEditorsNSData\\", \\"removeEmptyAttrs\\", \\"removeHiddenElems\\", \\"removeEmptyText\\", \\"removeEmptyContainers\\", \\"cleanupEnableBackground\\", \\"minifyStyles\\", \\"convertPathData\\", \\"convertTransform\\", \\"removeUnknownsAndDefaults\\", \\"cleanupIDs\\", \\"cleanupNumericValues\\", \\"moveElemsAttrsToGroup\\", \\"moveGroupAttrsToElems\\", \\"collapseGroups\\", \\"removeRasterImages\\", \\"mergePaths\\", \\"convertShapeToPath\\", \\"sortAttrs\\", \\"removeDimensions\\", \\"removeAttrs\\", \\"removeElementsByAttr\\", \\"addClassesToSVGElement\\", \\"addAttributesToSVGElement\\", \\"removeStyleElement\\", \\"removeScriptElement\\"]}" "${filePath}"`);
    console.log(`✓ Optimized: ${filePath}`);
  } catch (error) {
    console.warn(`⚠ Warning: Could not optimize ${filePath}: ${error.message}`);
  }
}

/**
 * Get file size in KB
 */
function getFileSizeKB(filePath) {
  const stats = fs.statSync(filePath);
  return Math.round(stats.size / 1024);
}

/**
 * Main optimization function
 */
function optimizeImages() {
  const imgDir = path.join(__dirname, '..', 'static', 'img');

  if (!fs.existsSync(imgDir)) {
    console.log('No static/img directory found. Creating...');
    fs.mkdirSync(imgDir, { recursive: true });
    return;
  }

  console.log('🔍 Searching for images to optimize...');
  const imageFiles = findImageFiles(imgDir);

  if (imageFiles.length === 0) {
    console.log('✅ No images found to optimize.');
    return;
  }

  console.log(`📦 Found ${imageFiles.length} image(s) to optimize:\n`);

  imageFiles.forEach(filePath => {
    const originalSize = getFileSizeKB(filePath);
    console.log(`- ${filePath} (${originalSize}KB)`);

    const ext = path.extname(filePath).toLowerCase();

    if (ext === '.svg') {
      optimizeSVG(filePath);
    }
    // For other formats, we would use different tools
    // This is a simplified version focusing on SVG since that's what we're using

    const newSize = getFileSizeKB(filePath);
    const reduction = Math.round(((originalSize - newSize) / originalSize) * 100);

    if (originalSize !== newSize) {
      console.log(`  📉 Size reduced by ${reduction}% (${originalSize}KB → ${newSize}KB)\n`);
    }
  });

  console.log('\n✅ Image optimization complete!');
}

// Run optimization if this script is executed directly
if (require.main === module) {
  console.log('🚀 Starting image optimization for Physical AI & Humanoid Robotics Book...\n');
  optimizeImages();
}

module.exports = {
  findImageFiles,
  isImageFile,
  optimizeImages,
  getFileSizeKB
};