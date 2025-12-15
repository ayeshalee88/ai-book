# Quickstart Guide: Physical AI & Humanoid Robotics Book Development

## Prerequisites

### System Requirements
- Node.js 18.x or higher
- Python 3.8 or higher
- Git version control
- Text editor or IDE of choice

### Development Tools
```bash
# Install Node.js dependencies
npm install

# Install Python dependencies for examples
pip install numpy matplotlib rospy

# ROS installation (if available on system)
# Follow ROS installation guide for your OS
```

## Setting Up the Development Environment

### 1. Clone and Initialize
```bash
# Navigate to the docusaurus-book directory
cd docusaurus-book

# Install dependencies
npm install

# Start development server
npm start
```

### 2. Verify Setup
- Open http://localhost:3000 in your browser
- You should see the Docusaurus welcome page
- Test editing a file to verify hot reloading

## Content Creation Workflow

### 1. Adding New Pages
```bash
# Create a new markdown file in docs/
touch docs/new-topic.md

# Add to sidebar in sidebars.js
```

### 2. Using Interactive Components
```md
import { InteractiveDemo } from '@site/src/components/InteractiveDemo';

# My Topic

Here's an interactive demonstration:

<InteractiveDemo />
```

### 3. Adding Code Examples
```md
## Python Example

```python
import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
```

For more details on syntax, see the Docusaurus documentation.
```

## Building and Deploying

### Local Build
```bash
npm run build
```

### Preview Build
```bash
npm run serve
```

### Deploy to GitHub Pages
```bash
GIT_USER=<your-github-username> npm run deploy
```

## Content Organization

### Directory Structure
```
docs/
├── intro.md                 # Introduction to the book
├── embodied-ai/            # Section 1: Embodied AI fundamentals
│   ├── introduction.md
│   ├── fundamentals.md
│   └── sensorimotor-loops.md
├── humanoid-robotics/      # Section 2: Humanoid robotics concepts
│   ├── design-principles.md
│   ├── kinematics.md
│   └── control-systems.md
└── ...
```

## Quality Standards

### Writing Style
- Use clear, concise language
- Define technical terms when first introduced
- Include practical examples with code
- Link to external resources for deeper dives

### Code Examples
- Keep examples minimal but complete
- Include expected output where applicable
- Add comments explaining key concepts
- Verify all examples run as expected

### Interactive Elements
- Use sparingly but effectively
- Ensure accessibility for all users
- Provide alternative explanations for complex concepts
- Test on multiple devices and browsers

## Next Steps

1. Review the complete specification in `book/spec.md`
2. Examine the implementation plan in `specs/book/plan.md`
3. Start creating content following the outlined structure
4. Use the task list in `specs/book/tasks.md` to track progress