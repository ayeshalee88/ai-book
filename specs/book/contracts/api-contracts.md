# API Contracts: Physical AI & Humanoid Robotics Book

## Interactive Components API

### Simulation Demo Component
```
Component: SimulationDemo
Props:
  - simulationType: "locomotion" | "manipulation" | "perception"
  - parameters?: {
      robotModel: string,
      environment: string,
      duration: number,
      speed: number
    }
  - onSimulationStart?: () => void
  - onSimulationEnd?: () => void

Methods:
  - start(): void
  - pause(): void
  - reset(): void
  - setParameters(params: object): void

Events:
  - simulation:start
  - simulation:end
  - simulation:error
```

### Code Runner Component
```
Component: CodeRunner
Props:
  - language: "python" | "javascript" | "cpp"
  - code: string
  - dependencies?: string[]
  - readOnly?: boolean
  - showLineNumbers?: boolean
  - onRun?: (result: RunResult) => void

Methods:
  - run(): Promise<RunResult>
  - stop(): void
  - reset(): void

Types:
  RunResult = {
    output: string,
    error?: string,
    executionTime: number,
    success: boolean
  }
```

### Visualization Component
```
Component: RobotKinematicsVisualizer
Props:
  - robotModel: RobotModel
  - jointAngles: number[]
  - targetPosition?: Vector3
  - showWorkspace?: boolean
  - interactive?: boolean

Methods:
  - updateJointAngles(angles: number[]): void
  - setTargetPosition(position: Vector3): void
  - getForwardKinematics(): Transform[]
  - getInverseKinematics(target: Vector3): number[] | null
```

## Data Exchange Formats

### Configuration File Schema
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "book": {
      "type": "object",
      "properties": {
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "authors": {"type": "array", "items": {"type": "string"}},
        "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"}
      },
      "required": ["title", "authors"]
    },
    "sections": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "title": {"type": "string"},
          "order": {"type": "number"},
          "chapters": {
            "type": "array",
            "items": {"$ref": "#/definitions/chapter"}
          }
        }
      }
    }
  },
  "definitions": {
    "chapter": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "difficulty": {"enum": ["basic", "intermediate", "advanced"]},
        "estimated_time": {"type": "number"},
        "content_file": {"type": "string"},
        "code_examples": {
          "type": "array",
          "items": {"$ref": "#/definitions/code_example"}
        }
      }
    },
    "code_example": {
      "type": "object",
      "properties": {
        "id": {"type": "string"},
        "language": {"type": "string"},
        "file_path": {"type": "string"},
        "description": {"type": "string"},
        "dependencies": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```

## Service Interfaces

### Content Management Service
```
GET /api/content/book
Response: BookMetadata

GET /api/content/section/{sectionId}
Response: SectionContent

GET /api/content/chapter/{chapterId}
Response: ChapterContent

POST /api/content/chapter/{chapterId}/progress
Request: {userId: string, completionPercentage: number}
Response: {success: boolean, timestamp: string}

GET /api/content/search?q={query}
Response: SearchResult[]
```

### Interactive Simulation Service
```
POST /api/simulation/run
Request: {
  type: "kinematics" | "dynamics" | "control",
  parameters: object,
  duration?: number
}
Response: {
  success: boolean,
  results: SimulationResults,
  visualizationData: VisualizationData
}

GET /api/simulation/models
Response: RobotModel[]
```

## Error Handling Contract

### Standard Error Response
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details",
    "timestamp": "ISO 8601 timestamp"
  }
}
```

### Common Error Codes
- `CONTENT_NOT_FOUND`: Requested content does not exist
- `VALIDATION_ERROR`: Request parameters failed validation
- `SIMULATION_FAILED`: Interactive simulation could not be executed
- `PERMISSION_DENIED`: User lacks permission for requested action
- `SERVICE_UNAVAILABLE`: Backend service temporarily unavailable