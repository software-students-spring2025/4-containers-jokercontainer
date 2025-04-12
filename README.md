![Web App Status](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)
![ML Client Status](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Audio Processing and Question-Answering System

A containerized system that integrates a web application, a machine learning client and mongodb. 
## Team Members

- [Alan Chen](https://github.com/Chen-zexi)
- [Jackson Chen](https://github.com/jaxxjj)
- [Ray Huang](https://github.com/RayHuang3339)

## Architecture

This project consists of three main components:

1. **Web Application (web-app)**:
- Interactive web interface with audio recording capabilities via HTML/JavaScript frontend
- Flask backend for handling web requests and user interactions
2. **Machine Learning Client (machine-learning-client)**: 
- Speech-to-text transcription using OpenAI models
- Intelligent answer retrieval using langchain and browser use agent
3. **MongoDB**: 
- MongoDB database integration for persistent storage of Q&A history

## Prerequisites

- Docker and Docker Compose (v2.0+)
- OpenAI API key with access to GPT-4o and GPT-4o-transcribe models
- Modern web browser with microphone access permissions

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/nyu-software-engineering/containerized-app-exercise.git
   cd containerized-app-exercise
   ```

2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. Build and start the containers:
   ```bash
   docker compose up --build
   ```

4. Once all containers are running, access the application at:
   ```
   http://localhost:5001
   ```

## Usage

1. Open your browser and go to `http://localhost:5001`
2. Click "Start Recording" to record your question
3. Speak your question clearly and click "Stop Recording" when finished
4. The system will process your audio and show a loading animation
5. When processing is complete, the answer will be displayed in formatted markdown

## Development

### Project Structure

```
.
├── common/                  # Shared code between web-app and ML client
│   └── models.py            # Database models and connection handling
├── machine-learning-client/ # ML service for audio processing
│   ├── Dockerfile           # Container configuration
│   ├── ml_app.py            # Main application code
│   └── requirements.txt     # Python dependencies
├── web-app/                 # Web application for user interface
│   ├── Dockerfile           # Container configuration
│   ├── app.py               # Flask application
│   ├── templates/           # HTML templates
│   │   └── index.html       # Main page template
│   └── requirements.txt     # Python dependencies
└── docker-compose.yml       # Multi-container configuration
```

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key

## Troubleshooting

- If audio recording doesn't work, ensure your browser has permission to access the microphone
- If the ML service fails to start, check your OpenAI API key is valid and has access to required models
- If containers fail to connect, ensure Docker Compose is running correctly
- If MongoDB connection issues occur, ensure the MongoDB container is running and accessible

## License

MIT License
