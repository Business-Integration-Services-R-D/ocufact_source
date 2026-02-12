# Sensitive Data Detection System

A comprehensive FastAPI + Flask application for OCR and sensitive data detection with real-time UI interactions.

## Features

- **OCR Processing**: PaddleOCR v5 for text extraction from images and PDFs
- **Sensitive Data Detection**: 
  - Binary classification (sensitive/non-sensitive)
  - Multi-label classification (NAME, EMAIL, MOBILE, ID_NO, etc.)
- **Smart Regex Generation**: Automatic regex pattern synthesis from sample data
- **Interactive Document Viewer**: 
  - Zoom and pan functionality
  - Overlay highlighting with hover tooltips
  - Click for detailed word information
- **Masking Capabilities**: Apply colored masks to sensitive areas
- **Real-time Updates**: Socket.IO for live processing status
- **Modern UI**: Responsive single-page application with Bootstrap 5

## Architecture

```
sensitive-detector/
├── backend/           # FastAPI service
│   ├── app/
│   │   ├── api/       # API endpoints
│   │   ├── services/  # Business logic services
│   │   ├── schemas.py # Pydantic models
│   │   └── utils.py   # Helper functions
│   └── Dockerfile
├── frontend/          # Flask UI service
│   ├── templates/     # HTML templates
│   ├── static/        # CSS, JS, assets
│   └── server.py      # Flask application
├── models/            # ML models directory
└── docker-compose.yml
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (recommended)
- At least 4GB RAM for model loading
- GPU support optional but recommended

## Quick Start with Docker

1. **Clone and prepare the project**:
   ```bash
   cd sensitive-detector
   ```

2. **Ensure models are available**:
   - Place your `binary_classification_model/` in the `sensitive-detector/` directory
   - Place your `label_detection_model/` in the `sensitive-detector/` directory
   - Place `smart_regex_synthesizer2.py` in the `sensitive-detector/` directory

3. **Start the services**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - Frontend UI: http://localhost:5000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Manual Installation

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   ```bash
   export PYTHONPATH=$PWD
   export HOST=0.0.0.0
   export PORT=8000
   ```

5. **Start the backend**:
   ```bash
   python -m app.main
   ```

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   ```bash
   export FLASK_APP=server.py
   export FLASK_ENV=development
   export BACKEND_URL=http://localhost:8000
   ```

5. **Start the frontend**:
   ```bash
   python server.py
   ```

## API Endpoints

### OCR Endpoints
- `POST /api/v1/ocr/image` - Extract text from image
- `POST /api/v1/ocr/pdf` - Extract text from PDF

### Detection Endpoints
- `POST /api/v1/detect/binary/image` - Binary sensitive data detection
- `POST /api/v1/detect/binary/pdf` - Binary detection for PDF
- `POST /api/v1/detect/labels/image` - Label classification
- `POST /api/v1/detect/labels/pdf` - Label classification for PDF

### Regex Endpoints
- `POST /api/v1/regex/generate` - Generate regex from phrases
- `GET /api/v1/regex/stored` - Get stored regex patterns
- `POST /api/v1/regex/match/image` - Match regex against image
- `POST /api/v1/regex/match/pdf` - Match regex against PDF

### Masking Endpoints
- `POST /api/v1/mask/image` - Apply masks to image
- `POST /api/v1/mask/pdf` - Apply masks to PDF

### Utility Endpoints
- `POST /api/v1/match/coords/image` - Match coordinates with OCR text

## Usage Guide

### 1. Upload Document
- Drag and drop an image or PDF file
- Supported formats: JPG, PNG, PDF
- Maximum file size: 50MB

### 2. Run Detection
- **OCR**: Automatically runs on upload
- **Binary Detection**: Click "Run Binary Detection" 
- **Label Detection**: Click "Run Label Detection"

### 3. Generate Regex Patterns
- Enter at least 8 sample phrases
- Click "Generate Regex" 
- Test patterns against documents

### 4. Apply Masking
- Run detection first
- Click "Preview Mask" to see overlay
- Click "Download Masked" to get processed file

### 5. Interactive Features
- **Zoom**: Mouse wheel or zoom buttons
- **Pan**: Click and drag
- **Hover**: See word details in tooltip
- **Click**: Open detailed word information modal

## Configuration

### Environment Variables

**Backend**:
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `PYTHONPATH`: Python path for imports

**Frontend**:
- `FLASK_PORT`: Flask port (default: 5000)
- `BACKEND_URL`: Backend API URL
- `FLASK_ENV`: Environment (development/production)

### Model Configuration

The system expects models in this structure:
```
sensitive-detector/
├── binary_classification_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── label_detection_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── smart_regex_synthesizer2.py
└── ...
```

## Color Coding

### Binary Detection
- 🔴 **Red**: Sensitive data
- 🟢 **Green**: Non-sensitive data

### Label Detection
- 🔴 **Red**: Names
- 🔵 **Teal**: Email addresses  
- 🔵 **Blue**: Phone numbers
- 🟡 **Yellow**: ID numbers
- 🟣 **Purple**: Addresses
- 🔵 **Light Blue**: Credit cards
- 🟠 **Orange**: Social security numbers

## Troubleshooting

### Common Issues

1. **Models not loading**:
   - Ensure model paths are correct
   - Check file permissions
   - Verify model format compatibility

2. **OCR errors**:
   - Install system dependencies for OpenCV
   - Check image format and quality
   - Ensure sufficient memory

3. **Connection issues**:
   - Verify backend is running on correct port
   - Check CORS settings
   - Ensure Socket.IO connectivity

### Docker Issues

1. **Build failures**:
   ```bash
   docker-compose build --no-cache
   ```

2. **Volume mounting**:
   ```bash
   # Check volume paths in docker-compose.yml
   # Ensure models directory exists
   ```

3. **Port conflicts**:
   ```bash
   # Change ports in docker-compose.yml
   ports:
     - "8001:8000"  # Backend
     - "5001:5000"  # Frontend
   ```

## Performance Optimization

1. **GPU Support**: 
   - Install CUDA-enabled PyTorch
   - Update model service to use GPU

2. **Model Optimization**:
   - Use model quantization
   - Implement model caching
   - Batch processing for multiple documents

3. **Frontend Optimization**:
   - Enable gzip compression
   - Implement lazy loading for large documents
   - Use CDN for static assets

## Development

### Adding New Detection Models

1. Place model in `models/` directory
2. Update `ModelService` class
3. Add new endpoint in `endpoints.py`
4. Update frontend UI accordingly

### Custom Regex Patterns

The system uses `smart_regex_synthesizer2.py` for automatic pattern generation. To customize:

1. Modify the synthesizer class
2. Update `RegexService` configuration
3. Adjust confidence thresholds

## License

This project is for educational and research purposes. Ensure compliance with your organization's data privacy policies when processing sensitive information.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check Docker logs: `docker-compose logs`
