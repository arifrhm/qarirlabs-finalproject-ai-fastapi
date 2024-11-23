# Object Detection API

## Overview

This project provides a high-performance RESTful API for object detection using YOLOv5, built with FastAPI. The API allows clients to upload images and receive detailed detection results, including annotated images with bounding boxes, confidence scores, and class labels.

## Features

- 🖼️ Image upload and object detection
- 🎯 Precise object detection using YOLOv5 model
- 📦 Annotated images with bounding boxes and labels
- ⚙️ Configurable model parameters
- 🚀 High-performance FastAPI implementation

## Requirements

### System Requirements
- Python 3.7+
- pip

### Dependencies
- FastAPI
- Uvicorn
- NumPy
- OpenCV-Python
- Internal YOLOv5 model library
- Organization-specific modules

## Installation

### 1. Clone the Repository
```bash
git clone https://your_repository_url.git
cd your_repository
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
# Or on Windows
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Internal Packages
Follow organizational guidelines for installing internal packages.

## Model Preparation

1. Place your trained YOLOv5 model file (e.g., `best.pt`) in the appropriate directory.
2. Update the model path in the code as necessary.

## Running the Application

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- `main:app` refers to the app object in `main.py`
- `--reload` enables auto-reload during development

Access API documentation at: `http://localhost:8000/docs`

## API Endpoints

### 1. Upload Image (`/upload-image`)
- **Method**: POST
- **Input**: Image file (`.jpg`, `.jpeg`, `.png`)
- **Response**: 
  - 200 OK: Detection results with output image path
  - 400 Bad Request: Invalid file

**Example Response**:
```json
{
  "message": "Detection complete.",
  "output_path": "results/detections_example.jpg"
}
```

### 2. Download Image (`/download-image`)
- **Method**: GET
- **Input**: Filename query parameter
- **Response**: 
  - 200 OK: Annotated image file
  - 404 Not Found: File does not exist

## Usage Examples

### Upload Image with curl
```bash
curl -X POST "http://localhost:8000/upload-image" \
  -F "file=@path_to_your_image.jpg"
```

### Download Annotated Image
```bash
curl -X GET "http://localhost:8000/download-image" \
  -G -d "filename=detections_your_image.jpg" \
  --output annotated_image.jpg
```

## Project Structure
```
├── main.py                 # Main application file
├── service/
│   ├── __init__.py
│   └── model_interface.py  # Model interaction logic
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
└── results/                # Annotated images directory
```

## Notes & Best Practices

- Follow organizational coding standards
- Validate uploaded image files
- Configure model parameters carefully
- Manage internal module dependencies

## Contributing

1. Follow organizational coding guidelines
2. Perform thorough testing
3. Update documentation
4. Adhere to code style standards

## License

Proprietary and confidential. Unauthorized use is prohibited.

## Contact

For issues or contributions, contact project maintainers through organizational channels.