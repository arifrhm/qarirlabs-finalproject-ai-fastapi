from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import cv2
from pathlib import Path
from typing import List
import time  # Import time for generating unique timestamps
from service.model_interface import ModelInterface

# Initialize FastAPI
app = FastAPI()

# Initialize YOLOv5 model interface
MODEL_PATH = "best (1).pt"
model_interface = ModelInterface(model_path=MODEL_PATH)

# Folder to save results
RESULTS_FOLDER = Path("results")
RESULTS_FOLDER.mkdir(exist_ok=True)

# Supported file extensions
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Endpoint to upload an image for object detection.
    """
    # Validate file type
    if not file.filename.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
        raise HTTPException(
            status_code=400, detail="File must be an image (jpg, jpeg, png)"
        )

    # Save uploaded file
    input_image_path = RESULTS_FOLDER / file.filename
    try:
        with open(input_image_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save the file: {str(e)}"
        )

    # Read the image
    try:
        image = cv2.imread(str(input_image_path))
        if image is None:
            raise ValueError("Failed to read the image.")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error reading the image: {str(e)}"
        )

    # Perform object detection
    try:
        annotated_image = model_interface.annotate_image(image)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model inference failed: {str(e)}"
        )

    # Generate unique output filename using timestamp
    timestamp = int(time.time())  # Current timestamp as integer
    output_filename = f"detections_{timestamp}_{file.filename}"
    output_image_path = RESULTS_FOLDER / output_filename

    # Save annotated image
    try:
        cv2.imwrite(str(output_image_path), annotated_image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save the annotated image: {str(e)}"
        )

    return {
        "message": "Detection complete.",
        "output_path": str(output_image_path),
    }


@app.get("/download-image")
async def download_image(filename: str):
    """
    Endpoint to download the annotated image.
    """
    output_image_path = RESULTS_FOLDER / filename
    if not output_image_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(output_image_path))


@app.get("/list-results")
async def list_results() -> List[str]:
    """
    Endpoint to list all annotated images in the results folder.
    """
    try:
        files = [
            file.name for file in RESULTS_FOLDER.iterdir() if file.is_file()
            ]
        return files
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving files: {str(e)}"
        )
