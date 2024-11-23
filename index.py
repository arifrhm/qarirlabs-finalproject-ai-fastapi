import yolov5
import cv2
import numpy as np
import os

# Load the model
model = yolov5.load("best.pt")

# Set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # Maximum number of detections per image

# Set image
img = "container-port-foreman-and-insurance-"
img += "claim-office-2022-08-01-04-26-48-utc.jpg"

# Perform inference
results = model(img, size=640)
results.print()

# Parse results
predictions = results.pred[0]
boxes = predictions[:, :4]  # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]

# Make a copy of the image array
img_copy = np.copy(results.ims[0])

# Get class names
class_names = model.names

# Draw detection bounding boxes and labels on the copied image
for box, score, category in zip(boxes, scores, categories):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Get class name and accuracy percentage
    class_name = class_names[int(category)]
    accuracy = round(float(score) * 100, 2)

    # Add label and accuracy percentage near the bounding box
    label = f"{class_name}: {accuracy}%"
    cv2.putText(
        img_copy, label, (
            x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

# Create the "results/" folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save the image with bounding boxes and
# labels to a file in the "results/" folder
output_path = "results/detections_with_labels.jpg"
cv2.imwrite(output_path, img_copy)

print(f"Image with bounding boxes and labels saved to: {output_path}")
