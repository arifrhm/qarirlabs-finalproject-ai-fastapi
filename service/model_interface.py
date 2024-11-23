import yolov5
import cv2
import numpy as np
# import os
from typing import Tuple


class ModelInterface:
    def __init__(self, model_path: str) -> None:
        # Load the YOLOv5 model
        self.model = yolov5.load(model_path)

        # Set model parameters
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 1000  # Maximum number of detections per image

    def predict(self, image: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Perform object detection on the input
        image and return bounding boxes, scores, and categories.
        """
        # Perform inference with the model
        results = self.model(image, size=640)

        # Print results to console
        results.print()

        # Parse detections
        detections = results.pred[0]
        boxes = detections[:, :4]  # Bounding boxes: [x1, y1, x2, y2]
        scores = detections[:, 4]  # Confidence scores
        categories = detections[:, 5]  # Class IDs

        return boxes, scores, categories

    def annotate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Annotate the input image with bounding
        boxes and labels based on predictions.
        """
        # Get predictions
        boxes, scores, categories = self.predict(image)

        # Create a copy of the image to draw on
        img_copy = np.copy(image)

        # Iterate through detections and draw bounding boxes and labels
        for box, score, category in zip(boxes, scores, categories):
            x1, y1, x2, y2 = map(int, box)
            class_name = self.model.names[int(category)]  # Get class name
            # Calculate confidence as percentage
            accuracy = round(float(score) * 100, 2)
            label = f"{class_name}: {accuracy}%"  # Label text

            # Draw the bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw the label
            cv2.putText(
                img_copy, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

        return img_copy

    # def save_results(
    #         self, image: np.ndarray, save_path: str = "results"
    # ) -> None:
    #     """
    #     Save the annotated image to the specified directory.
    #     """
    #     # Annotate the image
    #     annotated_image = self.annotate_image(image)

    #     # Ensure the results directory exists
    #     os.makedirs(save_path, exist_ok=True)

    #     # Save the annotated image to the results directory
    #     output_path = os.path.join(save_path, "annotated_image.jpg")
    #     cv2.imwrite(output_path, annotated_image)

    # print(f"Image with bounding boxes and labels saved to: {output_path}")


# Example usage:
# model_interface = ModelInterface("keremberke/yolov5n-construction-safety")
# image = cv2.imread("path/to/image.jpg")
# model_interface.save_results(image)
