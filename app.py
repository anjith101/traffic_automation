import numpy as np
import supervision as sv
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define polygon vertices for the zone of interest
polygon = np.array([[46, 510], [430, 530], [434, 266], [214, 278], [42, 510]])

# Function to detect objects in the defined zone
def process_frame(frame: np.ndarray) -> np.ndarray:
    # Get height, width, and number of channels from the input image
    height, width, px = frame.shape

    # Print image dimensions
    print('width: ' + str(width))
    print('height: ' + str(height))

    # Define the zone of interest in the image using a polygon
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(width, height))

    # Run object detection using the YOLO model
    results = model(frame, imgsz=1280)[0]
    
    # Convert YOLO results to Detections object
    detections = sv.Detections.from_ultralytics(results)
    
    # Filter detections for specific class IDs (2, 5, and 7 in this case)
    detections = detections[(detections.class_id == 2) | (detections.class_id == 5) | (detections.class_id == 7)]
    
    # Apply zone trigger to get detections within the defined zone
    zone.trigger(detections=detections)

    # Initialize annotators for bounding boxes and the defined zone
    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=6, text_thickness=6, text_scale=4)
    
    # Generate labels for detected objects
    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _, _ in detections]

    # Annotate the frame with bounding boxes and the defined zone
    # frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    # frame = zone_annotator.annotate(scene=frame)

    #save the image with just count
    frame = box_annotator.annotate(scene=frame, detections=(), labels=[])
    frame = zone_annotator.annotate(scene=frame)

    return frame

# Load an image from file
image = Image.open("./traffic.jpg")
image = np.array(image)

# Process the image using the defined function
image = process_frame(image)

# Convert the resulting array back to an image and save it
image = Image.fromarray(image)
image.save("./traffic-result.jpg")
