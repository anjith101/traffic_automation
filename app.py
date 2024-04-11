import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import serial
import time

# Load YOLO model
model = YOLO("best.pt")
# Define polygon vertices for the zone of interest
polygonVal = {
    'straightOne': {
        'valOne': np.array([[350, 246],[442, 250],[314, 706],[186, 678],[354, 230]]),
        'valTwo': np.array([[318, 690],[446, 222],[614, 226],[498, 710],[314, 694]])
    },
    'straightTwo': {
        'valOne': np.array([[350, 246],[442, 250],[314, 706],[186, 678],[354, 230]]),
        'valTwo': np.array([[318, 690],[446, 222],[614, 226],[498, 710],[314, 694]])
    },
    'sideOne': {
        'valOne': np.array([[350, 246],[442, 250],[314, 706],[186, 678],[354, 230]]),
        'valTwo': np.array([[318, 690],[446, 222],[614, 226],[498, 710],[314, 694]])
    },
    'sideTwo': {
        'valOne': np.array([[350, 246],[442, 250],[314, 706],[186, 678],[354, 230]]),
        'valTwo': np.array([[318, 690],[446, 222],[614, 226],[498, 710],[314, 694]])
    }
}
signals = {
    'straight':{
        'one': {
            'status': False,
            'angle': 'angle 90'
        },
        'two': {
            'status': False,
            'angle': 'angle 180'
        }
    },
    'side':{
        'one': {
            'status': False,
            'angle': 'angle -90'
        },
        'two': {
            'status': False,
            'angle': 'angle -180'
        }
    },
}
class Detection:
    def __init__(self, detected: bool, count: int):
        self.detected = detected
        self.count = count

class Result:
    def __init__(self, frame: np.ndarray, car: Detection, ambulance: Detection, total: Detection):
        self.frame = frame
        self.total = total
        self.car = car
        self.ambulance = ambulance
def reset():
    return {
        'straight':{
            'one': {
                'status': False,
                'angle': 'angle 90'
            },
            'two': {
                'status': False,
                'angle': 'angle 180'
            }
        },
        'side':{
            'one': {
                'status': False,
                'angle': 'angle -90'
            },
            'two': {
                'status': False,
                'angle': 'angle -180'
            }
        },
    }
def labelImage(result: Result):
    image = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    color = (255, 0, 0)
    text = ('Total Vehicle: ' + str(result.total.count) + '\n' + 'Car: ' + str(result.car.count) + '\n' + 'Ambulance: ' + str(result.ambulance.count))
    y0, dy = 50, 40
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        image = cv2.putText(image, line, (50, y ), font, fontScale, color, thickness, cv2.LINE_AA)
    return image

def sendCommand(message):
    try:
        # Open the serial port
        ser = serial.Serial('COM8', 9600)
        # Write the message to the serial port
        ser.write(message.encode())
        # Close the serial port
        ser.close()
        print("Message sent successfully!")
    except Exception as e:
        print("Error:", e)
def readMessage():
    # Open the serial port
    ser = serial.Serial('COM8', 9600)
    while True:
        if ser.in_waiting > 0:
            message = ser.read_until(b'\n').decode().strip()
            return message
# Function to detect objects in the defined zone
def process_frame(frame, polygon: np.ndarray) -> Result:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_np = np.array(frame_rgb)
    # Get height, width, and number of channels from the input image
    height, width, px = frame_np.shape
    # Define the zone of interest in the image using a polygon
    carZone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(width, height))
    ambulanceZone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(width, height))
    totalZone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(width, height))
    zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(width, height))
    # Run object detection using the YOLO model
    results = model(frame_np, imgsz=1280)[0]
    # Convert YOLO results to Detections object
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[(detections.confidence >= 0.7)]
    # Filter detections for specific class IDs
    carDetections = detections[(detections.class_id == 1)] # car
    ambulanceDetections = detections[(detections.class_id == 0)] #ambulance
    totalDetections = detections[(detections.class_id == 1) | (detections.class_id == 0)] # total
    # Apply zone trigger to get detections within the defined zone
    carZone.trigger(detections=carDetections)
    ambulanceZone.trigger(detections=ambulanceDetections)
    totalZone.trigger(detections=totalDetections)
    total: Detection = Detection(totalZone.current_count > 0, totalZone.current_count)
    car: Detection = Detection(carZone.current_count > 0, carZone.current_count)
    ambulance: Detection = Detection(ambulanceZone.current_count > 0, ambulanceZone.current_count)
    # Initialize annotators for bounding boxes and the defined zone
    triangle_annotator = sv.TriangleAnnotator(base=20,height=20)
    zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.WHITE, thickness=6, text_thickness=6, text_scale=4)
    # Annotate the frame with bounding boxes and the defined zone
    frame = triangle_annotator.annotate(scene=frame, detections=totalDetections)
    frame = zone_annotator.annotate(scene=frame)

    result: Result = Result(frame = frame, car = car, ambulance = ambulance, total = total)
    return result

# Load an image from file
cap = cv2.VideoCapture(1)
# Set the resolution to 1280x720 (HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Process the image using the defined function
while True:
    if(signals['straight']['one']['status'] == False and signals['side']['two']['status'] == False):
        signals['straight']['one']['status'] = True
        # Capture frame-by-frame
        ret, frame = cap.read()
        result = process_frame(frame, polygonVal['straightOne']['valOne'])
        # result1 = process_frame(frame, polygonVal['straightOne']['valTwo'])
        labeledImage = labelImage(result)
        # Display the resulting frame
        cv2.imshow('Live Video', labeledImage)
    elif(signals['straight']['one']['status'] == True and signals['side']['two']['status'] == True):
        sendCommand(signals['straight']['one']['angle'])
        message = readMessage()
        if(message == 'done'):
            signals = reset()
            signals['straight']['one']['status'] = True
            # Capture frame-by-frame
            ret, frame = cap.read()
            result = process_frame(frame, polygonVal['straightOne']['valOne'])
            # result1 = process_frame(frame, polygonVal['straightOne']['valTwo'])
            labeledImage = labelImage(result)
            # Display the resulting frame
            cv2.imshow('Live Video', labeledImage)
    elif(signals['straight']['one']['status'] == True and signals['straight']['two']['status'] == False):
        sendCommand(signals['straight']['two']['angle'])
        message = readMessage()
        if(message == 'done'):
            signals['straight']['two']['status'] = True
            # Capture frame-by-frame
            ret, frame = cap.read()
            result = process_frame(frame, polygonVal['straightOne']['valOne'])
            # result1 = process_frame(frame, polygonVal['straightOne']['valTwo'])
            labeledImage = labelImage(result)
            # Display the resulting frame
            cv2.imshow('Live Video', labeledImage)
    elif(signals['straight']['two']['status'] == True and signals['side']['one']['status'] == False):
        sendCommand(signals['side']['one']['angle'])
        message = readMessage()
        if(message == 'done'):
            signals['side']['one']['status'] = True
            # Capture frame-by-frame
            ret, frame = cap.read()
            result = process_frame(frame, polygonVal['straightOne']['valOne'])
            # result1 = process_frame(frame, polygonVal['straightOne']['valTwo'])
            labeledImage = labelImage(result)
            # Display the resulting frame
            cv2.imshow('Live Video', labeledImage)
    elif(signals['side']['one']['status'] == True and signals['side']['two']['status'] == False):
        sendCommand(signals['side']['two']['angle'])
        message = readMessage()
        if(message == 'done'):
            signals['side']['two']['status'] = True
            # Capture frame-by-frame
            ret, frame = cap.read()
            result = process_frame(frame, polygonVal['straightOne']['valOne'])
            # result1 = process_frame(frame, polygonVal['straightOne']['valTwo'])
            labeledImage = labelImage(result)
            # Display the resulting frame
            cv2.imshow('Live Video', labeledImage)
     # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
