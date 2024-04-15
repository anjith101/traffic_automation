# Import necessary libraries
import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import serial
import time
import threading

# Load YOLO model for object detection
model = YOLO("best_v1.pt")
timerEvent = threading.Event()

# Define polygon vertices for various zones of interest on the video frame
polygonVal = {
    'straightOne': {
        'valOne': np.array([[680, 706],[668, 94],[916, 82],[1156, 678],[679, 704]]),
        'valTwo': np.array([[318, 690],[446, 222],[614, 226],[498, 710],[314, 694]])
    },
    'straightTwo': {
        'valOne': np.array([[655, 692],[675, 44],[951, 28],[1151, 660],[659, 692]]),
        'valTwo': np.array([[318, 690],[446, 222],[614, 226],[498, 710],[314, 694]])
    },
    'sideOne': {
        'valOne': np.array([[711, 664],[687, 40],[979, 28],[1191, 632],[707, 660]]),
        'valTwo': np.array([[318, 690],[446, 222],[614, 226],[498, 710],[314, 694]])
    },
    'sideTwo': {
        'valOne': np.array([[623, 692],[643, 104],[887, 104],[1087, 696],[619, 692]]),
        'valTwo': np.array([[318, 690],[446, 222],[614, 226],[498, 710],[314, 694]])
    },
}

# Define control signals for directing the flow of the application logic
signals = {
    'straight': {'one': {'status': False, 'angle': 'angle 90'}, 'two': {'status': False, 'angle': 'angle 180'}},
    'side': {'one': {'status': False, 'angle': 'angle -90'}, 'two': {'status': False, 'angle': 'angle -180'}}
}

# Classes for storing detection results
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


#Function to set timer
def setTimer(seconds):
    timerEvent.set()
    timer = threading.Timer(seconds, callBack())
    timer.start()
    return

def callBack():
    timerEvent.clear()
    return

# Function to reset control signals
def reset():
    return {
        'straight': {'one': {'status': False, 'angle': 'angle 90'}, 'two': {'status': False, 'angle': 'angle 180'}},
        'side': {'one': {'status': False, 'angle': 'angle -90'}, 'two': {'status': False, 'angle': 'angle -180'}}
    }

# Function to label an image with detection results
def labelImage(result: Result):
    image = result.frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    thickness = 2
    color = (0, 0, 225)
    text = ('Total Vehicle: ' + str(result.total.count) + '\n' + 'Car: ' + str(result.car.count) + '\n' + 'Ambulance: ' + str(result.ambulance.count))
    y0, dy = 50, 40
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        image = cv2.putText(image, line, (50, y), font, fontScale, color, thickness, cv2.LINE_AA)
    return image

# Functions for serial communication with hardware (e.g., Arduino for controlling traffic signals)
def sendCommand(message):
    try:
        ser = serial.Serial('COM8', 9600)
        ser.write(message.encode())
        ser.close()
        print("Message sent successfully!")
        return
    except Exception as e:
        print("Error:", e)
        return

def readMessage():
    ser = serial.Serial('COM8', 9600)
    while True:
        if ser.in_waiting > 0:
            message = ser.read_until(b'\n').decode().strip()
            return message

def calculateSeconds(result1: Result, result2: Result):
    sendCommand('sensor')
    sensor = int(readMessage())
    multiplier = 3
    if(sensor > 150): multiplier = 5
    if(result1.total.count >= 6 or result2.total.count >= 6): return 6 * multiplier
    else:
        largest = max([result1.total.count, result2.total.count])
        return largest * multiplier

def calculateAmbulanceSeconds(result: Result):
    sendCommand('sensor')
    sensor = int(readMessage())
    multiplier = 3
    if(sensor > 150): multiplier = 5
    if(result.total.count >= 6): return 6 * multiplier
    else: result.total.count * multiplier

# Function to process each frame for object detection and annotation
def process_frame(frame, polygon: np.ndarray) -> Result:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_np = np.array(frame_rgb)
    # Get height, width, and number of channels from the input image
    height, width, px = frame_np.shape
    # Define the zone of interest in the image using a polygon
    carZone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(width, height))
    ambulanceZone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(width, height))
    totalZone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(width, height))
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
    zone_annotator = sv.PolygonZoneAnnotator(zone=totalZone, color=sv.Color.WHITE, thickness=6, text_thickness=6, text_scale=4)
    # Annotate the frame with bounding boxes and the defined zone
    frame = triangle_annotator.annotate(scene=frame, detections=totalDetections)
    frame = zone_annotator.annotate(scene=frame)

    result: Result = Result(frame = frame, car = car, ambulance = ambulance, total = total)
    return result

def main():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    result1: Result
    result2: Result
    seconds: int
    state = False
    while True:
        if(not state): 
            ret, frame = cap.read()
            cv2.imshow('Live Video',frame)
        if(state):
            if(signals['straight']['one']['status'] == False and signals['side']['two']['status'] == False):
                signals['straight']['one']['status'] = True
                ret, frame = cap.read()
                time.sleep(.2)
                result1 = process_frame(frame, polygonVal['straightOne']['valOne'])
                labeledImage = labelImage(result1)
                if(result1.ambulance.count > 0):
                    seconds = calculateAmbulanceSeconds(result1)
                    sendCommand('straight')
                    setTimer(seconds)
                cv2.imshow('Live Video', labeledImage)
            elif(signals['straight']['one']['status'] == True and signals['side']['two']['status'] == True):
                sendCommand(signals['straight']['one']['angle'])
                message = readMessage()
                if(message == 'done'):
                    time.sleep(.2)
                    signals = reset()
                    signals['straight']['one']['status'] = True
                    ret, frame = cap.read()
                    result1 = process_frame(frame, polygonVal['straightOne']['valOne'])
                    labeledImage = labelImage(result1)
                    if(result1.ambulance.count > 0):
                        seconds = calculateAmbulanceSeconds(result1)
                        sendCommand('straight')
                        setTimer(seconds)
                    cv2.imshow('Live Video', labeledImage)
            elif(signals['straight']['one']['status'] == True and signals['straight']['two']['status'] == False):
                sendCommand(signals['straight']['two']['angle'])
                message = readMessage()
                if(message == 'done'):
                    time.sleep(.2)
                    signals['straight']['two']['status'] = True
                    ret, frame = cap.read()
                    result2 = process_frame(frame, polygonVal['straightTwo']['valOne'])
                    labeledImage = labelImage(result2)
                    if(result2.ambulance.count > 0):
                        seconds = calculateAmbulanceSeconds(result2)
                        sendCommand('straight')  
                        setTimer(seconds)
                    else:
                        timerEvent.wait()
                        seconds = calculateSeconds(result1, result2)
                        sendCommand('straight')
                        setTimer(seconds)
                    cv2.imshow('Live Video', labeledImage)
            elif(signals['straight']['two']['status'] == True and signals['side']['one']['status'] == False):
                sendCommand(signals['side']['one']['angle'])
                message = readMessage()
                if(message == 'done'):
                    time.sleep(.2)
                    signals['side']['one']['status'] = True
                    ret, frame = cap.read()
                    result1 = process_frame(frame, polygonVal['sideOne']['valOne'])
                    labeledImage = labelImage(result1)
                    if(result1.ambulance.count > 0):
                        seconds = calculateAmbulanceSeconds(result1)
                        sendCommand('side')
                        setTimer(seconds)
                    cv2.imshow('Live Video', labeledImage)
            elif(signals['side']['one']['status'] == True and signals['side']['two']['status'] == False):
                sendCommand(signals['side']['two']['angle'])
                message = readMessage()
                if(message == 'done'):
                    time.sleep(.2)
                    signals['side']['two']['status'] = True
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    result2 = process_frame(frame, polygonVal['sideTwo']['valOne'])
                    labeledImage = labelImage(result2)
                    if(result2.ambulance.count > 0):
                        seconds = calculateAmbulanceSeconds(result2)
                        sendCommand('side')
                    else:
                        timerEvent.wait()
                        seconds = calculateSeconds(result1, result2)
                        sendCommand('side')
                        setTimer(seconds)
                    cv2.imshow('Live Video', labeledImage)


        # Check for key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            break
        # Check for key 
        if cv2.waitKey(1) & 0xFF == ord('s'):
            state = not state

if __name__ == "__main__":
    main()
