import torch
import cv2
import numpy as np

def detect_objects_webcam(output_path='output.mp4'):
    """Perform object detection using the webcam and save the output."""
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Constants for distance calculation
    KNOWN_WIDTH = 0.5  # Known width of the object in meters (e.g., average width of a person)
    FOCAL_LENGTH = 800  # Focal length in pixels (you might need to calibrate this for your camera)

    cap = cv2.VideoCapture(4)  # Open external webcam, change the index if needed
    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Perform inference
        results = model(frame)
        result_frame = np.squeeze(results.render())

        # Extract and print detected objects and distances
        detected_objects = results.pandas().xyxy[0]  # Extract information in a pandas DataFrame
        for index, row in detected_objects.iterrows():
            # Calculate the width of the detected object in pixels
            width_in_image = row['xmax'] - row['xmin']
            # Calculate the distance to the object
            distance = (KNOWN_WIDTH * FOCAL_LENGTH) / width_in_image
            print(f"Detected {row['name']} at ({row['xmin']}, {row['ymin']}, {row['xmax']}, {row['ymax']}) with distance {distance:.2f} meters")
            print("-------------------------------------------------------------------------------------------------------------------")
            # Check if the detected object is a traffic signal
            if row['name'] == 'traffic light':  # Adjust class name as per the model's labels
                print(f"Traffic signal detected at distance {distance:.2f} meters")

        # Display frame
        cv2.imshow('YOLOv5 Detection', result_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Write frame to output video
        out.write(result_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved as {output_path}")

if __name__ == "__main__":
    detect_objects_webcam()
