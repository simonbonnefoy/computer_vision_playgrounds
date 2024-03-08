import cv2 # Import OpenCV Library
from ultralytics import YOLO # Import Ultralytics Package
import sys

def run_tracker(filename, model):
    """
    This function is designed to run a video file or webcam stream
    concurrently with the YOLOv8 model, utilizing threading.

    - filename: The path to the video file or the webcam/external
    camera source.
    - model: The file path to the YOLOv8 model.
    - file_index: An argument to specify the count of the
    file being processed.
    """

    video = cv2.VideoCapture(filename)  # Read the video file

    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("detection", 900, 900)
        cv2.imshow("detection", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()


if __name__ == "__main__":
    video_file = sys.argv[1]
    model = YOLO('yolov8n.pt')
    run_tracker(video_file, model)
