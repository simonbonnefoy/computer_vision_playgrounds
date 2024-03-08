import cv2  # Import OpenCV Library
from sahi.utils.yolov8 import download_yolov8n_model
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Yolo on video.')
    parser.add_argument('--video',
                        dest='video_url',
                        required=True,
                        help='Video to parse through yolo algorithm')

    parser.add_argument("--sliding_win", dest="is_slide",
                        default=False,
                        action="store_true",
                        help="Use Sahi sliding windows")

    args = parser.parse_args()

    return args


def draw_boxes(result, frame):
    predictions = result.object_prediction_list

    for prediction in predictions:
        bbox = prediction.bbox
        white = (255, 255, 255)

        x1 = int(bbox.minx)
        y1 = int(bbox.miny)
        x2 = int(bbox.maxx)
        y2 = int(bbox.maxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), white, 2)
        cv2.putText(frame, f"Class: {prediction.category.name}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f"Prob: {prediction.score.value:.03f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, white, 2,
                    cv2.LINE_AA)


def run_tracker(filename, is_slide=False):
    """
    This function is designed to run a yolo algorithm over a video file
    or a webcam.
    - filename: The path to the video file or the webcam/external
    camera source.

    -is_slide: use sliding window from sahi algorithm
    """

    video = cv2.VideoCapture(filename)  # Read the video file

    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        #        results = get_sliced_prediction(frame, detection_model)
        if not is_slide:
            results = get_prediction(frame,
                                     detection_model,
                                     )
        else:
            results = get_sliced_prediction(frame,
                                     detection_model,
                                     )

        out_sliced = frame.copy()
        draw_boxes(results, out_sliced)
        cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("detection", 900, 900)
        cv2.imshow("detection", out_sliced)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()


if __name__ == "__main__":

    args = get_arguments()
    video_file = args.video_url

    # Download YOLOv8 model
    yolov8_model_path = "models/yolov8n.pt"
    download_yolov8n_model(yolov8_model_path)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=yolov8_model_path,
        confidence_threshold=0.3,
        device="cpu",  # or 'cuda:0'
    )
    run_tracker(video_file, is_slide = args.is_slide)
