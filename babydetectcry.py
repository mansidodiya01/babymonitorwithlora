import cv2
from ultralytics import YOLO
import os
import time
import csv
import threading
import argparse
from mediapipe.tasks import python
from mediapipe.tasks.python.audio.core import audio_record
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python import audio

# Paths
base_dir = "images"
original_dir = os.path.join(base_dir, "original")
cropped_dir = os.path.join(base_dir, "cropped")
log_file = "baby_monitoring_log.csv"

# Ensure directories exist
os.makedirs(original_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)

# Initialize log file with headers if not exists
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Baby Crying Status", "Baby Detection Status", "Cropped Image"])

# Global variable for baby crying detection
baby_crying_status = "Baby is not crying"


def get_timestamp():
    """Generate a timestamp string for filenames."""
    return time.strftime("%Y-%m-%d_%H%M%S")


def detect_baby():
    """Detects baby in video stream and logs results."""
    global baby_crying_status

    # Load YOLO model
    model_path = "best_ncnn_model"
    yolo_model = YOLO(model_path, task="detect")

    # Open video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Unable to access the camera.")
        return

    interval_between_inference = 2

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Unable to read frame from camera.")
            break

        results = yolo_model(frame)
        timestamp = get_timestamp()

        # Save original frame
        original_image_filename = f"original_{timestamp}.jpg"
        original_image_path = os.path.join(original_dir, original_image_filename)
        resized_original = cv2.resize(frame, (96, 96))
        cv2.imwrite(original_image_path, resized_original)
        print(f"✅ Original image saved: {original_image_path}")

        # Process detection results
        best_box = None
        best_score = 0

        for result in results:
            boxes = result.boxes.xyxy.numpy()
            scores = result.boxes.conf.numpy()

            for i, box in enumerate(boxes):
                score = scores[i]
                if score > best_score:
                    best_box = box
                    best_score = score

        if best_box is not None:
            x_min, y_min, x_max, y_max = map(int, best_box)

            cropped_image = frame[y_min:y_max, x_min:x_max]

            # Save cropped image with timestamp
            cropped_image_filename = f"baby_{timestamp}.jpg"
            cropped_image_path = os.path.join(cropped_dir, cropped_image_filename)
            cv2.imwrite(cropped_image_path, cv2.resize(cropped_image, (96, 96)))
            print(f"✅ Cropped image saved: {cropped_image_path}")

            baby_detection_status = "Baby is detected"
        else:
            print("⚠️ No baby detected.")
            baby_detection_status = "Baby is not detected"
            cropped_image_filename = "None"

        # Log the entry
        log_entry = [timestamp, baby_crying_status, baby_detection_status, cropped_image_filename]

        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)

        elapsed_time = time.time() - start_time
        sleep_time = max(0, interval_between_inference - elapsed_time)
        time.sleep(sleep_time)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Exiting video stream...")
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_baby_cry(model: str, max_results: int, score_threshold: float):
    """Continuously detect baby crying sounds."""
    global baby_crying_status

    def save_result(result: audio.AudioClassifierResult, timestamp_ms: int):
        global baby_crying_status
        detected = False
        for category in result.classifications[0].categories:
            if ("baby cry" in category.category_name.lower() or
                "infant cry" in category.category_name.lower()) and category.score > score_threshold:
                print(f"🔊 Baby Cry Detected! Confidence: {category.score:.2f}")
                detected = True
        baby_crying_status = "Baby is crying" if detected else "Baby is not crying"

    base_options = python.BaseOptions(model_asset_path=model)
    options = audio.AudioClassifierOptions(
        base_options=base_options,
        running_mode=audio.RunningMode.AUDIO_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result,
    )
    classifier = audio.AudioClassifier.create_from_options(options)

    buffer_size, sample_rate, num_channels = 15600, 16000, 1
    record = audio_record.AudioRecord(num_channels, sample_rate, buffer_size)
    audio_data = containers.AudioData(buffer_size, containers.AudioDataFormat(num_channels, sample_rate))

    record.start_recording()

    reference_time = time.monotonic_ns() // 1_000_000
    buffer_duration = buffer_size / sample_rate

    print("🔊 Starting baby cry detection. Press Ctrl+C to stop.")
    while True:
        try:
            start_time = time.time()
            data = record.read(buffer_size)
            audio_data.load_from_array(data)

            timestamp = reference_time
            reference_time += int(buffer_duration * 1000)

            classifier.classify_async(audio_data, timestamp)

            elapsed_time = time.time() - start_time
            print(f"🔹 Audio Processing Time: {elapsed_time:.4f} sec")

            time.sleep(buffer_duration)
        except KeyboardInterrupt:
            print("Stopping baby cry detection...")
            break


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='Name of the audio classification model.', required=False, default='yamnet.tflite')
    parser.add_argument('--maxResults', help='Maximum number of results to show.', required=False, default=5)
    parser.add_argument('--scoreThreshold', help='The score threshold of classification results.', required=False, default=0.3)
    args = parser.parse_args()

    # Start both detection tasks in separate threads
    thread1 = threading.Thread(target=detect_baby)
    thread2 = threading.Thread(target=detect_baby_cry, args=(args.model, int(args.maxResults), float(args.scoreThreshold)))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()


if __name__ == '__main__':
    main()
