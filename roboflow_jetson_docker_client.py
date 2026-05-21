import argparse
import time

import cv2
from inference_sdk import InferenceHTTPClient

from roboflow_jetson_camera import MODEL_ID, ROBOFLOW_API_KEY


# Method 2: Docker Inference Server client.
#
# When to use:
# - Best first choice when Jetson Nano package installation is unstable.
# - Roboflow's Jetson Docker container owns the model runtime.
# - This Python file only reads camera frames and asks the local server for results.
#
# Important:
# - The inference server is still local when api_url is http://localhost:9001.
# - Frames are not being sent to a cloud server in this setup.
# - OpenCV is used only to read camera frames and draw boxes.
# - The actual model inference runs inside the Roboflow Docker server.
# - There is still local HTTP request/response overhead, so latency may be higher
#   than direct InferencePipeline execution.
#
# Jetson Nano / JetPack 4.6 server example from the Roboflow Jetson flow:
#   sudo docker run -d \
#     --name inference-server \
#     --runtime nvidia \
#     --read-only \
#     -p 9001:9001 \
#     --volume ~/.inference/cache:/tmp:rw \
#     --security-opt="no-new-privileges" \
#     --cap-drop="ALL" \
#     --cap-add="NET_BIND_SERVICE" \
#     roboflow/roboflow-inference-server-jetson-4.6.1:latest

#서버가 켜졌는지 확인:

#sudo docker ps

  #그 다음 클라이언트 실행:

#cd /home/dydlz/jinu_test
#python3 roboflow_jetson_docker_client.py --camera 0


#
# Run this client after the server is up:
#   python3 roboflow_jetson_docker_client.py --camera 0
#
# Headless robo-car run without display:
#   python3 roboflow_jetson_docker_client.py --camera 0 --no-display --interval 0.1
#
# Stop:
#   Press Ctrl+C in the terminal.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read a Jetson camera and run inference through a local Roboflow Docker server."
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:9001",
        help="Local Roboflow Inference Server URL.",
    )
    parser.add_argument(
        "--model-id",
        default=MODEL_ID,
        help="Roboflow model id.",
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera source. Use 0 for /dev/video0, a video path, or an RTSP URL.",
    )
    parser.add_argument(
        "--api-key",
        default=ROBOFLOW_API_KEY,
        help="Roboflow API key imported from roboflow_jetson_camera.py.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence to draw and use detections.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Seconds to wait between inference requests. Use 0.05-0.2 on slow hardware.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display window for headless robo-car runs.",
    )
    return parser.parse_args()


def camera_reference(value):
    try:
        return int(value)
    except ValueError:
        return value


def prediction_items(result):
    if isinstance(result, dict):
        return result.get("predictions", [])
    return []


def draw_predictions(frame, predictions, min_confidence):
    for pred in predictions:
        confidence = float(pred.get("confidence", 0.0))
        if confidence < min_confidence:
            continue

        x = float(pred.get("x", 0.0))
        y = float(pred.get("y", 0.0))
        width = float(pred.get("width", 0.0))
        height = float(pred.get("height", 0.0))
        label = str(pred.get("class", "object"))

        left = int(x - width / 2)
        top = int(y - height / 2)
        right = int(x + width / 2)
        bottom = int(y + height / 2)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def handle_robot_control(predictions):
    # Put motor/steering logic here after testing detection stability.
    # Example idea:
    # - choose the highest-confidence target
    # - compare its x position with the frame center
    # - steer left/right or stop based on class and distance estimate
    return predictions


def main():
    args = parse_args()

    if not args.api_key:
        raise SystemExit(
            "Missing Roboflow API key. Edit ROBOFLOW_API_KEY in roboflow_jetson_camera.py."
        )

    client = InferenceHTTPClient(api_url=args.api_url, api_key=args.api_key)
    cap = cv2.VideoCapture(camera_reference(args.camera))

    if not cap.isOpened():
        raise SystemExit(f"Could not open camera source: {args.camera}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read camera frame.")
                break

            result = client.infer(frame, model_id=args.model_id)
            predictions = prediction_items(result)
            handle_robot_control(predictions)

            if not args.no_display:
                draw_predictions(frame, predictions, args.confidence)
                cv2.imshow("Roboflow Docker Server Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.interval > 0:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
