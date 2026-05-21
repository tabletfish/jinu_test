import argparse

from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes


# Method 1: Direct InferencePipeline execution.
#
# When to use:
# - Best first choice for a small robo-car control loop if this runs well on Jetson Nano.
# - Camera capture, model inference, and display happen in this Python process.
# - Lowest software overhead because frames do not go through a local HTTP server.
#
# Tradeoffs:
# - Jetson Python/CUDA/ONNX dependencies must be installed correctly on the device.
# - GPU use depends on the local inference package/runtime being installed correctly.
# - If installation is unstable on JetPack 4.x, try the Docker server method instead.
#
# Run:
#   python3 roboflow_jetson_camera.py --camera 0
#
# Tune for Jetson Nano:
#   python3 roboflow_jetson_camera.py --camera 0 --confidence 0.5 --max-fps 5
#
# Stop:
#   Press Ctrl+C in the terminal.

ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"
MODEL_ID = "my-first-project-ml6kp/2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Roboflow model locally on a Jetson camera with InferencePipeline."
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
        help="Roboflow API key. Defaults to the ROBOFLOW_API_KEY constant in this file.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=10,
        help="Maximum camera frames per second to process.",
    )
    return parser.parse_args()


def camera_reference(value):
    try:
        return int(value)
    except ValueError:
        return value


def main():
    args = parse_args()

    if not args.api_key:
        raise SystemExit(
            "Missing Roboflow API key. Edit ROBOFLOW_API_KEY at the top of this file."
        )

    pipeline = InferencePipeline.init(
        model_id=args.model_id,
        video_reference=camera_reference(args.camera),
        on_prediction=render_boxes,
        api_key=args.api_key,
        confidence=args.confidence,
        max_fps=args.max_fps,
    )

    pipeline.start()
    pipeline.join()


if __name__ == "__main__":
    main()
