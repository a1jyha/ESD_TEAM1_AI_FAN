# YOLO11n(사람)과 YuNet(얼굴)의 ONNX 추론과 전·후처리. 두 모델이 같은 형식(박스 목록)으로 결과를 내도록 통일합니다.
from pathlib import Path
from time import perf_counter
import argparse
import sys

import cv2
import numpy as np


def check_model_file(model_path):
    path = Path(model_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")

    # Git LFS 포인터나 HTML 페이지를 모델로 저장한 경우 확인
    with path.open("rb") as file:
        header = file.read(256).lstrip()

    if header.startswith(b"version https://git-lfs"):
        raise RuntimeError(
            f"This is a Git LFS pointer, not the model: {path}"
        )

    if header.lower().startswith((b"<!doctype html", b"<html")):
        raise RuntimeError(
            f"This is an HTML page, not an ONNX model: {path}"
        )

    return str(path)


def letterbox(frame, input_size):
    """비율을 유지해 축소/확대하고 남는 영역을 채운다."""
    height, width = frame.shape[:2]

    scale = min(input_size / width, input_size / height)
    new_width = max(1, min(input_size, round(width * scale)))
    new_height = max(1, min(input_size, round(height * scale)))

    resized = cv2.resize(frame, (new_width, new_height))

    pad_x = (input_size - new_width) // 2
    pad_y = (input_size - new_height) // 2

    canvas = np.full(
        (input_size, input_size, 3),
        114,
        dtype=np.uint8,
    )

    canvas[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    # 반올림된 실제 크기로 복원 비율 계산
    scale_x = new_width / width
    scale_y = new_height / height

    return canvas, (scale_x, scale_y, pad_x, pad_y)


def restore_box(box, transform, frame_shape):
    """모델 입력 좌표의 xyxy 박스를 원본 영상 좌표로 복원한다."""
    scale_x, scale_y, pad_x, pad_y = transform
    height, width = frame_shape[:2]

    x1, y1, x2, y2 = map(float, box)

    if not np.isfinite([x1, y1, x2, y2]).all():
        return None

    x1 = (x1 - pad_x) / scale_x
    x2 = (x2 - pad_x) / scale_x
    y1 = (y1 - pad_y) / scale_y
    y2 = (y2 - pad_y) / scale_y

    x1 = int(round(np.clip(x1, 0, width)))
    x2 = int(round(np.clip(x2, 0, width)))
    y1 = int(round(np.clip(y1, 0, height)))
    y2 = int(round(np.clip(y2, 0, height)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


class YuNetDetector:
    def __init__(
        self,
        model_path,
        input_size=320,
        score_threshold=0.7,
        nms_threshold=0.3,
        top_k=5000,
    ):
        model_path = check_model_file(model_path)
        self.input_size = int(input_size)

        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError(
                f"FaceDetectorYN is unavailable in OpenCV {cv2.__version__}"
            )

        self.detector = cv2.FaceDetectorYN_create(
            model_path,
            "",
            (self.input_size, self.input_size),
            score_threshold,
            nms_threshold,
            top_k,
        )

    def detect(self, frame):
        image, transform = letterbox(frame, self.input_size)

        self.detector.setInputSize(
            (self.input_size, self.input_size)
        )
        _, faces = self.detector.detect(image)

        detections = []

        if faces is None:
            return detections

        for face in faces:
            x, y, width, height = map(float, face[:4])

            box = restore_box(
                (x, y, x + width, y + height),
                transform,
                frame.shape,
            )

            if box is None:
                continue

            detections.append({
                "bbox": box,
                "score": float(face[14]),
                "label": "face",
            })

        return detections


class YOLOONNXDetector:
    """
    지원 출력:
        [1, 84, N] 또는 [1, N, 84]

    COCO 80개 클래스, 별도 objectness 없음,
    NMS가 모델 내부에 포함되지 않은 YOLO11 검출 모델 기준.
    """

    def __init__(
        self,
        model_path,
        input_size=640,
        conf_threshold=0.35,
        nms_threshold=0.45,
        person_class_id=0,
    ):
        model_path = check_model_file(model_path)

        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.person_class_id = int(person_class_id)

        if not 0 <= self.person_class_id < 80:
            raise ValueError("person_class_id must be between 0 and 79.")

    def detect(self, frame):
        image, transform = letterbox(frame, self.input_size)

        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)

        outputs = self.net.forward(
            self.net.getUnconnectedOutLayersNames()
        )

        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected one YOLO output, got {len(outputs)}."
            )

        output = np.asarray(outputs[0])

        if output.ndim != 3 or output.shape[0] != 1:
            raise RuntimeError(
                f"Unsupported YOLO output shape: {output.shape}"
            )

        pred = output[0]

        if pred.shape[0] == 84:
            pred = pred.T
        elif pred.shape[1] != 84:
            raise RuntimeError(
                f"Expected [1, 84, N] or [1, N, 84], "
                f"got {output.shape}. "
                "Check class count and NMS export settings."
            )

        person_scores = pred[:, 4 + self.person_class_id]
        selected = pred[
            np.isfinite(pred).all(axis=1)
            & (person_scores >= self.conf_threshold)
        ]

        boxes = []
        scores = []

        for row in selected:
            cx, cy, width, height = map(float, row[:4])

            if width <= 0 or height <= 0:
                continue

            # NMS는 모델 입력 좌표에서 수행
            boxes.append([
                cx - width / 2,
                cy - height / 2,
                width,
                height,
            ])
            scores.append(float(row[4 + self.person_class_id]))

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.conf_threshold,
            self.nms_threshold,
        )

        detections = []

        if indices is None or len(indices) == 0:
            return detections

        for index in np.asarray(indices).reshape(-1):
            index = int(index)
            x, y, width, height = boxes[index]

            box = restore_box(
                (x, y, x + width, y + height),
                transform,
                frame.shape,
            )

            if box is None:
                continue

            detections.append({
                "bbox": box,
                "score": scores[index],
                "label": "person",
            })

        return detections


def create_detector(name, model_dir, input_size, model_path=None):
    model_dir = Path(model_dir)

    if name == "yunet":
        path = model_path or (
            model_dir / "face_detection_yunet_2026may.onnx"
        )
        return YuNetDetector(path, input_size=input_size)

    if name == "yolo":
        path = model_path or (model_dir / "yolo11n.onnx")
        return YOLOONNXDetector(path, input_size=input_size)

    raise ValueError("Model must be 'yunet' or 'yolo'.")


def draw_detections(frame, detections, model_name, detect_ms, fps):
    height, width = frame.shape[:2]

    # 画面中央の基準線
    cv2.line(
        frame,
        (width // 2, 0),
        (width // 2, height - 1),
        (255, 255, 0),
        1,
    )

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        cv2.rectangle(
            frame,
            (x1, y1),
            (min(x2, width - 1), min(y2, height - 1)),
            (0, 255, 0),
            2,
        )

        cv2.circle(
            frame,
            (round(xc), round(yc)),
            4,
            (0, 0, 255),
            -1,
        )

        label = (
            f"{detection['label']} "
            f"{detection['score']:.2f} "
            f"xc={xc:.0f}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(65, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

    lines = [
        f"{model_name} | Count: {len(detections)}",
        f"Detect: {detect_ms:.1f} ms | Loop FPS: {fps:.1f}",
    ]

    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (10, 22 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )


def main():
    script_dir = Path(__file__).resolve().parent
    project_dir = (
        script_dir.parent if script_dir.name == "src" else script_dir
    )

    parser = argparse.ArgumentParser(
        description="PiCamera + YuNet / YOLO ONNX detection"
    )
    parser.add_argument(
        "--model",
        choices=["yunet", "yolo"],
        default="yunet",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_dir / "models",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit ONNX model path",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Square model input size",
    )
    args = parser.parse_args()

    input_size = args.input_size
    if input_size is None:
        input_size = 320 if args.model == "yunet" else 192

    if input_size <= 0 or input_size % 32 != 0:
        parser.error("--input-size must be a positive multiple of 32.")

    print(f"Python: {sys.executable}", flush=True)
    print(f"OpenCV: {cv2.__version__}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Input: {input_size} x {input_size}", flush=True)

    # 카메라 초기화 전에 모델 실행 여부를 확인
    detector = create_detector(
        args.model,
        args.model_dir,
        input_size,
        args.model_path,
    )

    print("Checking model inference...", flush=True)
    detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    print("Model check OK.", flush=True)

    from picamera2 import Picamera2

    camera = None
    started = False
    window_name = "Smart Fan - Detection"

    try:
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={
                "size": (640, 480),
                "format": "RGB888",
            }
        )
        camera.configure(config)
        camera.start()
        started = True

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)

        print(
            "Camera started. Click the window and press q or Esc to quit.",
            flush=True,
        )

        fps = 0.0
        frame_count = 0
        fps_start = perf_counter()

        while True:
            # Picamera2의 RGB888 배열은 여기서 OpenCV BGR로 사용
            frame = camera.capture_array("main")

            detect_start = perf_counter()
            detections = detector.detect(frame)
            detect_ms = (perf_counter() - detect_start) * 1000

            draw_detections(
                frame,
                detections,
                args.model,
                detect_ms,
                fps,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_count += 1
            now = perf_counter()
            elapsed = now - fps_start

            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = now

    finally:
        try:
            if camera is not None:
                try:
                    if started:
                        camera.stop()
                finally:
                    camera.close()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(
            f"\nERROR ({type(error).__name__}): {error}",
            file=sys.stderr,
        )
        print(
            "Send the error above together with the printed OpenCV version.",
            file=sys.stderr,
        )
        sys.exit(1)
