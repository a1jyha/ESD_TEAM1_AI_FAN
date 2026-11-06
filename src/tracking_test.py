# Picamera2와 AI 검출 모델(YuNet/YOLO)을 활용하여 화면 내 지정된 대상을 실시간으로 추적하는 메인 제어 코드
from pathlib import Path
from time import perf_counter
import argparse
import sys

import cv2

from detectors import create_detector
from target_tracker import TargetTracker, box_center


def draw_box(frame, detection, color, thickness, text):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, detection["bbox"])

    cv2.rectangle(
        frame,
        (x1, y1),
        (min(x2, width - 1), min(y2, height - 1)),
        color,
        thickness,
    )

    cv2.putText(
        frame,
        text,
        (x1, max(70, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["yunet", "yolo"],
        default="yunet",
    )
    parser.add_argument("--input-size", type=int, default=None)
    args = parser.parse_args()

    # 현재 보유한 모델의 입력 조건
    input_size = args.input_size
    if input_size is None:
        input_size = 320 if args.model == "yunet" else 192

    if input_size <= 0 or input_size % 32 != 0:
        parser.error("--input-size must be a positive multiple of 32.")

    script_dir = Path(__file__).resolve().parent
    project_dir = (
        script_dir.parent if script_dir.name == "src" else script_dir
    )

    detector = create_detector(
        args.model,
        project_dir / "models",
        input_size=input_size,
    )

    tracker = TargetTracker(max_gap_seconds=1.5)

    print(f"Python: {sys.executable}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"Model: {args.model}, input: {input_size}")
    print("q / Esc: quit")
    print("r: reset target for another test")

    from picamera2 import Picamera2

    camera = None
    started = False
    window_name = "Smart Fan - Target Tracking"

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

        while True:
            frame = camera.capture_array("main")

            # 영상 획득 직후의 시간을 대상 연결에 사용
            frame_time = perf_counter()

            detect_start = perf_counter()
            detections = detector.detect(frame)
            detect_ms = (perf_counter() - detect_start) * 1000

            target_index, state = tracker.update(
                detections,
                frame.shape,
                frame_time,
            )

            height, width = frame.shape[:2]

            cv2.line(
                frame,
                (width // 2, 0),
                (width // 2, height - 1),
                (255, 255, 0),
                1,
            )

            # 선택되지 않은 대상은 회색으로 표시
            for index, detection in enumerate(detections):
                if index != target_index:
                    draw_box(
                        frame,
                        detection,
                        (150, 150, 150),
                        1,
                        detection["label"],
                    )

            # 현재 프레임에서 연결에 성공한 대상만 초록색으로 표시
            if target_index is not None:
                target = detections[target_index]

                draw_box(
                    frame,
                    target,
                    (0, 255, 0),
                    3,
                    f"TARGET {target['score']:.2f}",
                )

                xc, yc = box_center(target["bbox"])

                cv2.circle(
                    frame,
                    (round(xc), round(yc)),
                    5,
                    (0, 0, 255),
                    -1,
                )

                status = f"TRACKING | xc={xc:.1f}"
                status_color = (0, 255, 0)

                # 다음 단계에서 이 xc를 각도 계산에 전달하면 된다.

            elif state == "WAITING":
                status = "WAITING FOR FIRST TARGET"
                status_color = (0, 255, 255)

            else:
                status = "TARGET LOST"
                if tracker.locked_lost:
                    status += " | Press r to restart test"
                status_color = (0, 0, 255)

                # 다음 단계에서는 이 상태일 때 마지막 모터 각도 유지.
                # 이전 박스를 현재 검출 위치인 것처럼 사용하지 않는다.

            cv2.putText(
                frame,
                status,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                status_color,
                2,
            )

            cv2.putText(
                frame,
                f"{args.model} | Detect: {detect_ms:.1f} ms",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("r"):
                tracker.reset()
                print("Target reset.")

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
