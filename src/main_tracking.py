
import cv2
import time
import argparse
import numpy as np

from picamera2 import Picamera2

from detectors import create_detector
from target_tracker import TargetTracker
from angle_controller import AngleController
from servo_controller import ServoController


def parse_args():
    parser = argparse.ArgumentParser(
        description="KCF-free face tracking fan controller"
    )

    # Camera
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    # Detector
    parser.add_argument("--model", type=str, default="yunet")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--input-size", type=int, default=320)

    # Target tracker
    parser.add_argument("--max-gap", type=float, default=1.5)

    # Angle controller
    parser.add_argument("--hfov", type=float, default=62.0)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--direction", type=int, default=1)

    parser.add_argument("--center-angle", type=float, default=90.0)
    parser.add_argument("--min-angle", type=float, default=60.0)
    parser.add_argument("--max-angle", type=float, default=120.0)

    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--deadband", type=float, default=1.0)
    parser.add_argument("--max-speed", type=float, default=30.0)

    # Servo
    parser.add_argument("--enable-servo", action="store_true")
    parser.add_argument("--pin", type=int, default=18)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)

    return parser.parse_args()


def draw_detection(frame, detections):
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        score = det.get("score", 0.0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (180, 180, 180),
            1
        )

        cv2.putText(
            frame,
            f"{score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )


def main():
    args = parse_args()

    camera = None
    servo = None

    try:
        # --------------------------------
        # 1. Detector 생성
        # --------------------------------
        detector = create_detector(
            name=args.model,
            model_dir=args.model_dir,
            input_size=args.input_size,
            model_path=args.model_path
        )

        # --------------------------------
        # 2. TargetTracker 생성
        # --------------------------------
        tracker = TargetTracker(
            max_gap_seconds=args.max_gap
        )

        # --------------------------------
        # 3. Camera 시작
        # --------------------------------
        camera = Picamera2()

        config = camera.create_preview_configuration(
            main={
                "size": (args.width, args.height),
                "format": "RGB888"
            }
        )

        camera.configure(config)
        camera.start()

        time.sleep(1.0)

        # 첫 프레임으로 실제 영상 크기 확인
        frame = camera.capture_array("main")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_height, frame_width = frame.shape[:2]

        # --------------------------------
        # 4. AngleController 생성
        # --------------------------------
        angle_controller = AngleController(
            frame_width=frame_width,
            fx=args.fx,
            cx=args.cx,
            hfov_deg=args.hfov,
            center_angle=args.center_angle,
            direction=args.direction,
            min_angle=args.min_angle,
            max_angle=args.max_angle,
            smoothing_tau=args.tau,
            deadband_deg=args.deadband,
            max_speed_deg_s=args.max_speed
        )

        # --------------------------------
        # 5. ServoController 생성
        # --------------------------------
        if args.enable_servo:
            servo = ServoController(
                pin=args.pin,
                host=args.host,
                port=args.port
            )

            servo.center()

        print("Tracking started")
        print("Press Q to quit")

        prev_time = time.time()
        fps = 0.0

        last_servo_angle = None

        # --------------------------------
        # 6. Main loop
        # --------------------------------
        while True:

            # Camera frame
            frame = camera.capture_array("main")
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            timestamp = time.time()

            # FPS 계산
            dt = timestamp - prev_time
            prev_time = timestamp

            if dt > 0:
                current_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * current_fps if fps else current_fps

            # --------------------------------
            # 6-1. YuNet 얼굴 검출
            # --------------------------------
            detections = detector.detect(frame)

            # --------------------------------
            # 6-2. 대상 추적 및 상태 업데이트
            # --------------------------------
            target_index, state = tracker.update(
                detections,
                frame.shape,
                timestamp
            )

            # 기본값
            target_bbox = None

            # --------------------------------
            # 6-3. 현재 대상 확인
            # --------------------------------
            state_name = str(state).split(".")[-1].upper()

            if target_index is not None:
                if 0 <= target_index < len(detections):
                    target_bbox = detections[target_index]["bbox"]

            # --------------------------------
            # 6-4. 대상 위치를 회전각으로 변환
            # --------------------------------
            result = angle_controller.update_bbox(
                target_bbox
            )

            # --------------------------------
            # 6-5. 서보모터 제어
            # --------------------------------
            if servo is not None and target_bbox is not None:

                command_angle = result.command_angle

                if (
                    last_servo_angle is None
                    or abs(command_angle - last_servo_angle) >= 0.5
                ):
                    servo.set_angle(command_angle)
                    last_servo_angle = command_angle

            # --------------------------------
            # 6-6. 화면 출력
            # --------------------------------
            draw_detection(frame, detections)

            if target_bbox is not None:

                x1, y1, x2, y2 = map(int, target_bbox)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(
                    frame,
                    (cx, cy),
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    "TARGET",
                    (x1, max(20, y1 - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # 상태 표시
            cv2.putText(
                frame,
                f"State: {state_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Angle: {result.command_angle:.1f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.imshow("Face Tracking Fan", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\nTracking stopped")

    finally:
        if camera is not None:
            camera.stop()

        if servo is not None:
            servo.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
