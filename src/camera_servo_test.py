"""PiCamera -> 기존 검출/대상 연결 -> 각도 계산 -> pigpio 서보 통합 시험.

동일 폴더 필요: detectors.py, target_tracker.py, angle_controller.py,
servo_controller.py. camera.py는 가져오지 않고 Picamera2를 직접 관리합니다.
카메라는 회전축 가까이에 고정하고 모터만 회전시킵니다.
표시되는 각도는 명령값이며 실제 축 각도 측정값이 아닙니다.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from detectors import create_detector
from target_tracker import TargetTracker
from angle_controller import AngleController


def arguments():
    root = Path(__file__).resolve().parent
    if root.name == "src":
        root = root.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("yunet", "yolo"), default="yunet")
    p.add_argument("--model-dir", type=Path, default=root / "models")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--input-size", type=int)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    calibration = p.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--hfov", type=float, help="실제 촬영 영상의 수평 시야각(도)")
    calibration.add_argument("--fx", type=float, help="현재 영상 해상도 기준 가로 초점거리(px)")
    p.add_argument("--cx", type=float)
    p.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    p.add_argument("--speed", type=float, default=30.0)
    p.add_argument("--tau", type=float, default=0.15)
    p.add_argument("--deadband", type=float, default=1.0)
    p.add_argument("--host", default="::1")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--pin", type=int, default=2, help="BCM 번호: 2는 물리 3번")
    p.add_argument("--dry-run", action="store_true", help="카메라와 계산만 실행, GPIO 미사용")
    a = p.parse_args()
    if a.input_size is None:
        a.input_size = 320 if a.model == "yunet" else 192
    if a.input_size <= 0 or a.input_size % 32:
        p.error("input-size는 양의 32 배수여야 합니다.")
    if a.width <= 1 or a.height <= 1:
        p.error("영상 크기는 1보다 커야 합니다.")
    return a


def draw_box(frame, detection, selected):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, detection["bbox"])
    color = (0, 255, 0) if selected else (150, 150, 150)
    cv2.rectangle(frame, (x1, y1), (min(x2, w-1), min(y2, h-1)),
                  color, 3 if selected else 1)
    if selected:
        cv2.circle(frame, (round((x1+x2)/2), round((y1+y2)/2)),
                   5, (0, 0, 255), -1)
        cv2.putText(frame, "TARGET", (x1, max(100, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    args = arguments()
    controller = AngleController(
        frame_width=args.width, fx=args.fx, cx=args.cx,
        hfov_deg=args.hfov, direction=args.direction,
        center_angle=90, min_angle=60, max_angle=120,
        smoothing_tau=args.tau, deadband_deg=args.deadband,
        max_speed_deg_s=args.speed,
    )
    detector = create_detector(args.model, args.model_dir,
                               input_size=args.input_size,
                               model_path=args.model_path)
    print("Checking model before motor connection...", flush=True)
    detector.detect(np.zeros((args.height, args.width, 3), dtype=np.uint8))
    print("Model OK.", flush=True)
    tracker = TargetTracker(max_gap_seconds=1.5)
    from picamera2 import Picamera2

    window = "Smart Fan - Camera Servo Test"
    servo = None
    paused = False
    last_report = 0.0
    print("Fixed camera / motor only rotates. Range: 60..120 deg.")
    if args.hfov is not None:
        print(f"HFOV approximation: {args.hfov} deg. Unmeasured values are for trial only.")
    print("q/Esc: quit | s: hold last command | r: resume/reset target")
    print("Click the camera window before pressing keys.")
    if not args.dry_run:
        print("MOTOR ENABLED: initialization commands 90 degrees.", flush=True)

    # LIFO cleanup: motor pulse off first, then camera stop/close, then GUI.
    with ExitStack() as stack:
        stack.callback(cv2.destroyAllWindows)
        camera = Picamera2()
        stack.callback(camera.close)
        config = camera.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"})
        camera.configure(config)
        camera.start()
        stack.callback(camera.stop)
        frame = camera.capture_array("main")
        if frame.shape[:2] != (args.height, args.width):
            raise RuntimeError(f"Unexpected camera size: {frame.shape}; check calibration.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.width, args.height)
        if not args.dry_run:
            from servo_controller import ServoController
            servo = ServoController(pin=args.pin, host=args.host, port=args.port)
            stack.callback(servo.close)
            servo.center()
            time.sleep(0.7)  # 초기 정면 이동 대기; 실제 도달 측정은 아님
        controller.reset(angle=90.0)

        while True:
            loop_start = time.perf_counter()
            frame = camera.capture_array("main")
            capture_time = time.perf_counter()
            if frame.shape[:2] != (args.height, args.width):
                raise RuntimeError("Camera dimensions changed; stop and recalibrate.")
            detect_start = time.perf_counter()
            detections = detector.detect(frame)
            detect_ms = (time.perf_counter() - detect_start) * 1000
            index, state = tracker.update(detections, frame.shape, capture_time)

            # 처리 프레임마다 키 확인. 추론 중에는 키 반응이 지연될 수 있습니다.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                paused = True
                print("PAUSED: holding last command (power remains on).")
            elif key == ord("r"):
                paused = False
                tracker.reset()
                controller.reset(angle=controller.command_angle)
                index, state = None, "WAITING"
                print("Target reset. Next detection selects closest to center.")

            bbox = None
            if not paused and index is not None:
                bbox = detections[index]["bbox"]
            result = controller.update_bbox(bbox)
            if servo is not None and result.changed:
                servo.set_angle(result.command_angle)

            for i, detection in enumerate(detections):
                draw_box(frame, detection, i == index)
            cv2.line(frame, (round(controller.cx), 0),
                     (round(controller.cx), args.height-1), (255, 255, 0), 1)
            mode = "DRY RUN" if args.dry_run else "MOTOR ON"
            status = "PAUSED" if paused else state
            lines = [
                f"{mode} | {status} | command: {result.command_angle:.1f} deg",
                f"{args.model} {args.input_size} | inference: {detect_ms:.0f} ms",
                "q: quit | s: hold | r: resume/reset target",
            ]
            if tracker.locked_lost:
                lines.append("Target locked LOST: press r to start a new test")
            if result.limited:
                lines.append("Target outside servo angle limits")
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, 23 + 23*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
            cv2.imshow(window, frame)
            now = time.perf_counter()
            if now - last_report >= 1:
                print(f"{status} | command={result.command_angle:.1f} | "
                      f"detect={detect_ms:.0f}ms | loop={(now-loop_start)*1000:.0f}ms")
                last_report = now


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as error:
        print(f"ERROR ({type(error).__name__}): {error}", file=sys.stderr)
        sys.exit(1)
