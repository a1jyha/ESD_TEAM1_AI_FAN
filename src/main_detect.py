"""매 처리 프레임 AI 검출. main_tracking.py와 동일 입력/제어/측정 공통 실행부.

필요 파일: camera.py, detectors.py, target_tracker.py, metrics.py.
각도 계산 시 angle_controller.py, --motor 시 servo_controller.py 추가.
기존 detectors.py의 OpenCV DNN 실행 경로를 사용하며 INT8 호환성은 모델별 확인 필요.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from detectors import create_detector
from target_tracker import TargetTracker
from metrics import Metrics
from camera import Camera, DEFAULT_ROTATE, DEFAULT_MIRROR


def arguments(mode):
    root = Path(__file__).resolve().parent
    if root.name == 'src':
        root = root.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['yunet', 'yolo'], default='yunet')
    p.add_argument('--model-dir', type=Path, default=root/'models')
    p.add_argument('--model-path', type=Path)
    p.add_argument('--input-size', type=int)
    p.add_argument('--precision', choices=['FP32', 'INT8', 'unknown'], default='unknown', help='기록용 라벨, 변환 기능 아님')
    p.add_argument('--video', type=Path, help='미지정 시 Picamera2. 영상은 모든 프레임 순서대로 처리')
    p.add_argument('--video-fps', type=float, help='파일 FPS 메타데이터가 부정확할 때 지정')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--rotate', type=int, choices=(0,90,180,270), default=DEFAULT_ROTATE, help='반시계 회전 각도')
    p.add_argument('--mirror', action=argparse.BooleanOptionalAction, default=DEFAULT_MIRROR, help='회전 후 좌우반전; 해제: --no-mirror')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--output-dir', type=Path, default=root/'results')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--motor', action='store_true', help='명시할 때만 모터 활성화, 최초 90도 이동')
    calibration = p.add_mutually_exclusive_group()
    calibration.add_argument('--hfov', type=float)
    calibration.add_argument('--fx', type=float)
    p.add_argument('--cx', type=float)
    p.add_argument('--direction', type=int, choices=[-1, 1], default=1)
    p.add_argument('--tau', type=float, default=.15)
    p.add_argument('--deadband', type=float, default=1.)
    p.add_argument('--speed', type=float, default=30.)
    p.add_argument('--host', default='::1')
    p.add_argument('--port', type=int, default=8888)
    p.add_argument('--pin', type=int, default=2)
    p.add_argument('--max-gap', type=float, default=1.5)
    if mode == 'tracking':
        p.add_argument('--detect-interval', type=int, default=3, help='처리 프레임 기준, 3이면 검출/추적/추적 반복')
    a = p.parse_args()
    if a.input_size is None:
        a.input_size = 320 if a.model == 'yunet' else 192
    if a.input_size <= 0 or a.input_size % 32:
        p.error('input-size는 양의 32 배수여야 합니다.')
    if min(a.width, a.height) <= 1 or a.threads < 1 or a.warmup < 1 or a.max_frames < 0:
        p.error('영상 크기/스레드/워밍업/프레임 제한 값을 확인하세요.')
    if not np.isfinite(a.max_gap) or a.max_gap <= 0:
        p.error('max-gap은 양수입니다.')
    if a.video_fps is not None and (not np.isfinite(a.video_fps) or a.video_fps <= 0):
        p.error('video-fps는 양수입니다.')
    if mode == 'tracking' and a.detect_interval < 1:
        p.error('detect-interval은 1 이상입니다.')
    if a.motor and (a.video or a.no_display):
        p.error('모터 시험은 카메라 + 화면 표시 모드에서 실행하세요.')
    if a.motor and a.hfov is None and a.fx is None:
        p.error('모터 제어에는 --hfov 또는 --fx 보정값이 필요합니다.')
    a.model_path = (a.model_path or a.model_dir/('face_detection_yunet_2026may.onnx' if a.model == 'yunet' else 'yolo11n.onnx')).resolve()
    return a


def run(mode='detect', hybrid_class=None):
    a = arguments(mode)
    cv2.setNumThreads(a.threads)
    detector = create_detector(a.model, a.model_dir, a.input_size, a.model_path)
    association = TargetTracker(max_gap_seconds=a.max_gap)
    hybrid = hybrid_class(a.detect_interval, a.max_gap) if hybrid_class else None
    controller = servo = meter = None
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    metadata['detectors_source_sha256'] = hashlib.sha256(Path(sys.modules[create_detector.__module__].__file__).read_bytes()).hexdigest()
    metadata.update(mode=mode, engine='OpenCV DNN (existing detectors.py)', opencv=cv2.__version__,
                    python=sys.version, numpy=np.__version__, model_bytes=a.model_path.stat().st_size,
                    model_sha256=hashlib.sha256(a.model_path.read_bytes()).hexdigest())
    reason = 'finished'
    try:
        with ExitStack() as stack:
            source = Camera(width=a.width, height=a.height,
                            rotate=a.rotate, mirror=a.mirror,
                            video=a.video, video_fps=a.video_fps)
            stack.callback(source.close)
            read = source.read
            fps = source.fps
            if a.video:
                metadata['source_fps'] = fps
            first = read()
            if first is None:
                raise RuntimeError('No input frame.')
            height, width = first.shape[:2]
            metadata['actual_frame_size'] = [width, height]
            print(f'입력 변환: 반시계 {a.rotate}도 / 좌우반전 {a.mirror} / 결과 {width}x{height}')
            if a.fx is not None or a.hfov is not None:
                print('보정값은 회전·반전 후 영상 기준이어야 합니다. 모터 방향도 확인하세요.')
            print('모델 실행 확인 / 워밍업...', flush=True)
            for _ in range(a.warmup):
                detector.detect(first)
            if a.fx is not None or a.hfov is not None:
                from angle_controller import AngleController
                controller = AngleController(width, fx=a.fx, cx=a.cx, hfov_deg=a.hfov,
                                             direction=a.direction, smoothing_tau=a.tau,
                                             deadband_deg=a.deadband, max_speed_deg_s=a.speed)
            if not a.no_display:
                stack.callback(cv2.destroyAllWindows)
                cv2.namedWindow('Smart Fan', cv2.WINDOW_NORMAL)
            if a.motor:
                from servo_controller import ServoController
                servo = ServoController(pin=a.pin, host=a.host, port=a.port)
                stack.callback(servo.close)
                print('MOTOR ON: 90도 초기 이동, 이후 60~120도 제한', flush=True)
                servo.center()
                time.sleep(.7)
            else:
                print('모터 미사용 (--motor 지정 시에만 움직임)')
            if controller:
                controller.reset(now=0. if a.video else time.monotonic())
            meter = Metrics(a.output_dir, metadata)
            print('q/Esc: 종료 | s: 각도 유지 | r: 대상 재선택/재개')
            number, paused = 0, False
            pending = first if a.video else None
            while a.max_frames == 0 or number < a.max_frames:
                start = time.perf_counter()
                frame = pending if pending is not None else read()
                pending = None
                received = time.perf_counter()
                if frame is None:
                    reason = 'end_of_video'
                    break
                if frame.shape[:2] != (height, width):
                    raise RuntimeError('Frame dimensions changed.')
                stamp = number/fps if a.video else time.monotonic()
                detect_ms, track_ms, detections = None, 0., []
                box, state, action = None, 'WAITING', 'detect'
                processing_start = time.perf_counter()
                do_detect = hybrid is None or hybrid.needs_detection(number, stamp)
                if hybrid and not do_detect:
                    t = time.perf_counter()
                    box, state = hybrid.track(frame, stamp)
                    track_ms += (time.perf_counter()-t)*1000
                    action = 'track' if box is not None else 'hold'
                    # 실패한 같은 프레임에서 즉시 재검출. LOST 잠금은 유지.
                    do_detect = box is None and not hybrid.association.locked_lost
                if do_detect:
                    t = time.perf_counter()
                    detections = detector.detect(frame)
                    detect_ms = (time.perf_counter()-t)*1000
                    t = time.perf_counter()
                    if hybrid:
                        box, state = hybrid.detect(detections, frame, stamp, number)
                    else:
                        index, state = association.update(detections, frame.shape, stamp)
                        box = detections[index]['bbox'] if index is not None else None
                    track_ms += (time.perf_counter()-t)*1000
                    action = 'detect'
                processing_ms = (time.perf_counter()-processing_start)*1000
                result = controller.update_bbox(None if paused else box, now=stamp) if controller else None
                if servo and result.changed:
                    servo.set_angle(result.command_angle)
                command_ms = (time.perf_counter()-received)*1000 if controller else None
                command = result.command_angle if result else None
                key = -1
                if not a.no_display:
                    for detection in detections:
                        x1,y1,x2,y2 = map(int, detection['bbox'])
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (140,140,140), 1)
                    if box is not None:
                        x1,y1,x2,y2 = map(round, box)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    label = f'{mode} | {action} | {state} | command={command if command is not None else "off"}'
                    if paused:
                        label = 'PAUSED MOTOR | ' + label
                    cv2.putText(frame, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                    cv2.putText(frame, 'q:quit  s:hold  r:reset target', (10,48), cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                    cv2.imshow('Smart Fan', frame)
                    key = cv2.waitKey(1) & 255
                    if cv2.getWindowProperty('Smart Fan', cv2.WND_PROP_VISIBLE) < 1:
                        key = ord('q')
                coords = box if box is not None else (None,)*4
                meter.record(frame=number, source_s=stamp if a.video else stamp-metadata.setdefault('live_start_monotonic', stamp),
                             action=action, state=state, target_available=box is not None,
                             capture_ms=(received-start)*1000, detect_pipeline_ms=detect_ms,
                             tracker_ms=track_ms, processing_ms=processing_ms,
                             frame_to_command_ms=command_ms, loop_ms=(time.perf_counter()-start)*1000,
                             x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3], command_deg=command, paused=paused)
                number += 1
                if key in (ord('q'),27):
                    reason = 'user_quit'
                    break
                if key == ord('s'):
                    paused = True
                if key == ord('r'):
                    paused = False
                    (hybrid if hybrid else association).reset()
                    if controller:
                        controller.reset(controller.command_angle, now=stamp)
    except KeyboardInterrupt:
        reason = 'interrupted'
    except BaseException as error:
        reason = f'error: {type(error).__name__}: {error}'
        raise
    finally:
        if meter:
            meter.close(reason)


def entry(function):
    try:
        function()
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception as error:
        print(f'ERROR ({type(error).__name__}): {error}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    entry(run)
