"""공통 영상 입력: Picamera2/저장 영상, 회전·반전, 종료 처리.

기본: 왼쪽 90도 회전 후 좌우반전. read() 결과 기준으로 검출/각도 계산.
python camera.py로 단독 시험 가능. import만으로 카메라를 켜지 않습니다.
"""
import argparse
import math
import cv2

DEFAULT_ROTATE = 90  # 반시계 방향: 0, 90, 180, 270
DEFAULT_MIRROR = True


class Camera:
    def __init__(self, width=640, height=480, rotate=DEFAULT_ROTATE,
                 mirror=DEFAULT_MIRROR, video=None, video_fps=None):
        if rotate not in (0, 90, 180, 270):
            raise ValueError('rotate는 0, 90, 180, 270 중 하나입니다.')
        if width <= 1 or height <= 1:
            raise ValueError('카메라 해상도는 1보다 커야 합니다.')
        self.rotate = rotate
        self.mirror = mirror
        self.source = None
        self.started = False
        self.closed = False
        self.is_video = video is not None
        self.fps = None
        try:
            if self.is_video:
                self.source = cv2.VideoCapture(str(video))
                if not self.source.isOpened():
                    raise RuntimeError(f'영상을 열 수 없습니다: {video}')
                self.fps = video_fps if video_fps is not None else self.source.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(self.fps) or self.fps <= 0:
                    raise ValueError('영상 FPS가 유효하지 않습니다. --video-fps를 지정하세요.')
            else:
                from picamera2 import Picamera2
                self.source = Picamera2()
                config = self.source.create_preview_configuration(
                    main={'size': (width, height), 'format': 'RGB888'})
                self.source.configure(config)
                self.source.start()
                self.started = True
        except BaseException:
            self.close()
            raise

    def read(self):
        """변환된 BGR 프레임 반환. 저장 영상 끝은 None. 카메라 오류는 예외."""
        if self.closed:
            raise RuntimeError('이미 종료된 영상 입력입니다.')
        if self.is_video:
            ok, frame = self.source.read()
            if not ok:
                return None
        else:
            frame = self.source.capture_array('main')
        if self.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def close(self):
        """여러 번 호출해도 안전. 창 종료는 호출자가 담당합니다."""
        if self.closed:
            return
        self.closed = True
        if self.source is None:
            return
        if self.is_video:
            self.source.release()
        else:
            try:
                if self.started:
                    self.source.stop()
            finally:
                self.started = False
                self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--rotate', type=int, choices=(0,90,180,270), default=DEFAULT_ROTATE)
    p.add_argument('--mirror', action=argparse.BooleanOptionalAction, default=DEFAULT_MIRROR)
    a = p.parse_args()
    try:
        with Camera(a.width, a.height, a.rotate, a.mirror) as camera:
            print('영상 창을 클릭한 뒤 q 또는 Esc를 누르면 종료합니다.')
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            while True:
                frame = camera.read()
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 255 in (ord('q'),27):
                    break
                if cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        print('\n카메라 시험을 종료합니다.')
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
