"""주기적 AI 검출 + 중간 프레임 KCF 추적. main_detect.py의 공통 실행부 사용."""
import math
import cv2
from target_tracker import TargetTracker
from main_detect import run, entry


def make_kcf():
    for namespace in (cv2, getattr(cv2, 'legacy', None)):
        factory = getattr(namespace, 'TrackerKCF_create', None)
        if factory is not None:
            return factory()
    raise RuntimeError('이 OpenCV에 KCF가 없습니다. contrib tracking 지원 빌드가 필요합니다. '
                       '현재 환경에 OpenCV 패키지를 중복 설치하기 전에 설치 구성을 확인하세요.')


class HybridTracker:
    """기존 박스 연결 규칙 + KCF. 얼굴 신원 인식/ReID가 아닙니다.

    KCF 위치는 다음 검출의 공간 연결 기준으로만 사용합니다.
    KCF 성공만으로 AI 확인 시각(last_seen)을 연장하지 않습니다.
    """
    def __init__(self, interval=3, max_gap=1.5):
        make_kcf()  # 카메라/모터 연결 전 지원 확인
        self.interval = interval
        self.max_gap = max_gap
        self.association = TargetTracker(max_gap_seconds=max_gap)
        self.reset()

    def reset(self):
        self.association.reset()
        self.visual = None
        self.last_detection_frame = None

    def needs_detection(self, frame_no, timestamp):
        a = self.association
        if a.locked_lost:
            return False
        return (self.visual is None or self.last_detection_frame is None
                or frame_no-self.last_detection_frame >= self.interval
                or (a.last_seen is not None and timestamp-a.last_seen >= self.max_gap*.75))

    def detect(self, detections, frame, timestamp, frame_no):
        self.last_detection_frame = frame_no
        index, state = self.association.update(detections, frame.shape, timestamp)
        self.visual = None
        if index is None:
            return None, state
        box = detections[index]['bbox']
        x1, y1, x2, y2 = map(int, box)
        self.visual = make_kcf()
        ok = self.visual.init(frame, (x1, y1, x2-x1, y2-y1))
        if ok is False:
            self.visual = None
        return box, state

    def track(self, frame, timestamp):
        a = self.association
        if a.locked_lost:
            return None, 'LOST'
        if a.last_seen is not None and timestamp-a.last_seen > self.max_gap:
            a.locked_lost = True
            self.visual = None
            return None, 'LOST'
        if self.visual is None:
            return None, 'LOST'
        ok, rect = self.visual.update(frame)
        if ok:
            x, y, w, h = rect
            height, width = frame.shape[:2]
            ok = all(math.isfinite(v) for v in rect) and w > 1 and h > 1
            if ok:
                box = (max(0., x), max(0., y), min(float(width), x+w), min(float(height), y+h))
                ok = box[2]-box[0] > 1 and box[3]-box[1] > 1
        if not ok:
            self.visual = None
            return None, 'LOST'
        a.box = box
        return box, 'TRACKING_KCF'


if __name__ == '__main__':
    entry(lambda: run('tracking', HybridTracker))
