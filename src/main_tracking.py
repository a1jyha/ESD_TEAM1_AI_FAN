"""Periodic detection + sparse Lucas-Kanade optical flow (no contrib required).

Intermediate frames estimate translation only; AI detections refresh box size
and feature points. This is not face recognition or guaranteed identity tracking.
Uses the shared CLI/camera/metrics runner in main_detect.py.
"""
import cv2
import numpy as np
from target_tracker import TargetTracker
from main_detect import run, entry


class OpticalFlowTracker:
    """Track box translation using forward/backward-checked feature points."""
    MIN_POINTS = 6

    def init(self, frame, box):
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.box = np.asarray(box, dtype=np.float32)
        height, width = self.gray.shape
        x1, y1, x2, y2 = self.box
        # Prefer interior features over box edges/background.
        dx, dy = (x2-x1)*0.1, (y2-y1)*0.1
        left, top = max(0, int(x1+dx)), max(0, int(y1+dy))
        right, bottom = min(width, int(x2-dx)), min(height, int(y2-dy))
        mask = np.zeros_like(self.gray)
        mask[top:bottom, left:right] = 255
        self.points = cv2.goodFeaturesToTrack(
            self.gray, maxCorners=80, qualityLevel=0.01,
            minDistance=5, mask=mask, blockSize=7)
        return self.points is not None and len(self.points) >= self.MIN_POINTS

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != self.gray.shape:
            return None
        params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                30, 0.01))
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, None, **params)
        if forward is None or status is None:
            return None
        old = self.points.reshape(-1, 2)
        new = forward.reshape(-1, 2)
        height, width = gray.shape
        good = ((status.ravel() == 1) & np.isfinite(new).all(axis=1)
                & (new[:, 0] >= 0) & (new[:, 0] < width)
                & (new[:, 1] >= 0) & (new[:, 1] < height))
        old, new = old[good], new[good]
        if len(new) < self.MIN_POINTS:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, new.reshape(-1, 1, 2), None, **params)
        if backward is None or back_status is None:
            return None
        backward = backward.reshape(-1, 2)
        good = ((back_status.ravel() == 1)
                & np.isfinite(backward).all(axis=1)
                & (np.linalg.norm(backward-old, axis=1) <= 1.5))
        old, new = old[good], new[good]
        if len(new) < self.MIN_POINTS:
            return None
        moves = new-old
        median = np.median(moves, axis=0)
        residual = np.linalg.norm(moves-median, axis=1)
        # Reject inconsistent motions; thresholds are tunable, not accuracy guarantees.
        inliers = residual <= max(2.0, 2.5*float(np.median(residual)))
        if np.count_nonzero(inliers) < max(self.MIN_POINTS, int(np.ceil(len(new)*0.6))):
            return None
        shift = np.median(moves[inliers], axis=0)
        candidate = self.box + np.tile(shift, 2)
        candidate[[0, 2]] = np.clip(candidate[[0, 2]], 0, width)
        candidate[[1, 3]] = np.clip(candidate[[1, 3]], 0, height)
        if candidate[2]-candidate[0] <= 2 or candidate[3]-candidate[1] <= 2:
            return None
        retained = new[inliers]
        inside = ((retained[:, 0] >= candidate[0]) & (retained[:, 0] < candidate[2])
                  & (retained[:, 1] >= candidate[1]) & (retained[:, 1] < candidate[3]))
        retained = retained[inside]
        if len(retained) < self.MIN_POINTS:
            return None
        self.gray = gray
        self.points = retained.astype(np.float32).reshape(-1, 1, 2)
        self.box = candidate
        return tuple(float(v) for v in candidate)


class HybridTracker:
    """Detection association + LK. Only AI confirmation updates last_seen."""
    def __init__(self, interval=3, max_gap=1.5):
        for name in ('calcOpticalFlowPyrLK', 'goodFeaturesToTrack'):
            if not hasattr(cv2, name):
                raise RuntimeError(f'OpenCV function missing: {name}')
        if interval < 1 or max_gap <= 0:
            raise ValueError('interval must be >= 1; max_gap must be > 0')
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
                or (a.last_seen is not None and timestamp-a.last_seen >= self.max_gap*0.75))

    def detect(self, detections, frame, timestamp, frame_no):
        self.last_detection_frame = frame_no
        index, state = self.association.update(detections, frame.shape, timestamp)
        self.visual = None
        if index is None:
            return None, state
        box = detections[index]['bbox']
        visual = OpticalFlowTracker()
        if visual.init(frame, box):
            self.visual = visual
        # A valid detection is usable even when too few corners were found.
        # Next frame will then run detection again.
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
        box = self.visual.update(frame)
        if box is None:
            self.visual = None
            # Shared runner immediately tries detection on this same frame.
            return None, 'LOST'
        a.box = box
        return box, 'TRACKING_LK'


if __name__ == '__main__':
    entry(lambda: run('tracking', HybridTracker))
