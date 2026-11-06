# 핵심 로직
# 시작 시 중앙에 가장 가까운 사람을 대상으로 고정하고, 다른 사람이 나타나도 유지하며, 대상을 잃으면 마지막 각도를 유지합니다. 중간 프레임 추적기도 여기에 둡니다.
import math


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a, b):
    """두 박스의 겹침 정도: 0~1."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])

    intersection = max(0, right - left) * max(0, bottom - top)
    union = box_area(a) + box_area(b) - intersection

    return intersection / union if union > 0 else 0.0


class TargetTracker:
    def __init__(
        self,
        max_gap_seconds=1.5,
        min_area_ratio=0.4,
        ambiguity_margin=0.15,
    ):
        # 아래 값들은 예비 시험용 설정값
        self.max_gap_seconds = max_gap_seconds
        self.min_area_ratio = min_area_ratio
        self.ambiguity_margin = ambiguity_margin
        self.reset()

    def reset(self):
        self.box = None
        self.last_seen = None
        self.state = "WAITING"
        self.locked_lost = False

    def update(self, detections, frame_shape, timestamp):
        """
        반환값:
            target_index: 현재 검출 목록에서 선택한 대상의 번호 또는 None
            state: WAITING / TRACKING / LOST

        timestamp에는 perf_counter() 값을 전달한다.
        """
        height, width = frame_shape[:2]

        # 아직 한 번도 대상을 선택하지 않은 경우
        if self.box is None:
            if not detections:
                return None, self.state

            # 최초 선택에만 화면 가로 중앙과의 거리 사용
            target_index = min(
                range(len(detections)),
                key=lambda i: abs(
                    box_center(detections[i]["bbox"])[0] - width / 2
                ),
            )

            self._accept(detections[target_index], timestamp)
            return target_index, self.state

        # 장시간 상실 후에는 다른 사람을 자동으로 선택하지 않는다.
        if self.locked_lost:
            return None, "LOST"

        elapsed = timestamp - self.last_seen

        if elapsed > self.max_gap_seconds:
            self.state = "LOST"
            self.locked_lost = True
            return None, self.state

        if not detections:
            self.state = "LOST"
            return None, self.state

        previous_box = self.box
        previous_center = box_center(previous_box)
        previous_area = box_area(previous_box)

        # 프레임 간격이 길면 이동 허용 거리를 조금 늘리되 상한 적용
        frame_diagonal = math.hypot(width, height)
        distance_gate = min(0.20, 0.04 + 0.15 * elapsed)

        candidates = []

        for index, detection in enumerate(detections):
            box = detection["bbox"]
            area = box_area(box)

            if area <= 0 or previous_area <= 0:
                continue

            area_ratio = min(area, previous_area) / max(
                area, previous_area
            )

            # 박스 크기가 지나치게 달라진 대상은 제외
            if area_ratio < self.min_area_ratio:
                continue

            center = box_center(box)
            distance = math.hypot(
                center[0] - previous_center[0],
                center[1] - previous_center[1],
            )
            normalized_distance = distance / frame_diagonal

            if normalized_distance > distance_gate:
                continue

            overlap = box_iou(previous_box, box)
            proximity = 1.0 - normalized_distance / distance_gate

            match_score = (
                0.50 * proximity
                + 0.35 * overlap
                + 0.15 * area_ratio
            )

            candidates.append((match_score, index))

        if not candidates:
            self.state = "LOST"
            return None, self.state

        candidates.sort(reverse=True)

        # 두 후보가 비슷하면 억지로 연결하지 않는다.
        if len(candidates) >= 2:
            score_difference = candidates[0][0] - candidates[1][0]

            if score_difference < self.ambiguity_margin:
                self.state = "LOST"
                return None, self.state

        target_index = candidates[0][1]
        self._accept(detections[target_index], timestamp)

        return target_index, self.state

    def _accept(self, detection, timestamp):
        self.box = tuple(detection["bbox"])
        self.last_seen = timestamp
        self.state = "TRACKING"
