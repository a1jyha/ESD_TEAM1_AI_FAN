# 좌표를 각도로 변환

"""고정 카메라용 좌표->각도 제어. GPIO/pigpio를 사용하지 않습니다.

입력 좌표와 fx, cx는 반드시 같은 해상도/크롭/좌우 방향 기준이어야 합니다.
검출기의 resize/letterbox 좌표는 원본 영상 좌표로 복원해서 전달하세요.
렌즈 왜곡은 이 모듈에서 보정하지 않습니다. 카메라와 회전축이 가까우며
정면 방향이 정렬되어 있다는 근사입니다. 반환값은 실제 측정 각도가 아닙니다.
"""
import argparse
import math
import time
from dataclasses import dataclass


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}은 유한한 숫자여야 합니다.")
    return value


def clip(value, low, high):
    return max(low, min(high, value))


@dataclass(frozen=True)
class AngleResult:
    detected: bool
    raw_target: object
    filtered_target: object
    command_angle: float
    changed: bool
    limited: bool


class AngleController:
    def __init__(self, frame_width, fx=None, cx=None, hfov_deg=None,
                 center_angle=90.0, direction=1, min_angle=60.0,
                 max_angle=120.0, smoothing_tau=0.15,
                 deadband_deg=1.0, max_speed_deg_s=30.0, max_dt=0.1):
        """fx: 픽셀 단위 초점거리. hfov_deg는 fx가 없을 때 근사에 사용.

        smoothing_tau: 평활화 시간 상수(초). 0이면 비활성화.
        deadband_deg: 마지막 명령 각도와의 차이가 이 값 이하면 유지.
        max_speed_deg_s: 명령 각도의 최대 변화 속도(실제 측정 속도 아님).
        max_dt: 처리 중단 후 급격한 이동을 막는 계산 시간 간격 상한(초).
        초기 상태는 중심 각도를 명령한 것으로 가정. 통합 시 servo.center()로 맞추세요.
        """
        self.width = finite(frame_width, "frame_width")
        if self.width <= 1:
            raise ValueError("frame_widthは1より大きくしてください。")
        self.cx = self.width / 2 if cx is None else finite(cx, "cx")
        if not 0 <= self.cx < self.width:
            raise ValueError("cxは画像内の座標にしてください。")
        if fx is not None and hfov_deg is not None:
            raise ValueError("fxとhfov_degはどちらか一方を指定してください。")
        if fx is None:
            if hfov_deg is None:
                raise ValueError("fxまたはhfov_degが必要です。")
            fov = finite(hfov_deg, "hfov_deg")
            if not 0 < fov < 180:
                raise ValueError("hfov_degは0より大きく180未満にしてください。")
            fx = self.width / (2 * math.tan(math.radians(fov) / 2))
        self.fx = finite(fx, "fx")
        if self.fx <= 0:
            raise ValueError("fxは正の値にしてください。")
        if direction not in (-1, 1):
            raise ValueError("directionは+1または-1です。")
        self.direction = direction
        self.center = finite(center_angle, "center_angle")
        self.low = finite(min_angle, "min_angle")
        self.high = finite(max_angle, "max_angle")
        if not self.low <= self.center <= self.high or self.low >= self.high:
            raise ValueError("min_angle < max_angle、中心角はその範囲内です。")
        self.tau = finite(smoothing_tau, "smoothing_tau")
        self.deadband = finite(deadband_deg, "deadband_deg")
        self.speed = finite(max_speed_deg_s, "max_speed_deg_s")
        self.max_dt = finite(max_dt, "max_dt")
        if self.tau < 0 or self.deadband < 0 or self.speed <= 0 or self.max_dt <= 0:
            raise ValueError("평활화·데드밴드는 0 이상, 속도·시간 상한은 양수입니다.")
        self.reset()

    def reset(self, angle=None, now=None):
        """초기화 또는 수동 각도 변경 후 내부 명령 각도를 동기화합니다."""
        angle = self.center if angle is None else finite(angle, "angle")
        if not self.low <= angle <= self.high:
            raise ValueError("초기 각도가 허용 범위를 벗어났습니다.")
        self.command_angle = angle
        self.filtered_target = None
        self.last_time = time.monotonic() if now is None else finite(now, "now")

    def update(self, x_center=None, now=None):
        """매 처리 프레임 호출. 추적 불확실/실패 시 x_center=None 전달.

        now는 테스트용 단조 증가 시간(초). 실제 사용에서는 생략하세요.
        """
        t = time.monotonic() if now is None else finite(now, "now")
        if t < self.last_time:
            raise ValueError("now는 이전 시간보다 작을 수 없습니다.")
        x = None if x_center is None else finite(x_center, "x_center")
        if x is not None and not 0 <= x < self.width:
            raise ValueError("x_center가 영상 범위를 벗어났습니다.")
        dt = min(t - self.last_time, self.max_dt)
        self.last_time = t
        if x is None:
            # 상실 중에는 이전 목표로 계속 움직이지 않습니다.
            self.filtered_target = None
            return AngleResult(False, None, None, self.command_angle, False, False)

        phi = math.degrees(math.atan((x - self.cx) / self.fx))
        raw = self.center + self.direction * phi
        target = clip(raw, self.low, self.high)
        if self.filtered_target is None:
            self.filtered_target = self.command_angle
        alpha = 1.0 if self.tau == 0 else -math.expm1(-dt / self.tau)
        self.filtered_target += alpha * (target - self.filtered_target)
        error = self.filtered_target - self.command_angle
        previous = self.command_angle
        if abs(error) > self.deadband:
            step = clip(error, -self.speed * dt, self.speed * dt)
            self.command_angle = clip(previous + step, self.low, self.high)
        return AngleResult(True, raw, self.filtered_target, self.command_angle,
                           self.command_angle != previous, target != raw)

    def update_bbox(self, bbox, now=None):
        """bbox=(x1,y1,x2,y2), 원본 프레임 좌표. 대상 없으면 None."""
        if bbox is None:
            return self.update(None, now)
        if len(bbox) != 4:
            raise ValueError("bbox는 (x1,y1,x2,y2)입니다.")
        x1, y1, x2, y2 = [finite(v, "bbox") for v in bbox]
        if not (0 <= x1 < x2 <= self.width and y1 < y2):
            raise ValueError("bbox의 범위 또는 좌표 순서를 확인하세요.")
        return self.update((x1 + x2) / 2, now)


def main():
    parser = argparse.ArgumentParser(description="모터 없이 좌표->각도 계산 시험")
    parser.add_argument("--width", type=int, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fx", type=float)
    group.add_argument("--hfov", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=1)
    args = parser.parse_args()
    try:
        controller = AngleController(args.width, fx=args.fx, cx=args.cx,
                                     hfov_deg=args.hfov, direction=args.direction)
    except ValueError as error:
        parser.error(str(error))
    print("GPIO 미사용: 모터는 움직이지 않습니다.")
    print("x 좌표 입력: 해당 위치가 1초 동안 관측된 상황을 20Hz로 모의 계산")
    print("n: 대상 없음 / q: 종료")
    try:
        while True:
            command = input("x > ").strip().lower()
            if command == "q":
                break
            if not command:
                continue
            try:
                x = None if command == "n" else float(command)
                # 입력을 기다린 시간을 제어 시간으로 누적하지 않습니다.
                controller.last_time = time.monotonic()
                for i in range(20):
                    time.sleep(0.05)
                    result = controller.update(x)
                    if i % 4 == 3:
                        print(result)
            except ValueError as error:
                print("입력 오류:", error)
    except (KeyboardInterrupt, EOFError):
        print("\n종료")


if __name__ == "__main__":
    main()
