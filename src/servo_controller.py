#	픽셀 좌표 → 목표 각도 변환과 서보 제어. 데드밴드, 회전 속도 제한, 회전 범위 제한을 적용합니다.

"""
servo_controller.py

모터 없이 목표 각도를 확인하는 테스트용 모듈.
GPIO를 사용하지 않으며 실제 모터는 움직이지 않습니다.
"""

import math


class ServoController:
    def __init__(
        self,
        min_angle=30.0,
        max_angle=150.0,
        center_angle=90.0,
    ):
        # 아래 범위는 소프트웨어 테스트용 임시 설정입니다.
        # 실제 모터 연결 전 허용 범위를 확인해야 합니다.
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        self.center_angle = float(center_angle)

        values = (
            self.min_angle,
            self.max_angle,
            self.center_angle,
        )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("각도 설정은 유한한 숫자여야 합니다.")

        if not (
            0 <= self.min_angle
            <= self.center_angle
            <= self.max_angle
            <= 180
        ):
            raise ValueError(
                "0 <= 최소 각도 <= 정면 각도 <= 최대 각도 <= 180"
                " 조건을 만족해야 합니다."
            )

        # 실제 측정 각도가 아니라 마지막 명령 각도입니다.
        self.last_command_angle = self.center_angle

        print("[모의 구동] GPIO 출력 없음 · 실제 모터 동작 없음")
        print(
            f"[설정] 범위: {self.min_angle:.1f}~"
            f"{self.max_angle:.1f}° | "
            f"정면: {self.center_angle:.1f}°"
        )

    def set_angle(self, target_angle):
        """목표 각도를 범위 안으로 제한하고 출력합니다."""
        requested_angle = float(target_angle)

        if not math.isfinite(requested_angle):
            raise ValueError("목표 각도는 유한한 숫자여야 합니다.")

        command_angle = max(
            self.min_angle,
            min(self.max_angle, requested_angle),
        )

        previous_angle = self.last_command_angle
        change = command_angle - previous_angle

        self.last_command_angle = command_angle

        limited = (
            " | 범위 제한 적용"
            if command_angle != requested_angle
            else ""
        )

        print(
            f"[각도 명령] 요청: {requested_angle:.1f}°"
            f" | 출력: {command_angle:.1f}°"
            f" | 변화: {change:+.1f}°"
            f"{limited}"
        )

        return command_angle

    def hold(self):
        """대상 상실 시 마지막 명령 각도를 유지합니다."""
        print(
            "[각도 유지] 대상 상실 또는 판단 불확실"
            f" | 마지막 명령: {self.last_command_angle:.1f}°"
        )

        return self.last_command_angle

    def center(self):
        """정면 기준 각도로 명령을 갱신합니다."""
        return self.set_angle(self.center_angle)

    def close(self):
        """모의 구동 종료. 모터 이동 명령은 보내지 않습니다."""
        print("[종료] 서보모터 모의 구동 종료")


def main():
    servo = ServoController()

    print("\n숫자 입력: 목표 각도 출력")
    print("h: 마지막 각도 유지")
    print("c: 정면 각도로 초기화")
    print("q: 종료\n")

    try:
        while True:
            user_input = input("목표 각도 또는 명령 입력 > ").strip()

            if user_input.lower() == "q":
                break

            if user_input.lower() == "h":
                servo.hold()
                continue

            if user_input.lower() == "c":
                servo.center()
                continue

            try:
                servo.set_angle(user_input)
            except ValueError:
                print(
                    "[입력 오류] 유한한 숫자 또는 h, c, q를 입력하세요."
                )

    except (KeyboardInterrupt, EOFError):
        print("\n종료 요청을 받았습니다.")

    finally:
        servo.close()


if __name__ == "__main__":
    main()
