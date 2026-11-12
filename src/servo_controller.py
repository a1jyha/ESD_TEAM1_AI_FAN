#	픽셀 좌표 → 목표 각도 변환과 서보 제어. 데드밴드, 회전 속도 제한, 회전 범위 제한을 적용합니다.
"""
servo_controller.py

SG90 단독 구동 테스트
신호선: BCM GPIO2 = 물리 핀 3번

실습 코드와 동일한 설정:
- PWM 주파수: 50 Hz
- 듀티비 변환 범위: 3~12%
- 마지막 각도 명령을 0.1초마다 반복 전달

일반 터미널에서 실행하세요.
"""

import math
import select
import sys
import time

import RPi.GPIO as GPIO


class ServoController:
    def __init__(self, pin=2):
        self.pin = pin

        self.frequency = 50
        self.min_duty = 3.0
        self.max_duty = 12.0

        # 최초 테스트에서는 끝 위치를 피합니다.
        self.min_angle = 60.0
        self.max_angle = 120.0
        self.center_angle = 90.0

        self.last_command_angle = None
        self.enabled = False
        self.closed = False
        self.pwm = None

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

        try:
            self.pwm = GPIO.PWM(self.pin, self.frequency)
            self.pwm.start(0)
        except BaseException:
            GPIO.cleanup(self.pin)
            raise

        print(f"[준비] BCM GPIO{self.pin} / PWM 50 Hz")
        print("[방식] 실습 코드와 동일한 듀티비 3~12% 매핑")
        print("[상태] 숫자를 입력하기 전까지 PWM 출력 없음")

    def _check_open(self):
        if self.closed:
            raise RuntimeError("이미 종료된 컨트롤러입니다.")

    def angle_to_duty(self, angle):
        """실습 코드와 동일한 각도 → 듀티비 변환."""
        return self.min_duty + (
            angle * (self.max_duty - self.min_duty) / 180.0
        )

    def set_angle(self, angle):
        """새 목표 각도를 저장하고 즉시 출력합니다."""
        self._check_open()

        angle = float(angle)

        if not math.isfinite(angle):
            raise ValueError("유한한 숫자를 입력하세요.")

        if not self.min_angle <= angle <= self.max_angle:
            raise ValueError(
                f"{self.min_angle:.0f}~{self.max_angle:.0f}도만 "
                "입력할 수 있습니다."
            )

        self.last_command_angle = angle
        self.enabled = True
        self.update()

        print(
            f"[명령] 목표 {angle:.1f}도"
            f" / 듀티비 {self.angle_to_duty(angle):.2f}%"
        )
        return angle

    def update(self):
        """반복 루프에서 호출하여 마지막 명령을 재전달합니다."""
        self._check_open()

        if self.enabled and self.last_command_angle is not None:
            duty = self.angle_to_duty(self.last_command_angle)
            self.pwm.ChangeDutyCycle(duty)

    def center(self):
        return self.set_angle(self.center_angle)

    def hold(self):
        """출력 중인 마지막 각도 명령을 유지합니다."""
        self._check_open()

        if not self.enabled:
            print("[상태] PWM이 중지되어 있습니다. 각도를 입력하세요.")
            return

        print(
            f"[유지] 마지막 명령 "
            f"{self.last_command_angle:.1f}도"
        )

    def release(self):
        """PWM 출력 중지. 모터 전원 자체를 끄는 것은 아닙니다."""
        self._check_open()
        self.enabled = False
        self.pwm.ChangeDutyCycle(0)
        print("[중지] PWM 출력 중지")

    def close(self):
        if self.closed:
            return

        try:
            if self.pwm is not None:
                self.pwm.ChangeDutyCycle(0)
                self.pwm.stop()
                self.pwm = None
        finally:
            GPIO.cleanup(self.pin)
            self.closed = True

        print("[종료] PWM 및 GPIO 정리 완료")


def main():
    servo = None

    try:
        servo = ServoController(pin=2)

        print("\n각도 입력: 60~120")
        print("c: 정면 90도")
        print("h: 현재 명령 유지")
        print("s: PWM 출력 중지")
        print("q: 종료")
        print("\n명령을 입력한 뒤 Enter를 누르세요.")
        print("첫 테스트는 90을 입력하세요.\n")

        while True:
            # 입력이 없어도 기다리지 않고 다음 단계로 진행합니다.
            readable, _, _ = select.select([sys.stdin], [], [], 0)

            if readable:
                line = sys.stdin.readline()

                # 터미널 입력이 닫히면 종료
                if line == "":
                    break

                command = line.strip().lower()

                if command == "q":
                    break
                elif command == "s":
                    servo.release()
                elif command == "c":
                    servo.center()
                elif command == "h":
                    servo.hold()
                elif command:
                    try:
                        servo.set_angle(command)
                    except ValueError as error:
                        print(f"[입력 오류] {error}")

            # 실습처럼 같은 명령을 약 0.1초마다 재전달
            servo.update()
            time.sleep(0.1)

    except (KeyboardInterrupt, EOFError):
        print("\n종료 요청")

    finally:
        if servo is not None:
            servo.close()


if __name__ == "__main__":
    main()
