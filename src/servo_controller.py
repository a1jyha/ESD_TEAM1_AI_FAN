#	픽셀 좌표 → 목표 각도 변환과 서보 제어. 데드밴드, 회전 속도 제한, 회전 범위 제한을 적용합니다.
"""
servo_controller.py

Raspberry Pi 4 + SG90 단독 구동 테스트
신호선: BCM GPIO2 = 물리 핀 3번

카메라는 사용하지 않습니다.
터미널에서 각도를 입력하면 실제 모터에 PWM을 출력합니다.
"""

import math

import RPi.GPIO as GPIO


class ServoController:
    # 최초 테스트에서는 끝 위치를 피하도록 범위를 제한합니다.
    MIN_ANGLE = 60.0
    MAX_ANGLE = 120.0
    CENTER_ANGLE = 90.0

    PWM_FREQUENCY = 50  # Hz

    # 보수적인 초기 펄스 설정입니다.
    # 실제 회전각과의 관계는 모터별로 확인·보정해야 합니다.
    PULSE_AT_0_MS = 1.0
    PULSE_AT_180_MS = 2.0

    def __init__(self, pin=2):
        self.pin = pin
        self.last_command_angle = None
        self._closed = False
        self._pwm = None

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

        try:
            self._pwm = GPIO.PWM(self.pin, self.PWM_FREQUENCY)
            self._pwm.start(0)  # 처음에는 회전 명령을 보내지 않습니다.
        except BaseException:
            GPIO.cleanup(self.pin)
            raise

        print(f"[준비] BCM GPIO{self.pin}, PWM 50 Hz")
        print(
            f"[범위] {self.MIN_ANGLE:.0f}~{self.MAX_ANGLE:.0f}도"
            f" | 정면 기준 {self.CENTER_ANGLE:.0f}도"
        )
        print("[주의] 출력 각도는 명령값이며 실제 측정 각도가 아닙니다.")

    def _check_open(self):
        if self._closed:
            raise RuntimeError("이미 종료한 ServoController입니다.")

    def set_angle(self, target_angle):
        """각도를 검증하고 PWM 신호를 출력합니다."""
        self._check_open()

        angle = float(target_angle)

        if not math.isfinite(angle):
            raise ValueError("유한한 숫자를 입력하세요.")

        if not self.MIN_ANGLE <= angle <= self.MAX_ANGLE:
            raise ValueError(
                f"{self.MIN_ANGLE:.0f}~{self.MAX_ANGLE:.0f}도 "
                "범위에서 입력하세요."
            )

        # 각도 명령을 펄스 폭(ms)으로 변환
        pulse_ms = self.PULSE_AT_0_MS + (
            angle / 180.0
        ) * (self.PULSE_AT_180_MS - self.PULSE_AT_0_MS)

        # 50 Hz이면 주기 20 ms
        period_ms = 1000.0 / self.PWM_FREQUENCY

        # 듀티비(%) = 펄스 폭 / 주기 × 100
        duty_cycle = pulse_ms / period_ms * 100.0

        self._pwm.ChangeDutyCycle(duty_cycle)
        self.last_command_angle = angle

        print(
            f"[명령 전송] {angle:.1f}도"
            f" | 펄스 {pulse_ms:.3f} ms"
            f" | 듀티비 {duty_cycle:.2f}%"
        )

        return angle

    def center(self):
        """정면 기준 위치로 명령합니다."""
        return self.set_angle(self.CENTER_ANGLE)

    def hold(self):
        """현재 출력 중인 PWM 명령을 그대로 유지합니다."""
        self._check_open()

        if self.last_command_angle is None:
            print("[유지] 아직 전송한 각도 명령이 없습니다.")
            return None

        print(
            f"[유지] 마지막 명령 "
            f"{self.last_command_angle:.1f}도"
        )
        return self.last_command_angle

    def release(self):
        """PWM 출력을 중지합니다. 위치 유지력은 보장되지 않습니다."""
        self._check_open()
        self._pwm.ChangeDutyCycle(0)
        self.last_command_angle = None
        print("[해제] PWM 출력 중지")

    def close(self):
        """PWM을 종료하고 사용한 GPIO 핀을 정리합니다."""
        if self._closed:
            return

        try:
            if self._pwm is not None:
                self._pwm.ChangeDutyCycle(0)
                self._pwm.stop()
                self._pwm = None
        finally:
            GPIO.cleanup(self.pin)
            self._closed = True

        print("[종료] PWM 중지 및 GPIO 정리 완료")


def main():
    servo = None

    try:
        servo = ServoController(pin=2)

        print("\n숫자: 목표 각도 입력 (60~120)")
        print("c: 정면 기준 90도")
        print("h: 현재 명령 유지")
        print("s: PWM 출력 중지")
        print("q: 종료")
        print("\n먼저 90을 입력하고, 잠시 기다린 뒤 80과 100을 시험하세요.\n")

        while True:
            command = input("각도 또는 명령 > ").strip().lower()

            if command == "q":
                break
            elif command == "c":
                servo.center()
            elif command == "h":
                servo.hold()
            elif command == "s":
                servo.release()
            else:
                try:
                    servo.set_angle(command)
                except ValueError as error:
                    print(f"[입력 오류] {error}")

    except (KeyboardInterrupt, EOFError):
        print("\n종료 요청을 받았습니다.")

    finally:
        if servo is not None:
            servo.close()


if __name__ == "__main__":
    main()
