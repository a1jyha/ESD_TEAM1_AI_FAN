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
        print("[방식] 0.5도 이상 차이날 때만 듀티비 업데이트 (데드밴드 적용)")
        print("[상태] 숫자를 입력하기 전까지 PWM 출력 없음")

    def _check_open(self):
        if self.closed:
            raise RuntimeError("이미 종료된 컨트롤러입니다.")

    def angle_to_duty(self, angle):
        return self.min_duty + (
            angle * (self.max_duty - self.min_duty) / 180.0
        )

    def set_angle(self, angle):
        self._check_open()

        angle = float(angle)

        if not math.isfinite(angle):
            raise ValueError("유한한 숫자를 입력하세요.")

        if not self.min_angle <= angle <= self.max_angle:
            raise ValueError(
                f"{self.min_angle:.0f}~{self.max_angle:.0f}도만 "
                "입력할 수 있습니다."
            )

        # 데드밴드: 이전 각도와 0.5도 미만 차이나면 무시하여 떨림 방지
        if self.last_command_angle is not None and abs(self.last_command_angle - angle) < 0.5:
            return angle

        duty = self.angle_to_duty(angle)
        self.last_command_angle = angle
        self.enabled = True
        self.pwm.ChangeDutyCycle(duty)
        
        print(
            f"[명령] 목표 {angle:.1f}도"
            f" / 듀티비 {duty:.2f}%"
        )
        return angle

    def update(self):
        pass

    def center(self):
        return self.set_angle(self.center_angle)

    def hold(self):
        self._check_open()
        if not self.enabled:
            print("[상태] PWM이 중지되어 있습니다. 각도를 입력하세요.")
            return
        print(f"[유지] 마지막 명령 {self.last_command_angle:.1f}도")

    def release(self):
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
            readable, _, _ = select.select([sys.stdin], [], [], 0)

            if readable:
                line = sys.stdin.readline()
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

            time.sleep(0.1)

    except (KeyboardInterrupt, EOFError):
        print("\n종료 요청")

    finally:
        if servo is not None:
            servo.close()


if __name__ == "__main__":
    main()
