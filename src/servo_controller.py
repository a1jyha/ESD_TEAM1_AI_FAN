import math

import pigpio


class ServoController:
    def __init__(self, pin=2, host="::1", port=8888):
        self.pin = pin

        self.min_angle = 60.0
        self.max_angle = 120.0
        self.center_angle = 90.0

        # 기존 실습의 50Hz, 듀티비 3~12%에 대응하는 값.
        # 실제 사용하는 60~120도는 1200~1800us에 해당합니다.
        self.pulse_at_0_us = 600
        self.pulse_at_180_us = 2400

        self.last_command_angle = None
        self.last_pulse_us = None
        self.enabled = False
        self.closed = False

        # 현재 pigpiod가 IPv6 로컬 주소 ::1에서 대기 중입니다.
        self.pi = pigpio.pi(host, port)

        if not self.pi.connected:
            self.pi.stop()
            self.closed = True
            raise RuntimeError(
                f"pigpiod 연결 실패: [{host}]:{port}\n"
                "pigpiod 실행 상태와 접속 주소를 확인하세요."
            )

        try:
            self.pi.set_mode(self.pin, pigpio.OUTPUT)
            self.pi.set_servo_pulsewidth(self.pin, 0)
            self.pi.write(self.pin, 0)
        except BaseException:
            self.pi.stop()
            self.closed = True
            raise

        print(f"[준비] BCM GPIO{self.pin} / pigpio")
        print(f"[연결] [{host}]:{port}")
        print("[상태] 각도를 입력하기 전까지 제어 펄스 없음")

    def _check_open(self):
        if self.closed:
            raise RuntimeError("이미 종료된 컨트롤러입니다.")

    def angle_to_pulse(self, angle):
        """명령 각도를 펄스 폭(us)으로 변환합니다."""
        pulse = self.pulse_at_0_us + (
            angle / 180.0
        ) * (self.pulse_at_180_us - self.pulse_at_0_us)

        return int(round(pulse))

    def set_angle(self, angle):
        """60~120도 범위의 목표 각도를 명령합니다."""
        self._check_open()

        angle = float(angle)

        if not math.isfinite(angle):
            raise ValueError("유한한 숫자를 입력하세요.")

        if not self.min_angle <= angle <= self.max_angle:
            raise ValueError(
                f"{self.min_angle:.0f}~{self.max_angle:.0f}도만 "
                "입력할 수 있습니다."
            )

        pulse_us = self.angle_to_pulse(angle)

        # 동일한 펄스 폭이라면 다시 명령하지 않습니다.
        # 펄스 자체는 pigpiod가 계속 반복 출력합니다.
        if not self.enabled or pulse_us != self.last_pulse_us:
            self.pi.set_servo_pulsewidth(self.pin, pulse_us)
            self.last_pulse_us = pulse_us
            self.enabled = True

        self.last_command_angle = angle

        print(
            f"[명령] 목표 {angle:.1f}도"
            f" / 펄스 폭 {pulse_us}us"
        )
        return angle

    def update(self):
        """pigpiod가 펄스를 유지하므로 반복 갱신은 필요 없습니다."""
        self._check_open()

    def center(self):
        return self.set_angle(self.center_angle)

    def hold(self):
        """현재 출력 중인 목표 각도 명령을 유지합니다."""
        self._check_open()

        if not self.enabled:
            print("[상태] 펄스 출력 중지 상태입니다. 각도를 입력하세요.")
            return

        print(
            f"[유지] 마지막 명령 {self.last_command_angle:.1f}도"
            f" / 펄스 폭 {self.last_pulse_us}us"
        )

    def release(self):
        """제어 펄스를 중지합니다. 모터 전원을 끄는 것은 아닙니다."""
        self._check_open()

        self.pi.set_servo_pulsewidth(self.pin, 0)
        self.pi.write(self.pin, 0)

        self.enabled = False
        self.last_pulse_us = None

        print("[중지] 제어 펄스 중지 — 각도 유지력은 보장되지 않습니다.")

    def close(self):
        if self.closed:
            return

        try:
            if self.pi.connected:
                self.pi.set_servo_pulsewidth(self.pin, 0)
                self.pi.write(self.pin, 0)
        finally:
            self.pi.stop()
            self.enabled = False
            self.closed = True

        print("[종료] 제어 펄스 중지 및 연결 해제")


def main():
    servo = None

    try:
        servo = ServoController(pin=2, host="::1", port=8888)

        print("\n각도 입력: 60~120")
        print("c: 정면 90도")
        print("h: 현재 명령 유지")
        print("s: 제어 펄스 중지")
        print("q: 종료")
        print("\n먼저 90을 입력해 정지 상태의 떨림을 확인하세요.")

        while True:
            command = input("\n입력 > ").strip().lower()

            if command == "q":
                break
            elif command == "c":
                servo.center()
            elif command == "h":
                servo.hold()
            elif command == "s":
                servo.release()
            elif command:
                try:
                    servo.set_angle(command)
                except ValueError as error:
                    print(f"[입력 오류] {error}")

    except (KeyboardInterrupt, EOFError):
        print("\n종료 요청")

    except (RuntimeError, pigpio.error, OSError) as error:
        print(f"\n[오류] {error}")

    finally:
        if servo is not None:
            servo.close()


if __name__ == "__main__":
    main()
