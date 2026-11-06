from picamera2 import Picamera2
import cv2


def main():
    picam2 = Picamera2()
    started = False

    try:
        config = picam2.create_preview_configuration(
            main={
                "size": (640, 480),
                "format": "RGB888",
            }
        )

        picam2.configure(config)
        picam2.start()
        started = True

        print("영상 창을 클릭한 뒤 q를 누르면 종료합니다.")

        while True:
            frame = picam2.capture_array()
            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n카메라 시험을 종료합니다.")

    finally:
        try:
            if started:
                picam2.stop()
        finally:
            try:
                picam2.close()
            finally:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
