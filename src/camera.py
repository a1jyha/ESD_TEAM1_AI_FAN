# 카메라 초기화, 해상도 설정, 프레임 읽기, 연결 오류 처리를 한곳에서 관리
from picamera2 import Picamera2
import cv2


picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480),"format":"RGB888"})
picam2.configure(config)
picam2.start()

while True:
    frame = picam2.capture_array()  # 기본 출력 포맷 사용
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
