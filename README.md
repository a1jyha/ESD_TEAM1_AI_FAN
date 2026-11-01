# ESD_TEAM1_AI_FAN
Raspberry Pi-based AI Personal Fan Tracking System

# 프로젝트 개요
본 프로젝트는 Raspberry Pi 4와 YOLO11n 및 YuNet 모델을 활용하여, 카메라로부터 촬영된 영상을 바탕으로 사용자의 위치를 검출하고 서보모터를 연동해 송풍 방향을 자동으로 조절하는 개인용 선풍기 시스템을 구현합니다. 이때 매 프레임 검출 방식과 주기적 검출·중간 프레임 추적 방식을 비교 평가하여, 제한된 저사양 하드웨어 환경에서 최적의 대상 추적 성능과 시스템 자원 효율성을 검증합니다.

# 전체 구조
<img width="1626" height="967" alt="image" src="https://github.com/user-attachments/assets/5a0bc812-772f-42d9-913d-f63b481f4154" />
