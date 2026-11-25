# ESD_TEAM1_AI_FAN
Raspberry Pi-based AI Personal Fan Tracking System

# 프로젝트 개요
본 프로젝트는 Raspberry Pi 4와 YOLO11n 및 YuNet 모델을 활용하여, 카메라로부터 촬영된 영상을 바탕으로 사용자의 위치를 검출하고 서보모터를 연동해 송풍 방향을 자동으로 조절하는 개인용 선풍기 시스템을 구현합니다. 이때 매 프레임 검출 방식과 주기적 검출·중간 프레임 추적 방식을 비교 평가하여, 제한된 저사양 하드웨어 환경에서 최적의 대상 추적 성능과 시스템 자원 효율성을 검증합니다.

# 전체 구조
<img width="1626" height="967" alt="image" src="https://github.com/user-attachments/assets/5a0bc812-772f-42d9-913d-f63b481f4154" />

# 서보모터 제어 및 실행 방법
서보모터 정지 상태에서 발생하는 미세 떨림과 지터를 줄이기 위해 소프트웨어 PWM 대신 하드웨어 타이밍 기반의 `pigpiod`을 사용합니다.
### 1. 환경 활성화 및 작업 디렉터리 이동
가상환경(`esd_env`)을 활성화한 뒤 프로젝트 폴더로 이동합니다. 필수는 아니지만, 의존성 충돌 방지와 재현성 확보를 위해 가상환경에서 실행하였습니다.
### 2. 실행 중인 pigpiod 프로세스 확인
```bash
pgrep -a pigpiod
### 3. pigpio 데몬 시작
```bash
sudo pigpiod -l
