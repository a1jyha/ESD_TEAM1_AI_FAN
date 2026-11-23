import glob
import os
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

# ===== 본인 모델과 폴더에 맞게 수정할 부분 =====
INPUT_MODEL = "yolo11n_320.onnx"         # 원본 FP32 ONNX 파일명
OUTPUT_MODEL = "yolo11n_320_int8.onnx"   # 새로 만들 INT8 ONNX 파일명
CALIB_DIR = "calib_images"               # 보정용 사진(.jpg, .png 등)이 들어있는 폴더명
IMG_W = 320                              # 모델 입력 가로 크기 (예: 320 또는 160)
IMG_H = 320                              # 모델 입력 세로 크기 (예: 320 또는 120)
SCALE_TO_01 = True                       # YOLO11n: True (0~1 정규화), YuNet: False (0~255 유지)
MAX_IMAGES = 100                         # 보정에 사용할 최대 사진 수 (50~150장 권장)
# ==============================================


class FolderCalibrationReader(CalibrationDataReader):

  def __init__(self, image_dir, input_name, img_w, img_h, scale_to_01, max_images):
    self.input_name = input_name
    self.img_w = img_w
    self.img_h = img_h
    self.scale_to_01 = scale_to_01

    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"):
      paths.extend(glob.glob(os.path.join(image_dir, ext)))
    self.paths = sorted(set(paths))[:max_images]

    if not self.paths:
      raise RuntimeError(f"'{image_dir}' 폴더에서 이미지를 찾지 못했습니다.")

    self._iter = iter(self._gen())

  def _gen(self):
    for p in self.paths:
      bgr = cv2.imread(p)
      if bgr is None:
        continue

      if self.scale_to_01:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
      else:
        resized = cv2.resize(bgr, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        blob = resized.transpose(2, 0, 1).astype(np.float32)

      yield {self.input_name: np.expand_dims(blob, axis=0)}

  def get_next(self):
    return next(self._iter, None)


def main():
  if not os.path.exists(INPUT_MODEL):
    raise FileNotFoundError(f"입력 모델 파일이 없습니다: {INPUT_MODEL}")

  model = onnx.load(INPUT_MODEL)
  input_name = model.graph.input[0].name
  print(f"모델 로드 완료: {INPUT_MODEL} (입력 이름: {input_name})")
  print(f"보정 이미지 폴더 '{CALIB_DIR}'에서 양자화 시작...")

  reader = FolderCalibrationReader(
      image_dir=CALIB_DIR,
      input_name=input_name,
      img_w=IMG_W,
      img_h=IMG_H,
      scale_to_01=SCALE_TO_01,
      max_images=MAX_IMAGES,
  )

  quantize_static(
      model_input=INPUT_MODEL,
      model_output=OUTPUT_MODEL,
      calibration_data_reader=reader,
      quant_format=QuantFormat.QDQ,
      activation_type=QuantType.QUInt8,
      weight_type=QuantType.QInt8,
  )

  print(f"양자화 완료! 저장된 파일: {OUTPUT_MODEL}")


if __name__ == "__main__":
  main()
