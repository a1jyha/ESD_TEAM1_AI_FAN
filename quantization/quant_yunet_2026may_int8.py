import glob
import os
import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

BASE_DIR = "/mnt/c/ESD_calib"
SRC_MODEL = os.path.join(BASE_DIR, "face_detection_yunet_2026may.onnx")
DST_MODEL = os.path.join(BASE_DIR, "face_detection_yunet_2026may_int8.onnx")
CALIB_DIR = os.path.join(BASE_DIR, "calibimages")
IMG_SIZE = 320  # 필요시 192 등으로 변경 가능
MAX_IMGS = 100


class YuNetDynamicCalibReader(CalibrationDataReader):
    def __init__(self, img_dir: str, input_name: str, size: int = 320, max_imgs: int = 100):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(img_dir, ext)))
        paths = sorted(paths)[:max_imgs]

        if not paths:
            raise RuntimeError(f"이미지를 찾지 못했습니다: {img_dir}")

        self.samples = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[WARN] 건너뜀: {p}")
                continue
            resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            # YuNet ONNX는 보통 0~255 float32 NCHW 입력을 사용
            chw = np.transpose(resized.astype(np.float32), (2, 0, 1))[None, ...].astype(np.float32)
            self.samples.append({input_name: chw})

        if not self.samples:
            raise RuntimeError("유효한 캘리브레이션 이미지가 0장입니다.")

        self.idx = 0
        print(f"[INFO] 캘리브레이션 이미지 {len(self.samples)}장 로드 완료 ({size}x{size})")

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        item = self.samples[self.idx]
        self.idx += 1
        return item

    def rewind(self):
        self.idx = 0


def main():
    sess = ort.InferenceSession(SRC_MODEL, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"[INFO] 입력 이름: {input_name} | 캘리브레이션 해상도: {IMG_SIZE}x{IMG_SIZE}")

    reader = YuNetDynamicCalibReader(
        img_dir=CALIB_DIR,
        input_name=input_name,
        size=IMG_SIZE,
        max_imgs=MAX_IMGS,
    )

    quantize_static(
        model_input=SRC_MODEL,
        model_output=DST_MODEL,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"[DONE] 저장 완료: {DST_MODEL}")


if __name__ == "__main__":
    main()