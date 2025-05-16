import json
from PIL import Image
import numpy as np
import argparse

# --- 모델 특정 파라미터 (config.pbtxt 및 모델 특성에 따라 수정) ---
MODEL_INPUT_NAME = "data_0"  # config.pbtxt의 input name과 일치
MODEL_OUTPUT_NAME = "fc6_1" # config.pbtxt의 output name과 일치

MODEL_EXPECTED_H = 224
MODEL_EXPECTED_W = 224
MODEL_EXPECTED_CHANNELS = 3
MODEL_DATATYPE = "FP32"  # config.pbtxt의 input data_type과 일치 (TYPE_FP32 -> FP32)

# 일반적인 ImageNet 정규화 값 (모델 학습 시 사용된 값에 맞춰야 함)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# --- 모델 특정 파라미터 끝 ---

def preprocess_image(image_path):
    """
    이미지를 로드하고, 모델 입력에 맞게 전처리합니다.
    config.pbtxt의 input format: FORMAT_NCHW, dims: [3, 224, 224] 에 맞춥니다.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None

    # 모델 입력 크기로 리사이즈
    img_resized = img.resize((MODEL_EXPECTED_W, MODEL_EXPECTED_H))
    # Numpy 배열로 변환 및 0-1 스케일링
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    # 정규화
    img_np = (img_np - MEAN) / STD
    # NCHW 형식으로 변경 (Channel, Height, Width)
    # 현재 img_np shape: (MODEL_EXPECTED_H, MODEL_EXPECTED_W, MODEL_EXPECTED_CHANNELS)
    # 변경 후 img_np_chw shape: (MODEL_EXPECTED_CHANNELS, MODEL_EXPECTED_H, MODEL_EXPECTED_W)
    img_np_chw = img_np.transpose((2, 0, 1))

    # config.pbtxt의 input dims는 [3, 224, 224]로, 배치 차원이 없습니다.
    # 따라서 페이로드의 shape도 [3, 224, 224]로 맞춰줍니다.
    # flatten()을 통해 1차원 리스트로 만듭니다.
    return img_np_chw.flatten().tolist(), list(img_np_chw.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_file_path", help="Path to the image file for preprocessing.")
    parser.add_argument(
        "--payload_file",
        type=str,
        default="payload.json",
        help="Filename for the output JSON payload.",
    )
    args = parser.parse_args()

    preprocessed_data_list, actual_shape_for_payload = preprocess_image(args.image_file_path)

    if preprocessed_data_list is None:
        exit(1)

    # KServe V2 API 표준에 따른 요청 페이로드 구성
    triton_request_payload = {
        "inputs": [
            {
                "name": MODEL_INPUT_NAME,
                "shape": actual_shape_for_payload,  # 예: [3, 224, 224]
                "datatype": MODEL_DATATYPE,
                "data": preprocessed_data_list,     # 1차원으로 펼쳐진 데이터
            }
        ],
        "outputs": [{"name": MODEL_OUTPUT_NAME}], # 요청할 출력 텐서 지정
    }

    with open(args.payload_file, "w") as f:
        json.dump(triton_request_payload, f)

    print(f"Preprocessed data and created JSON payload: {args.payload_file}")
    print(f"Input tensor name for the request: {MODEL_INPUT_NAME}")
    print(f"Input tensor shape for the request (in {args.payload_file}): {actual_shape_for_payload}")
    print(f"Output tensor name requested: {MODEL_OUTPUT_NAME}")