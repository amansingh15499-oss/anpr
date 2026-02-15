import os
from paddleocr import PaddleOCR, draw_ocr
import cv2

# =========================
# SETTINGS
# =========================

# Path to test images folder
TEST_IMAGE_DIR = "./test_images"
OUTPUT_DIR = "./inference_results"

# Choose models: v5 and v4
OCR_V5_MODEL = "en_PP-OCRv5_mobile"  # or server version if available
OCR_V4_MODEL = "en_PP-OCRv4_mobile"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# INIT OCR MODELS
# =========================

print("Initializing PPOCR v5 model...")
ocr_v5 = PaddleOCR(use_angle_cls=True, lang='en', rec_model_dir=None, det_model_dir=None,
                   rec_char_type='en', use_gpu=True, rec_model_name=OCR_V5_MODEL)

print("Initializing PPOCR v4 model...")
ocr_v4 = PaddleOCR(use_angle_cls=True, lang='en', rec_model_dir=None, det_model_dir=None,
                   rec_char_type='en', use_gpu=True, rec_model_name=OCR_V4_MODEL)

# =========================
# RUN INFERENCE
# =========================

test_images = [f for f in os.listdir(TEST_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not test_images:
    raise Exception(f"No images found in {TEST_IMAGE_DIR}")

for img_name in test_images:
    img_path = os.path.join(TEST_IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    print(f"\nProcessing {img_name}...")

    # PPOCR v5 Inference
    result_v5 = ocr_v5.ocr(img_path, cls=True)
    v5_texts = [line[1][0] for line in result_v5[0]]
    v5_scores = [line[1][1] for line in result_v5[0]]

    # PPOCR v4 Inference
    result_v4 = ocr_v4.ocr(img_path, cls=True)
    v4_texts = [line[1][0] for line in result_v4[0]]
    v4_scores = [line[1][1] for line in result_v4[0]]

    # Annotate image for v5
    annotated_image_v5 = draw_ocr(image, result_v5[0], font_path='')  # Add font_path if needed
    annotated_image_v5 = cv2.cvtColor(annotated_image_v5, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_v5.png"), annotated_image_v5)

    # Annotate image for v4
    annotated_image_v4 = draw_ocr(image, result_v4[0], font_path='')  # Add font_path if needed
    annotated_image_v4 = cv2.cvtColor(annotated_image_v4, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_v4.png"), annotated_image_v4)

    print("PPOCR v5 Results:", v5_texts, v5_scores)
    print("PPOCR v4 Results:", v4_texts, v4_scores)

print("\n✅ Inference completed. Annotated images saved in:", OUTPUT_DIR)
