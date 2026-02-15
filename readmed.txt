===============================================================
Car License Plate Detection & Recognition Project
===============================================================

Overview:
---------
This project provides a complete end-to-end pipeline for 
detecting and recognizing car license plates using 
PaddleOCR (PPOCR) versions v5 and v4. 

It includes:
- Dataset preparation
- Training scripts
- Inference scripts
- Colab notebooks for easy execution
- Exported models for inference

===============================================================
Folder Structure:
-----------------
PPOCR_License_Plate_Project/
│
├── README.txt                       <-- This file
├── train.py                         <-- Python training script
├── inference.py                     <-- Python inference script
├── train.txt                         <-- Training dataset label file
├── val.txt                           <-- Validation dataset label file
├── configs/
│   └── rec/
│       ├── en_PP-OCRv5_mobile_rec.yml
│       └── en_PP-OCRv4_mobile_rec.yml
└── notebooks/
    ├── PPOCR_License_Plate_Training.ipynb
    └── PPOCR_License_Plate_Inference.ipynb

===============================================================
Requirements:
-------------
- Python 3.7+
- PaddleOCR 3.x
- PaddlePaddle (GPU recommended)
- OpenCV
- NumPy
- Matplotlib

Install dependencies using pip:

    pip install paddleocr paddlepaddle opencv-python numpy matplotlib

===============================================================
Dataset Format:
---------------
- Label files must be in the format:

    image_path<TAB>label_text

Example:

    plate_dataset/images/plate_0001.png    U123AB4567
    plate_dataset/images/plate_0002.png    X987YZ4321

- Minimum dataset: 100 images
- Split dataset: 80% training, 20% validation

===============================================================
Training Instructions:
---------------------
1. Make sure dataset images are in the correct folder:
   
       ./train_data/plate_dataset/images/

2. Prepare train.txt and val.txt with format above.
3. Run training script:

       python3 train.py

- Supports PPOCR v5 & v4 training
- Saves trained models in the inference/ folder
- Monitors training metrics (loss, accuracy)

===============================================================
Inference Instructions:
-----------------------
1. Place test images in any folder, e.g., ./test_images/
2. Update inference.py paths if needed
3. Run inference script:

       python3 inference.py

- Generates annotated images with bounding boxes
- Recognized text and confidence scores
- Batch processing supported

===============================================================
Notebooks:
----------
1. PPOCR_License_Plate_Training.ipynb
   - End-to-end training pipeline
   - Visualization of loss curves and metrics

2. PPOCR_License_Plate_Inference.ipynb
   - Loads trained models
   - Runs inference on sample images
   - Annotates images with predictions

===============================================================
References:
-----------
- PaddleOCR Official Documentation: 
  https://www.paddleocr.ai/main/en/index.html

===============================================================
License:
--------
MIT License
