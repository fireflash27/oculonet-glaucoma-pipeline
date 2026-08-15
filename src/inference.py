import os
import sys

# System path setup MUST be before local imports so Python finds the 'models' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local imports
from models.transunet import TransUNet
from models.densenet import GlaucomaDenseNet
from src.extract_roi import extract_roi

def predict_glaucoma(image_path, transunet_path, densenet_path, threshold=0.50):
    """
    Runs a single fundus image through the dual-model pipeline.
    """
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Initializing pipeline on: {DEVICE.upper()}")
    
    # -----------------------------------------
    # 1. LOAD MODELS
    # -----------------------------------------
    print("[*] Building model architectures...")
    transunet = TransUNet().to(DEVICE)
    densenet = GlaucomaDenseNet().to(DEVICE)
    
    print("[*] Loading trained weights...")
    if not os.path.exists(transunet_path):
        raise FileNotFoundError(f"Missing TransUNet weights at: {transunet_path}")
    if not os.path.exists(densenet_path):
        raise FileNotFoundError(f"Missing DenseNet weights at: {densenet_path}")

    # Load weights safely regardless of GPU/CPU availability
    transunet.load_state_dict(torch.load(transunet_path, map_location=DEVICE))
    densenet.load_state_dict(torch.load(densenet_path, map_location=DEVICE))
    
    # Lock models into inference mode (freezes BatchNorm and Dropout)
    transunet.eval()
    densenet.eval()
    
    # -----------------------------------------
    # 2. IMAGE PREPARATION & ROI EXTRACTION
    # -----------------------------------------
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Eye image not found at: {image_path}")
        
    print(f"[*] Reading image: {os.path.basename(image_path)}")
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    print("[*] Phase 1: Locating Optic Disc (TransUNet)...")
    # Dynamically crop the optic nerve head using your custom function
    cropped_roi = extract_roi(image_rgb, transunet, DEVICE)
    
    # -----------------------------------------
    # 3. GLAUCOMA CLASSIFICATION
    # -----------------------------------------
    # Standard transform for the cropped ROI before classification
    cls_transform = A.Compose([
        A.Normalize(), 
        ToTensorV2()
    ])
    
    tensor_roi = cls_transform(image=cropped_roi)["image"].unsqueeze(0).to(DEVICE)
    
    print("[*] Phase 2: Analyzing nerve damage (DenseNet)...")
    with torch.no_grad():
        logits = densenet(tensor_roi)
        probability = torch.sigmoid(logits).item()
        
    # -----------------------------------------
# -----------------------------------------
    # 4. CLINICAL DECISION
    # -----------------------------------------
    # CORRECTED: Probability > threshold means Class 1 (Glaucoma)
    diagnosis = "Glaucoma Detected (1)" if probability >= threshold else "Healthy (0)"
    
    print("\n" + "="*50)
    print(" CLINICAL DIAGNOSIS REPORT")
    print("="*50)
    print(f"File             : {os.path.basename(image_path)}")
    print(f"Glaucoma Prob.   : {probability:.4f} ({(probability*100):.1f}%)")
    print(f"Active Threshold : {threshold:.4f}")
    print(f"Final Diagnosis  : {diagnosis}")
    print("="*50 + "\n")
    
    return probability, diagnosis
if __name__ == "__main__":
    # -----------------------------------------
    # CONFIGURATION 
    # -----------------------------------------
    # Drop an image named test_eye.jpg into your main Glaucoma VS folder
    SAMPLE_IMAGE = "test_eye.png" 
    
    # Paths to the weight files located in your root directory
    TRANSUNET_WEIGHTS = "models/transunetsegmentationmodel(1).pth"
    DENSENET_WEIGHTS = "models/finetuned_densenet_glaucoma.pth"
    # The threshold you discovered in Cell 14 that guarantees >= 90% Recall
    CALIBRATED_THRESHOLD = 0.42  
    
    # Run the pipeline
    try:
        predict_glaucoma(
            image_path=SAMPLE_IMAGE, 
            transunet_path=TRANSUNET_WEIGHTS, 
            densenet_path=DENSENET_WEIGHTS, 
            threshold=CALIBRATED_THRESHOLD
        )
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")