import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def extract_roi(image_rgb, model, device, margin=0.25, seg_img_size=224, roi_img_size=224):
    """Dynamically crops the fundus image around the TransUNet-predicted Optic Disc."""
    h, w, _ = image_rgb.shape
    
    # Transform specifically for TransUNet
    seg_transform = A.Compose([
        A.Resize(seg_img_size, seg_img_size),
        A.Normalize(),
        ToTensorV2()
    ])
    
    tensor_img = seg_transform(image=image_rgb)["image"].unsqueeze(0).to(device)
    
    # Ensure TransUNet is in eval mode for inference
    model.eval()
    with torch.no_grad():
        prob_od = torch.sigmoid(model(tensor_img))[0, 0].cpu().numpy()
    
    mask = cv2.resize((prob_od > 0.5).astype(np.uint8), (w, h))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
        px, py = int(cw * margin), int(ch * margin)
        cropped = image_rgb[max(0, y-py):min(h, y+ch+py), max(0, x-px):min(w, x+cw+px)]
    else:
        # Fallback to center crop if network fails to find disc
        cs = min(h, w) // 2
        cropped = image_rgb[h//2-cs//2:h//2+cs//2, w//2-cs//2:w//2+cs//2]
        
    return cv2.resize(cropped, (roi_img_size, roi_img_size))