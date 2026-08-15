import os
import torch

# 1. Define the path to your saved weights
FINETUNED_MODEL_PATH = "finetuned_densenet_glaucoma.pth"

print(f"[STEP 1/3] Loading FINE-TUNED weights from '{FINETUNED_MODEL_PATH}'...", flush=True)

# 2. Safely load the weights and lock the model for inference
if os.path.exists(FINETUNED_MODEL_PATH):
    # map_location ensures it loads safely regardless of CPU/GPU availability
    densenet_model.load_state_dict(torch.load(FINETUNED_MODEL_PATH, map_location=Config.DEVICE))
    
    # .eval() is critical: it shuts off Dropout and freezes BatchNorm layers
    densenet_model.eval()
else:
    raise FileNotFoundError(f"Could not find checkpoint: {FINETUNED_MODEL_PATH}. Did Cell 11 finish?")
#The Architecture: You must instantiate your densenet_model = GlaucomaDenseNet() before running this block, otherwise PyTorch has no "skeleton" to load the weights into.
#The Device: Ensure Config.DEVICE is defined beforehand (e.g., DEVICE = "cuda" if torch.cuda.is_available() else "cpu").