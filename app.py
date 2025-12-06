# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
import numpy as np
import cv2
import base64
import torchvision.transforms as transforms
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# Import your model definition
from general_ad import General_AD 

app = FastAPI()

# --- CONFIGURATION ---
MODEL_PATH =  "general_ad-epoch=15-train_loss=0.82.ckpt" # Or your specific path
THRESHOLD = 0.7  # Replace with the optimal threshold you found
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
print("Loading model...")
try:
    model = General_AD.load_from_checkpoint(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- HELPER FUNCTIONS ---
def tensor_to_base64(image_np):
    """Converts a numpy image (RGB) to base64 string"""
    img = Image.fromarray(image_np)
    buff = io.BytesIO()
    img.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return img_str

def get_heatmap_and_bbox(model, img_tensor, original_image):
    """Generates Heatmap and Bounding Box visualization"""
    
    # 1. Inference
    with torch.no_grad():
        patch_scores = model(img_tensor.to(DEVICE))

    # 2. Reshape & Interpolate to get Heatmap
    n_patches = int(np.sqrt(patch_scores.shape[1]))
    grid_scores = patch_scores.view(1, 1, n_patches, n_patches)
    
    heatmap_tensor = F.interpolate(
        grid_scores, size=(518, 518), mode='bilinear', align_corners=False
    )
    heatmap_np = heatmap_tensor.squeeze().cpu().numpy()
    
    # Smooth the heatmap
    heatmap_smooth = gaussian_filter(heatmap_np, sigma=4)
    
    # 3. Create Visualization Images
    # Convert original to numpy (resize to 518x518 to match heatmap)
    orig_np = np.array(original_image.resize((518, 518)))
    
    # A. Heatmap Image (Colorized)
    # Normalize 0-1 to 0-255
    heatmap_norm = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min() + 1e-8)
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # OpenCV uses BGR
    
    # B. Overlay
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)
    
    # C. Bounding Box
    mask = (heatmap_smooth > THRESHOLD).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_img = orig_np.copy()
    
    crack_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < 50: continue # Filter noise
        crack_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (255, 0, 0), 3) # Red Box
        cv2.putText(bbox_img, "CRACK", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    return heatmap_color, overlay, bbox_img, crack_detected

# --- API ENDPOINT ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read Image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0) # Add batch dim
    
    # Generate Visualizations
    heatmap, overlay, bbox_img, is_crack = get_heatmap_and_bbox(model, img_tensor, image)
    
    # Convert images to Base64 to send via JSON
    return JSONResponse(content={
        "prediction": "Anomaly Detected" if is_crack else "Normal",
        "heatmap_b64": tensor_to_base64(heatmap),
        "overlay_b64": tensor_to_base64(overlay),
        "bbox_b64": tensor_to_base64(bbox_img)
    })
