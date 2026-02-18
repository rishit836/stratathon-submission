from flask import Flask, render_template, request, url_for
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder relative to the app root
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure directories exist
os.makedirs(os.path.join(app.root_path, UPLOAD_FOLDER), exist_ok=True)
os.makedirs(os.path.join(app.root_path, RESULTS_FOLDER), exist_ok=True)

# --- Configuration & Model Definition ---
IMG_H = 532  # Must match training
IMG_W = 952
EMBED_DIM = 384
PATCH_H = IMG_H // 14
PATCH_W = IMG_W // 14
N_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Color palette (same as training)
COLOR_PALETTE = np.array([
    [0, 0, 0],        [34, 139, 34],   [0, 255, 0],     [210, 180, 140],
    [139, 90, 43],    [128, 128, 0],   [139, 69, 19],   [128, 128, 128],
    [160, 82, 45],    [135, 206, 235],
], dtype=np.uint8)

CLASS_NAMES = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
               'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ConvBNReLU(in_ch, out_ch, 1, 0))
        for r in rates:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_ch, out_ch, 1, 0),
        )
        self.project = ConvBNReLU(out_ch * (len(rates) + 2), out_ch, 1, 0)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [conv(x) for conv in self.convs]
        feats.append(F.interpolate(self.pool(x), size=(h, w), mode='bilinear', align_corners=False))
        return self.dropout(self.project(torch.cat(feats, dim=1)))

class EnhancedSegHead(nn.Module):
    def __init__(self, embed_dim, n_classes, patch_h, patch_w):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.project = ConvBNReLU(embed_dim, 256, 1, 0)
        self.aspp = ASPP(256, 256, rates=(6, 12, 18))
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 128),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBNReLU(64, 64),
            ConvBNReLU(64, 64),
        )
        self.classifier = nn.Sequential(
            ConvBNReLU(64, 64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, n_classes, 1),
        )

    def forward(self, patch_tokens):
        B, N, C = patch_tokens.shape
        x = patch_tokens.reshape(B, self.patch_h, self.patch_w, C).permute(0, 3, 1, 2)
        x = self.project(x)
        x = self.aspp(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.classifier(x)
        return x

# Global model variables
backbone = None
seg_head = None

def load_models():
    global backbone, seg_head
    if backbone is not None and seg_head is not None:
        return

    print("Loading models...")
    # Paths - adjust as needed based on where you run the flask app from
    # Assuming running from 'frontend/' folder
    
    BACKBONE_PATH = os.path.join("model/", "best_finetuned_backbone.pth")
    HEAD_PATH = os.path.join("model/", "best_segmentation_head.pth")

    if not os.path.exists(BACKBONE_PATH):
        print(f"Error: Backbone weights not found at {BACKBONE_PATH}")
    if not os.path.exists(HEAD_PATH):
        print(f"Error: Head weights not found at {HEAD_PATH}")

    # Load Backbone
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    # Clean state dict keys if needed (sometimes saved as module.xxx)
    state_dict = torch.load(BACKBONE_PATH, map_location=DEVICE)
    backbone.load_state_dict(state_dict)
    backbone.to(DEVICE)
    backbone.eval()

    # Load Head
    seg_head = EnhancedSegHead(EMBED_DIM, N_CLASSES, PATCH_H, PATCH_W)
    seg_head.load_state_dict(torch.load(HEAD_PATH, map_location=DEVICE))
    seg_head.to(DEVICE)
    seg_head.eval()
    print("Models loaded successfully!")

def mask_to_color(mask_np):
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(N_CLASSES):
        rgb[mask_np == c] = COLOR_PALETTE[c]
    return rgb

def predict_segmentation(image_path):
    """
    Loads model if needed, predicts segmentation, saves result image, returns result path.
    """
    # Ensure models are loaded
    try:
        load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")
        return image_path # Fallback

    # Read image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return image_path
    
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess
    img_resized = cv2.resize(img_rgb, (IMG_W, IMG_H))
    # Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        feats = backbone.forward_features(img_t)["x_norm_patchtokens"]
        logits = seg_head(feats)
        # Upsample to model input size first
        pred = F.interpolate(logits, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)
        pred = pred.argmax(dim=1).squeeze().cpu().numpy()

    # Resize mask (nearest) back to original size
    pred_orig = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Colorize
    color_mask = mask_to_color(pred_orig)
    
    # Save result
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    result_filename = f"{name}_seg.png"
    result_path = os.path.join(app.root_path, app.config['RESULTS_FOLDER'], result_filename)
    
    # Convert RGB back to BGR for cv2 saving
    cv2.imwrite(result_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))

    return url_for('static', filename=f'results/{result_filename}')

@app.route('/', methods=['GET', 'POST'])
def home():
    input_image = None
    segmented_image = None
    
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['image']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file')
            
        if file:
            filename = secure_filename(file.filename)
            # Save file
            file_path = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Get relative path for template
            input_image = url_for('static', filename=f'uploads/{filename}')
            
            # Run prediction
            segmented_image = predict_segmentation(file_path)
            
    # Prepare legend data
    colors_css = [f"rgb({c[0]},{c[1]},{c[2]})" for c in COLOR_PALETTE]
    # Pass as list so it can be iterated multiple times if needed and is JSON serializable if we were using it in JS
    legend_items = list(zip(CLASS_NAMES, colors_css))

    return render_template('index.html', input_image=input_image, segmented_image=segmented_image, legend_items=legend_items)

if __name__ == '__main__':
    app.run(debug=True, port=8000)