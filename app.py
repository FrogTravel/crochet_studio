import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import io
import matplotlib.patches as patches
import matplotlib.transforms as transforms

# 1. Configuration: Abbreviations and Colors
CLASS_CONFIG = {
    "treble":      {"abbr": "tr", "color": "#00FF00"}, 
    "double":      {"abbr": "dc", "color": "#FF00FF"}, 
    "half_double": {"abbr": "hd", "color": "#00FFFF"}, 
    "chain":       {"abbr": "ch", "color": "#0000FF"}, 
    "single":      {"abbr": "sc", "color": "#FFD700"}, 
    "slip_stitch": {"abbr": "sl", "color": "#FF8000"}, 
    "fan":         {"abbr": "fa", "color": "#FF0000"},
}

# --- 1. Load Model ---
# Use st.cache_resource so it doesn't reload the model on every click
@st.cache_resource
def load_crochet_model():
    return YOLO('runs/obb/train5/weights/best.pt') # Ensure your best.pt is in the same folder

model = load_crochet_model()

# --- 2. Reuse your SVG Logic ---
def get_reconstruction(results, class_config):
    res = results[0]
    h, w = res.orig_img.shape[:2]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('white')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')

    if len(res.obb) > 0:
        for box in res.obb:
            corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2)
            cls_id = int(box.cls[0])
            full_name = res.names[cls_id]
            
            config = class_config.get(full_name, {"color": "#808080"})
            
            # Draw your logic here (Ellipse for chain, stems for treble, etc.)
            # [Insert your existing draw_svg_icon function here]
            DEFAULT_COLOR = "#808080"

            # 2. SVG Icon Drawing Logic (FIXED rotate_deg_around)
            def draw_svg_icon(ax, label, x, y, w, h, angle_deg, color):
                """Draws geometric crochet symbols using Matplotlib."""
                # Create the transform: Rotate around the center (x, y) then link to plot axes
                t = transforms.Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData
                
                if label == "chain":
                    # Chain is an oval
                    icon = patches.Ellipse((x, y), w*0.8, h*0.4, fill=False, color=color, linewidth=2, transform=t)
                    ax.add_patch(icon)
                
                elif label in ["half_double", "double", "treble", "fan"]:
                    # Vertical Stem
                    ax.plot([x, x], [y-h/2, y+h/2], color=color, lw=2, transform=t)
                    # Top Horizontal T-bar
                    ax.plot([x-w/3, x+w/3], [y-h/2, y-h/2], color=color, lw=2, transform=t)
                    
                    if label == "double":
                        # One angled bar
                        ax.plot([x-w/4, x+w/4], [y-h/8, y+h/8], color=color, lw=1.5, transform=t)
                    elif label == "treble":
                        # Two angled bars
                        ax.plot([x-w/4, x+w/4], [y-h/4, y-h/12], color=color, lw=1.5, transform=t)
                        ax.plot([x-w/4, x+w/4], [y+h/12, y+h/4], color=color, lw=1.5, transform=t)
                    elif label == "fan":
                        # Extra angled stems to represent a fan
                        ax.plot([x, x-w/3], [y+h/2, y-h/2], color=color, lw=1.5, transform=t, alpha=0.6)
                        ax.plot([x, x+w/3], [y+h/2, y-h/2], color=color, lw=1.5, transform=t, alpha=0.6)

                else: 
                    # Default 'X' for single crochet or unknown
                    ax.plot([x-w/3, x+w/3], [y-h/3, y+h/3], color=color, lw=2, transform=t)
                    ax.plot([x-w/3, x+w/3], [y+h/3, y-h/3], color=color, lw=2, transform=t)

            # 3. Main Execution
            trained_model = model
            results = trained_model.predict(res.orig_img, conf=0.25)
            res = results[0]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Ax1: Detection Boxes
            img_rgb = cv.cvtColor(res.orig_img, cv.COLOR_BGR2RGB)
            ax1.imshow(img_rgb)
            ax1.set_title("YOLO OBB Detections")

            # Ax2: Clean SVG Reconstruction
            ax2.set_facecolor('white')
            ax2.set_xlim(0, res.orig_img.shape[1])
            ax2.set_ylim(res.orig_img.shape[0], 0) # Maintain image coordinate system
            ax2.set_title("Reconstructed SVG Scheme")

            if len(res.obb) > 0:
                for box in res.obb:
                    # xyxyxyxy shape is (1, 4, 2)
                    corners = box.xyxyxyxy.cpu().numpy().reshape(4, 2)
                    cls_id = int(box.cls[0])
                    full_name = res.names[cls_id]
                    
                    config = CLASS_CONFIG.get(full_name, {"abbr": "??", "color": DEFAULT_COLOR})
                    color = config["color"]
                    abbr = config["abbr"]

                    # --- Ax1 Drawing ---
                    closed_corners = np.vstack([corners, corners[0]])
                    ax1.plot(closed_corners[:, 0], closed_corners[:, 1], color=color, linewidth=2)
                    ax1.text(corners[0, 0], corners[0, 1] - 5, abbr, color='white', fontsize=8, 
                            fontweight='bold', bbox=dict(facecolor=color, edgecolor='none', alpha=0.7))

                    # --- Ax2 Drawing Logic ---
                    center = corners.mean(axis=0)
                    width = np.linalg.norm(corners[0] - corners[1])
                    height = np.linalg.norm(corners[0] - corners[3])
                    
                    # Calculate angle (the angle of the top/bottom edge)
                    angle = np.degrees(np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0]))
                    
                    draw_svg_icon(ax2, full_name, center[0], center[1], width, height, angle, color)

            ax1.axis('off')
            ax2.axis('off')
            plt.tight_layout()
            plt.show()
            
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return Image.open(buf)

# --- 3. Streamlit UI ---
st.set_page_config(layout="wide", page_title="Crochet Scheme Generator")

st.title("🧶 Crochet Symbol to Scheme")
st.write("Upload a photo of a crochet chart to get a clean digital reconstruction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, channels="BGR", use_container_width=True)
        
    if st.button("Generate Scheme"):
        with st.spinner('Analyzing stitches...'):
            # Run Inference
            results = model.predict(image, conf=0.25)
            
            # Generate the SVG-style reconstruction
            # (Assuming you use your CLASS_CONFIG from earlier)
            reconstructed_img = get_reconstruction(results, CLASS_CONFIG)
            
            with col2:
                st.subheader("Digital Scheme")
                st.image(reconstructed_img, use_container_width=True)
                
                # Add a download button for the result
                buf = io.BytesIO()
                reconstructed_img.save(buf, format="PNG")
                st.download_button(
                    label="Download Scheme",
                    data=buf.getvalue(),
                    file_name="crochet_scheme.png",
                    mime="image/png"
                )