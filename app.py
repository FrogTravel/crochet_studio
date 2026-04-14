import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import io
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from util.tiler import predict_tiled

# 1. Configuration: Abbreviations and Colors (matches data/project_yolo_obb/data.yaml)
CLASS_CONFIG = {
    "chain":          {"abbr": "ch",  "color": "#0000FF"},
    "double":         {"abbr": "dc",  "color": "#FF00FF"},
    "double treble":  {"abbr": "dtr", "color": "#00AA00"},
    "enseble_chain":  {"abbr": "ec",  "color": "#FF8000"},
    "fan":            {"abbr": "fa",  "color": "#FF0000"},
    "half_double":    {"abbr": "hd",  "color": "#00FFFF"},
    "noise":          {"abbr": "no",  "color": "#808080"},
    "single":         {"abbr": "sc",  "color": "#FFD700"},
    "treble":         {"abbr": "tr",  "color": "#00FF00"},
}

# --- 1. Load Model ---
# Use st.cache_resource so it doesn't reload the model on every click
@st.cache_resource
def load_crochet_model():
    return YOLO('runs/obb/train6/weights/best.pt') # Ensure your best.pt is in the same folder

model = load_crochet_model()

DEFAULT_COLOR = "#808080"

def draw_svg_icon(ax, label, x, y, w, h, angle_deg, color):
    """Draws geometric crochet symbols using Matplotlib."""
    t = transforms.Affine2D().rotate_deg_around(x, y, angle_deg) + ax.transData

    if label == "chain":
        icon = patches.Ellipse((x, y), w*0.8, h*0.4, fill=False, color=color, linewidth=2, transform=t)
        ax.add_patch(icon)
    elif label in ["half_double", "double", "double treble", "treble", "fan"]:
        ax.plot([x, x], [y-h/2, y+h/2], color=color, lw=2, transform=t)
        ax.plot([x-w/3, x+w/3], [y-h/2, y-h/2], color=color, lw=2, transform=t)
        if label == "double":
            ax.plot([x-w/4, x+w/4], [y-h/8, y+h/8], color=color, lw=1.5, transform=t)
        elif label == "treble":
            ax.plot([x-w/4, x+w/4], [y-h/4, y-h/12], color=color, lw=1.5, transform=t)
            ax.plot([x-w/4, x+w/4], [y+h/12, y+h/4], color=color, lw=1.5, transform=t)
        elif label == "double treble":
            ax.plot([x-w/4, x+w/4], [y-h/3, y-h/6], color=color, lw=1.5, transform=t)
            ax.plot([x-w/4, x+w/4], [y-h/12, y+h/12], color=color, lw=1.5, transform=t)
            ax.plot([x-w/4, x+w/4], [y+h/6, y+h/3], color=color, lw=1.5, transform=t)
        elif label == "fan":
            ax.plot([x, x-w/3], [y+h/2, y-h/2], color=color, lw=1.5, transform=t, alpha=0.6)
            ax.plot([x, x+w/3], [y+h/2, y-h/2], color=color, lw=1.5, transform=t, alpha=0.6)
    elif label == "enseble_chain":
        n_ovals = max(2, int(h / (w * 0.5)))
        oval_h = h / n_ovals
        for ci in range(n_ovals):
            cy = y - h/2 + oval_h * (ci + 0.5)
            ax.add_patch(patches.Ellipse((x, cy), w*0.6, oval_h*0.7, fill=False,
                                         color=color, linewidth=1.5, transform=t))
    elif label == "noise":
        ax.add_patch(patches.Circle((x, y), min(w, h)*0.3, fill=False,
                                    color=color, linewidth=1.5, transform=t))
    elif label == "single":
        ax.plot([x-w/3, x+w/3], [y-h/3, y+h/3], color=color, lw=2, transform=t)
        ax.plot([x-w/3, x+w/3], [y+h/3, y-h/3], color=color, lw=2, transform=t)
    else:
        ax.plot([x-w/3, x+w/3], [y-h/3, y+h/3], color=color, lw=2, transform=t)
        ax.plot([x-w/3, x+w/3], [y+h/3, y-h/3], color=color, lw=2, transform=t)


# --- 2. Reconstruction from tiled detections ---
def get_reconstruction(image: np.ndarray, detections, class_config):
    h, w = image.shape[:2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    ax1.set_title("YOLO OBB Detections")

    ax2.set_facecolor('white')
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.set_title("Reconstructed SVG Scheme")

    for det in detections:
        corners = det.corners
        full_name = det.cls_name
        config = class_config.get(full_name, {"abbr": "??", "color": DEFAULT_COLOR})
        color = config["color"]
        abbr = config["abbr"]

        closed_corners = np.vstack([corners, corners[0]])
        ax1.plot(closed_corners[:, 0], closed_corners[:, 1], color=color, linewidth=2)
        ax1.text(corners[0, 0], corners[0, 1] - 5, abbr, color='white', fontsize=8,
                fontweight='bold', bbox=dict(facecolor=color, edgecolor='none', alpha=0.7))

        center = corners.mean(axis=0)
        width = np.linalg.norm(corners[0] - corners[1])
        height = np.linalg.norm(corners[0] - corners[3])
        angle = np.degrees(np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0]))
        draw_svg_icon(ax2, full_name, center[0], center[1], width, height, angle, color)

    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
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
            detections = predict_tiled(model, image, tile_size=640, overlap=0.2, conf=0.25)
            reconstructed_img = get_reconstruction(image, detections, CLASS_CONFIG)
            
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