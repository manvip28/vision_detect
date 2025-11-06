import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# YOLOv8 (pretrained on COCO 80 classes)
from ultralytics import YOLO

st.set_page_config(page_title="Real-Time Object Detector", page_icon="🧠", layout="wide")

# -----------------------------
# Sidebar: Educational content
# -----------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("Model and detection settings")
    conf = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    iou = st.slider("NMS IoU threshold", 0.1, 0.9, 0.5, 0.05)

    st.markdown("---")
    st.subheader("Camera settings")
    cam_index = st.number_input("Camera index", min_value=0, value=0, step=1)
    resolution = st.selectbox(
        "Resolution",
        options=["640x480", "1280x720", "1920x1080"],
        index=0,
    )

    st.caption("If the camera fails to open, try another index or a lower resolution.")

# -----------------------------
# Cache model (download once)
# -----------------------------
@st.cache_resource
def load_model():
    # ultralytics will auto-download yolov8s.pt on first use
    model = YOLO("yolov8s.pt")
    model.overrides["imgsz"] = 640
    return model

model = load_model()

# -----------------------------
# Helpers
# -----------------------------
def set_resolution(cap, res_str):
    w, h = map(int, res_str.split("x"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

def draw_detections(frame_bgr, results):
    """
    Draw bounding boxes and labels from YOLO results on the BGR frame.
    """
    if not results or len(results) == 0:
        return frame_bgr

    res = results[0]
    if res.boxes is None:
        return frame_bgr

    boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else []
    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else []
    clss  = res.boxes.cls.cpu().numpy().astype(int) if hasattr(res.boxes, "cls") else []

    names = res.names if hasattr(res, "names") else model.model.names

    for (x1, y1, x2, y2), c, cls_id in zip(boxes, confs, clss):
        label = f"{names.get(cls_id, str(cls_id))} {c*100:.1f}%"
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # rectangle
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 160, 255), 2)

        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 160, 255), -1)
        # text
        cv2.putText(frame_bgr, label, (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    return frame_bgr

# -----------------------------
# Start/Stop controls
# -----------------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "stop" not in st.session_state:
    st.session_state.stop = False

def start():
    st.session_state.run = True
    st.session_state.stop = False

def stop():
    st.session_state.stop = True
    st.session_state.run = False

controls_col1, controls_col2, controls_col3 = st.columns([1,1,2])
with controls_col1:
    st.button("Start", on_click=start, use_container_width=True)
with controls_col2:
    st.button("Stop", on_click=stop, use_container_width=True)

# -----------------------------
# Main video panel and stats
# -----------------------------
video_col, stats_col = st.columns([3, 1])

with video_col:
    st.markdown("### Live camera")
    frame_placeholder = st.empty()

with stats_col:
    st.markdown("### Stats")
    fps_text = st.empty()
    count_text = st.empty()

# -----------------------------
# Run loop when active
# -----------------------------
if st.session_state.run and not st.session_state.stop:
    cap = cv2.VideoCapture(int(cam_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error("Could not open camera. Try another index.")
        stop()
    else:
        set_resolution(cap, resolution)
        prev = time.time()
        try:
            while st.session_state.run and not st.session_state.stop:
                ok, frame_bgr = cap.read()
                if not ok:
                    st.warning("No frame received from camera.")
                    break

                # Inference
                # stream=False returns a list with one result
                results = model.predict(
                    source=frame_bgr,
                    conf=conf,
                    iou=iou,
                    imgsz=640,       # keep camera resolution
                    verbose=False,
                    device="cpu"
                )

                # Drawing
                frame_bgr = draw_detections(frame_bgr, results)

                # FPS
                now = time.time()
                fps = 1.0 / max(1e-6, (now - prev))
                prev = now

                # Object count in this frame
                try:
                    obj_count = int(len(results[0].boxes))
                except Exception:
                    obj_count = 0

                # Show
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB",use_container_width=True)

                fps_text.markdown(f"FPS: {fps:.1f}")
                count_text.markdown(f"Objects detected: {obj_count}")

                # Small sleep to reduce CPU usage
                time.sleep(0.01)

        finally:
            cap.release()
            stop()
