"""Streamlit interface for the trained GreenSight weed segmentation model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from predict import predict

MODEL_PATH = Path(os.getenv("GREENSIGHT_MODEL", "runs/weed_detector/weights/best.pt"))
OUTPUT_DIR = Path(os.getenv("GREENSIGHT_OUTPUT_DIR", "outputs"))

st.set_page_config(page_title="GreenSight Weed Detection", page_icon="🌱", layout="wide")
st.title("GreenSight Weed Detection")
st.caption("Python + YOLO instance segmentation")

if not MODEL_PATH.is_file():
    st.error(
        f"Trained weights were not found at `{MODEL_PATH}`. "
        "Train the model with `python train.py` before using this app."
    )
    st.stop()

uploaded = st.file_uploader("Upload a field image", type=["jpg", "jpeg", "png", "webp"])
confidence = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25)

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Input image", use_container_width=True)
    if st.button("Run weed segmentation", type="primary"):
        temporary_input = OUTPUT_DIR / f"_streamlit_{uploaded.name}"
        temporary_input.parent.mkdir(parents=True, exist_ok=True)
        temporary_input.write_bytes(uploaded.getvalue())
        try:
            with st.spinner("Running YOLO inference..."):
                result = predict(MODEL_PATH, temporary_input, OUTPUT_DIR, confidence)
            annotated_path = OUTPUT_DIR / result["annotated_image"]
            columns = st.columns(3)
            columns[0].metric("Weed instances", result["count"])
            columns[1].metric(
                "Best confidence",
                f"{max((item['confidence'] for item in result['detections']), default=0):.1%}",
            )
            columns[2].metric("Model", MODEL_PATH.name)
            st.image(annotated_path, caption="Segmentation result", use_container_width=True)
            st.json(result)
            st.download_button(
                "Download result JSON",
                data=json.dumps(result, indent=2),
                file_name="greensight_result.json",
                mime="application/json",
            )
        finally:
            temporary_input.unlink(missing_ok=True)
