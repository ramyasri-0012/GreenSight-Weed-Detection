# GreenSight Python model

This is the real Python/Ultralytics pipeline. It replaces the React-only simulated
results with a trainable YOLO instance-segmentation model once labelled data and
weights are supplied.

## 1. Create the dataset

Use YOLO segmentation labels:

```text
dataset/
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg                 # optional
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt                 # optional
```

Each label file contains one object polygon per line:

```text
class_id x1 y1 x2 y2 ... xn yn
```

All polygon coordinates must be normalized between 0 and 1. The included
`data.yaml` defines one class, `weed`. If your dataset has more classes, update
`names`.

## 2. Install

From `python_backend`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Train

```powershell
python train.py --data data.yaml --model yolo11n-seg.pt --epochs 100 --imgsz 640 --batch 16
```

The pretrained segmentation checkpoint is downloaded by Ultralytics on first use. The best
trained weights are written to `runs/weed_detector/weights/best.pt`.
Use `--device 0` for an NVIDIA GPU or `--device cpu` for CPU training.

## 4. Run segmentation inference

```powershell
python predict.py --model runs/weed_detector/weights/best.pt --image .\sample.jpg
```

This writes an annotated image and JSON detections plus mask polygons to `outputs/`.

## 5. Start the Python UI

```powershell
$env:GREENSIGHT_MODEL = "runs/weed_detector/weights/best.pt"
python -m streamlit run app.py
```

Upload an image, choose a confidence threshold, and download the real
segmentation JSON from the browser UI.

## 6. Start the API

```powershell
$env:GREENSIGHT_MODEL = "runs/weed_detector/weights/best.pt"
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Or, on Windows, use the checked launcher:

```powershell
.\start_api.ps1
```

Keep this terminal open while using the React interface. If the browser says
“Cannot connect to the Python API”, this process is not running on port 8000.

Then upload an image to `POST http://127.0.0.1:8000/predict`. Check
`GET http://127.0.0.1:8000/health` before connecting the frontend.

## 7. Connect the React frontend

Run the API from `python_backend`:

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In another terminal, run the React app from the project root:

```powershell
npm run dev
```

The React **Analyse image** button now sends the uploaded file to the API. It
does not generate fixed demo boxes or fixed confidence values. If the API or
`best.pt` is missing, the UI shows the actual error instead of fake results.

## Important

No trained weights or labelled dataset existed in the supplied project, so this
repository cannot claim a trained accuracy yet. Training metrics are only valid
after running this pipeline on your actual field dataset and reviewing the
validation results.
