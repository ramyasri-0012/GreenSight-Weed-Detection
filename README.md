# GreenSight Weed Detection

## Real Python model

The `python_backend` folder contains the actual Python training and inference
pipeline requested for this project. Start with
[python_backend/README.md](python_backend/README.md). It uses Ultralytics YOLO,
expects labelled YOLO-format field images, trains `best.pt`, and exposes a
FastAPI `/predict` endpoint.

A beginner-friendly academic website prototype for the CSE471 Deep Learning for Computer Vision syllabus.

## Requirements
- Node.js 18+ (Node.js 20+ recommended)
- npm

## Run locally
1. Extract the ZIP.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   npm install
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```

   The Vite watcher ignores the Python virtual environment, trained model outputs,
   and dataset folders so it can run alongside the Python backend on Windows.
5. Open the local URL printed by Vite (usually `http://localhost:5173`).

Do not use the VS Code **Live Server / Go Live** button for this project. It
serves the uncompiled `src/main.tsx` file and produces a blank page. Use the
Vite command above instead:

```powershell
npm run dev
```

If port 5173 is busy, Vite will print another available URL in the terminal.

To create a production build:
```bash
npm run build
```

## Included features
- Detector-focused upload, preprocessing preview, and result layout.
- CSS/SVG crop-field illustration that works without external image assets.
- Image upload and preview.
- Visual preprocessing filters: original, resized-preview, grayscale and normalized-preview.
- Client-side demo analysis with loading state, simulated detection boxes and populated result cards.
- Laser and herbicide treatment-selection simulations.
- Responsive light green/white design.

## Important limitation
This is a frontend-only learning prototype. The demo boxes are illustrative, not predictions from a trained model. The supplied project did not include a labelled dataset or trained weights, so
the browser demo is still simulation-only until you train the Python model.
There is no real-time camera feed, laser, sprayer, or farm equipment connection.
Do not use the interface to make real treatment decisions.
