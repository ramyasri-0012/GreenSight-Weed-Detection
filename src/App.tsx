import { useEffect, useMemo, useState } from "react";
import {
  Sprout, ScanSearch, Upload, ShieldCheck,
  ArrowRight, CircleHelp, Eye, CheckCircle2,
  Menu, X
} from "lucide-react";

type View = "original" | "resized" | "grayscale" | "normalized";
type Detection = { confidence: number; class_name: string };
type Analysis = {
  count: number;
  detections: Detection[];
  annotated_image: string;
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

function FieldIllustration({ overlay = false }: { overlay?: boolean }) {
  return <svg viewBox="0 0 760 430" role="img" aria-label="Illustration of green crop rows and weeds in a field" className="field-svg">
    <defs>
      <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#dff3cf"/><stop offset="1" stopColor="#f5f4d7"/></linearGradient>
      <linearGradient id="soil" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#c5a46b"/><stop offset="1" stopColor="#967448"/></linearGradient>
    </defs>
    <rect width="760" height="430" rx="24" fill="url(#sky)"/>
    <path d="M0 150 Q170 100 320 150 T760 135 V430 H0Z" fill="#a4c47a"/>
    <path d="M0 230 Q200 175 380 225 T760 200 V430 H0Z" fill="url(#soil)"/>
    {Array.from({length: 9}, (_, i) => <g key={i} transform={`translate(${i*98-20},${160 + i*15})`}>
      <path d="M0 270 Q25 100 50 -20" fill="none" stroke="#7e9e42" strokeWidth="8" strokeLinecap="round"/>
      {[0,1,2,3,4].map((j) => <g key={j} transform={`translate(${j%2===0?-18:18},${j*48+25}) rotate(${j%2===0?-35:35})`}>
        <ellipse rx="16" ry="8" fill={j%2===0 ? "#4c8d3a" : "#6aa84f"}/>
      </g>)}
    </g>)}
    {Array.from({length: 13}, (_, i) => <g key={i} transform={`translate(${35+i*57},${280+(i%3)*35})`}>
      <path d="M0 0 Q-5 -22 -12 -29 M0 0 Q5 -23 15 -27 M0 0 L0 -33" stroke="#356d35" strokeWidth="4" fill="none" strokeLinecap="round"/>
      <ellipse cx="-13" cy="-29" rx="10" ry="5" transform="rotate(28 -13 -29)" fill="#72b84b"/>
      <ellipse cx="15" cy="-27" rx="10" ry="5" transform="rotate(-25 15 -27)" fill="#4d9a42"/>
      <ellipse cx="0" cy="-34" rx="7" ry="12" fill="#4d8c3c"/>
    </g>)}
    {overlay && <>
      <rect x="92" y="244" width="94" height="110" rx="8" fill="none" stroke="#f8fbf6" strokeWidth="4" strokeDasharray="8 4"/>
      <rect x="92" y="217" width="115" height="27" rx="6" fill="#f8fbf6"/>
      <text x="102" y="236" fill="#23663b" fontSize="15" fontWeight="700">Crop · demo</text>
      <rect x="485" y="276" width="91" height="83" rx="8" fill="none" stroke="#ffdf74" strokeWidth="4" strokeDasharray="8 4"/>
      <rect x="485" y="249" width="114" height="27" rx="6" fill="#ffdf74"/>
      <text x="495" y="268" fill="#634b12" fontSize="15" fontWeight="700">Weed · demo</text>
    </>}
  </svg>;
}

function App() {
  const [view, setView] = useState<View>("original");
  const [overlay, setOverlay] = useState(false);
  const [upload, setUpload] = useState<string | null>(null);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState("");
  const [mobileMenu, setMobileMenu] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const isUpload = Boolean(upload);

  useEffect(() => () => {
    if (upload) URL.revokeObjectURL(upload);
  }, [upload]);
  const visualFilter = useMemo(() => {
    if (view === "grayscale") return "grayscale(1)";
    if (view === "normalized") return "contrast(1.15) saturate(1.1)";
    if (view === "resized") return "contrast(.96)";
    return "none";
  }, [view]);

  function handleUpload(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      alert("Please choose an image file such as PNG, JPG or WEBP.");
      return;
    }
    if (upload) URL.revokeObjectURL(upload);
    setUpload(URL.createObjectURL(file));
    setUploadFile(file);
    setUploadName(file.name);
    setOverlay(false);
    setAnalysis(null);
    setError("");
  }

  function resetSample() {
    if (upload) URL.revokeObjectURL(upload);
    setUpload(null);
    setUploadFile(null);
    setUploadName("");
    setOverlay(false);
    setAnalysis(null);
    setError("");
    setView("original");
  }

  async function runAnalysis() {
    if (!uploadFile) {
      setError("Upload an image before running the trained model.");
      return;
    }
    setIsAnalyzing(true);
    setError("");
    const formData = new FormData();
    formData.append("image", uploadFile);
    const startedAt = performance.now();
    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      const body = await response.json() as { detail?: string } & Partial<Analysis>;
      if (!response.ok) {
        throw new Error(body.detail ?? `Model API returned HTTP ${response.status}.`);
      }
      if (!body.annotated_image || !Array.isArray(body.detections) || typeof body.count !== "number") {
        throw new Error("The model API returned an invalid prediction response.");
      }
      setAnalysis({
        count: body.count,
        detections: body.detections,
        annotated_image: body.annotated_image,
      });
      setOverlay(true);
    } catch (reason) {
      setAnalysis(null);
      const message = reason instanceof Error ? reason.message : "The model request failed.";
      setError(message === "Failed to fetch"
        ? "Cannot connect to the Python API. Start python_backend/start_api.ps1, then try again."
        : message);
    } finally {
      setIsAnalyzing(false);
      const elapsed = performance.now() - startedAt;
      if (elapsed < 250) await new Promise(resolve => window.setTimeout(resolve, 250 - elapsed));
    }
  }

  const navItems = ["Detector", "Results"];
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
    setMobileMenu(false);
  };

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#home" aria-label="GreenSight home"><span className="brand-mark"><Sprout size={23}/></span><span>GreenSight<span className="brand-dot">.</span><small>COMPUTER VISION FOR CROPS</small></span></a>
      <nav className={mobileMenu ? "nav-links open" : "nav-links"}>
        {navItems.map(item => <button key={item} onClick={() => scrollTo(item === "Detector" ? "demo" : "results")}>{item}</button>)}
      </nav>
      <button className="nav-cta" onClick={() => scrollTo("demo")}>Try the demo <ArrowRight size={16}/></button>
      <button className="mobile-toggle" onClick={() => setMobileMenu(!mobileMenu)} aria-label="Toggle navigation">{mobileMenu ? <X/> : <Menu/>}</button>
    </header>

    <main>
      <section className="hero section-wrap" id="home">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-dot"/> AI WEED SEGMENTATION</div>
          <h1>Detect weeds.<br/><span>Protect the crops.</span></h1>
          <p className="hero-subtitle">Upload a field image and run the trained YOLO segmentation model to locate weeds with confidence scores and pixel masks.</p>
          <div className="hero-actions"><button className="button-primary" onClick={() => scrollTo("demo")}>Try weed detection <ArrowRight size={18}/></button><button className="button-secondary" onClick={() => scrollTo("how")}><CircleHelp size={17}/> How it works</button></div>
          <div className="hero-note"><ShieldCheck size={18}/> Real model inference · Results depend on your trained weights</div>
        </div>
        <div className="hero-visual"><div className="visual-label"><span className="live-dot"/> FIELD IMAGE · DEMO</div><FieldIllustration overlay/><div className="float-card"><span className="float-icon"><ScanSearch size={20}/></span><span><b>Visual detection demo</b><small>Illustrative boxes, not model predictions</small></span></div></div>
      </section>

      <section className="demo-section" id="demo">
        <div className="section-wrap">
          <div className="section-heading"><div className="eyebrow">LIVE MODEL INFERENCE</div><h2>Upload an image and <span>detect weeds.</span></h2><p>Send your field image to the Python YOLO segmentation API and view its actual masks and confidence scores.</p></div>
          <div className="demo-layout">
            <div className="demo-image-card">
              <div className="demo-card-header"><div><span className="status-pill"><span className="live-dot"/> DEMO MODE</span><h3>Field image preview</h3></div><button className="text-button" onClick={resetSample}>Reset sample</button></div>
              <div className="image-stage">
                {upload ? <img className="uploaded-image" src={analysis ? `${API_URL}/outputs/${analysis.annotated_image}` : upload} alt={`Uploaded field image ${uploadName}`} style={{filter: visualFilter}}/> :
                  <div className="field-image-wrap" style={{filter: visualFilter}}><FieldIllustration overlay={overlay}/>{overlay && <div className="overlay-note">Illustrative detections only</div>}</div>}
                {isUpload && <span className="uploaded-tag">{analysis ? "MODEL OUTPUT" : "YOUR IMAGE"}</span>}
              </div>
              <div className="preprocess-toolbar"><span className="toolbar-label">PREPROCESSING VIEW</span><div className="view-buttons">{(["original","resized","grayscale","normalized"] as View[]).map(v=><button key={v} className={view===v?"view-button active":"view-button"} onClick={()=>setView(v)}>{v[0].toUpperCase()+v.slice(1)}</button>)}</div></div>
              <div className="image-actions">
                <label className="button-secondary upload-button"><Upload size={17}/> Upload your image<input type="file" accept="image/*" onChange={handleUpload}/></label>
                <button className="button-primary" onClick={runAnalysis} disabled={isAnalyzing}>{isAnalyzing ? "Analysing…" : analysis ? "Run again" : "Analyse image"} <ScanSearch size={17}/></button>
                {!isUpload && <button className="button-secondary" onClick={()=>setOverlay(!overlay)}><Eye size={17}/>{overlay?"Hide demo boxes":"Show demo boxes"}</button>}
              </div>
              {error && <p className="error-message" role="alert">{error}</p>}
              <p className="fine-print">{analysis ? `${analysis.count} weed instance(s) returned by the Python model.` : isUpload ? `Previewing ${uploadName}. Click Analyse image to run the trained model.` : "Upload an image to begin."}</p>
            </div>
            <aside className="demo-controls">
              <div className="safety-card"><ShieldCheck size={21}/><div><b>Real inference mode</b><p>The production Python app in <code>python_backend</code> runs YOLO segmentation with your trained <code>best.pt</code> weights.</p></div></div>
              <div className="model-status"><span className="status-circle"/><div><b>Model status</b><p>Connect the Python API to replace this browser preview with real predictions.</p></div></div>
            </aside>
          </div>
        </div>
      </section>

      <section className="section-wrap section-block" id="results">
        <div className="section-heading"><div className="eyebrow">MODEL OUTPUT</div><h2>Detection <span>results.</span></h2><p>Predictions from the connected YOLO segmentation model appear here.</p></div>
        <div className="metric-grid">
          <article className="metric-card"><span className="metric-icon"><ScanSearch/></span><small>Weed instances</small><strong>{analysis?.count ?? "—"}</strong><p>{analysis ? "Detected regions" : "No image analysed"}</p></article>
          <article className="metric-card"><span className="metric-icon"><Eye/></span><small>Confidence score</small><strong>{analysis ? `${Math.round(Math.max(...analysis.detections.map(item => item.confidence), 0) * 100)}%` : "—"}</strong><p>{analysis ? "Best prediction" : "No model connected"}</p></article>
          <article className="metric-card"><span className="metric-icon"><CheckCircle2/></span><small>Detected class</small><strong>{analysis?.detections[0]?.class_name ?? "—"}</strong><p>{analysis ? "From model output" : "Not measured"}</p></article>
        </div>
        <div className="notice-box"><CircleHelp/><p><b>{analysis ? "Inference complete" : "No results yet"}</b> {analysis ? "The output above represents the current model prediction." : "Upload an image and run inference to populate the results."}</p></div>
      </section>
    </main>

    <footer className="footer"><div className="footer-brand"><span className="brand-mark"><Sprout size={20}/></span><span><b>GreenSight.</b><small>Real-time weed segmentation</small></span></div><p>YOLO model inference</p><a href="#home">Back to top ↑</a></footer>
  </div>;
}

export default App;
