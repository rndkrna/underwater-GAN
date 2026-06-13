import React, { useState, useEffect, useRef } from "react";
import {
  Layers,
  UploadCloud,
  Trash2,
  Image as ImageIcon,
  LineChart as ChartIcon,
  Cpu,
  Info,
  Sliders,
  Download,
  Eye,
  RefreshCw,
  SlidersHorizontal,
  ChevronRight,
  TrendingUp,
  Compass,
  Zap,
} from "lucide-react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8080";

// --- CUSTOM SVG LINE CHART COMPONENT ---
// Renders high-fidelity, interactive, custom SVG line graphs with area gradients and hover tooltips.
function SVGLineChart({ data, keys, colors, labels, yLabel }) {
  const [hoveredIdx, setHoveredIdx] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  if (!data || data.length === 0) {
    return <div className="no-data-chart">No history data available.</div>;
  }

  // Chart dimensions
  const paddingLeft = 50;
  const paddingRight = 20;
  const paddingTop = 20;
  const paddingBottom = 30;
  const width = 500;
  const height = 240;

  const chartWidth = width - paddingLeft - paddingRight;
  const chartHeight = height - paddingTop - paddingBottom;

  // Find Min/Max for Y scaling
  let maxVal = -Infinity;
  let minVal = Infinity;

  data.forEach((d) => {
    keys.forEach((key) => {
      const val = d[key];
      if (val !== null && val !== undefined) {
        if (val > maxVal) maxVal = val;
        if (val < minVal) minVal = val;
      }
    });
  });

  // Adjust bounds for padding
  const valRange = maxVal - minVal;
  maxVal = valRange === 0 ? maxVal + 1 : maxVal + valRange * 0.1;
  minVal = valRange === 0 ? minVal - 1 : Math.max(0, minVal - valRange * 0.1);

  const getX = (index) => {
    return paddingLeft + (index / (data.length - 1)) * chartWidth;
  };

  const getY = (val) => {
    if (val === null || val === undefined) return paddingTop + chartHeight;
    const ratio = (val - minVal) / (maxVal - minVal);
    return paddingTop + chartHeight - ratio * chartHeight;
  };

  // Generate paths for each key
  const paths = keys.map((key) => {
    let dStr = "";
    data.forEach((d, idx) => {
      const x = getX(idx);
      const y = getY(d[key]);
      if (idx === 0) {
        dStr += `M ${x} ${y}`;
      } else {
        dStr += ` L ${x} ${y}`;
      }
    });
    return dStr;
  });

  // Generate Area paths (with gradient fills)
  const areaPaths = keys.map((key) => {
    if (data.length === 0) return "";
    let dStr = `M ${getX(0)} ${getY(minVal)}`;
    data.forEach((d, idx) => {
      dStr += ` L ${getX(idx)} ${getY(d[key])}`;
    });
    dStr += ` L ${getX(data.length - 1)} ${getY(minVal)} Z`;
    return dStr;
  });

  // Handle Mouse Hovering
  const handleMouseMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const clientX = e.clientX - rect.left;
    const clientY = e.clientY - rect.top;

    // Convert clientX to chart coordinates relative to viewport box
    const svgX = (clientX / rect.width) * width;
    const innerX = svgX - paddingLeft;
    
    // Find closest index
    const percentX = Math.max(0, Math.min(1, innerX / chartWidth));
    const closestIdx = Math.round(percentX * (data.length - 1));
    
    setHoveredIdx(closestIdx);
    setTooltipPos({
      x: (getX(closestIdx) / width) * rect.width,
      y: (getY(data[closestIdx][keys[0]]) / height) * rect.height,
    });
  };

  const handleMouseLeave = () => {
    setHoveredIdx(null);
  };

  // Render Horizontal Gridlines
  const yTicks = 4;
  const gridLines = [];
  for (let i = 0; i <= yTicks; i++) {
    const ratio = i / yTicks;
    const yVal = minVal + ratio * (maxVal - minVal);
    const yPixel = getY(yVal);
    gridLines.push(
      <g key={i}>
        <line
          x1={paddingLeft}
          y1={yPixel}
          x2={width - paddingRight}
          y2={yPixel}
          stroke="rgba(255, 255, 255, 0.05)"
          strokeDasharray="4,4"
        />
        <text
          x={paddingLeft - 8}
          y={yPixel + 4}
          fill="var(--text-muted)"
          fontSize="9px"
          textAnchor="end"
        >
          {yVal.toFixed(yVal > 10 ? 1 : 3)}
        </text>
      </g>
    );
  }

  // Render X-axis ticks (e.g. 5 ticks across epochs)
  const xTicksCount = Math.min(5, data.length);
  const xTicks = [];
  for (let i = 0; i < xTicksCount; i++) {
    const idx = Math.round((i / (xTicksCount - 1)) * (data.length - 1));
    const xPixel = getX(idx);
    xTicks.push(
      <g key={i}>
        <line
          x1={xPixel}
          y1={height - paddingBottom}
          x2={xPixel}
          y2={height - paddingBottom + 4}
          stroke="rgba(255, 255, 255, 0.1)"
        />
        <text
          x={xPixel}
          y={height - paddingBottom + 16}
          fill="var(--text-muted)"
          fontSize="9px"
          textAnchor="middle"
        >
          Ep {data[idx].epoch}
        </text>
      </g>
    );
  }

  return (
    <div 
      className="svg-chart-container" 
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" height="100%">
        <defs>
          {keys.map((key, i) => (
            <linearGradient key={key} id={`grad-${key}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={colors[i]} stopOpacity="0.25" />
              <stop offset="100%" stopColor={colors[i]} stopOpacity="0.0" />
            </linearGradient>
          ))}
        </defs>

        {/* Gridlines */}
        {gridLines}

        {/* X and Y axes */}
        <line
          x1={paddingLeft}
          y1={height - paddingBottom}
          x2={width - paddingRight}
          y2={height - paddingBottom}
          stroke="rgba(255, 255, 255, 0.15)"
        />
        <line
          x1={paddingLeft}
          y1={paddingTop}
          x2={paddingLeft}
          y2={height - paddingBottom}
          stroke="rgba(255, 255, 255, 0.15)"
        />

        {/* X Ticks */}
        {xTicks}

        {/* Area paths */}
        {areaPaths.map((path, i) => (
          <path
            key={`area-${keys[i]}`}
            d={path}
            fill={`url(#grad-${keys[i]})`}
            stroke="none"
          />
        ))}

        {/* Line paths */}
        {paths.map((path, i) => (
          <path
            key={`line-${keys[i]}`}
            d={path}
            fill="none"
            stroke={colors[i]}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ))}

        {/* Hover vertical bar & dots */}
        {hoveredIdx !== null && (
          <g>
            <line
              x1={getX(hoveredIdx)}
              y1={paddingTop}
              x2={getX(hoveredIdx)}
              y2={height - paddingBottom}
              stroke="rgba(0, 240, 255, 0.25)"
              strokeWidth="1.5"
              strokeDasharray="2,2"
            />
            {keys.map((key, i) => (
              <circle
                key={`dot-${key}`}
                cx={getX(hoveredIdx)}
                cy={getY(data[hoveredIdx][key])}
                r="4.5"
                fill={colors[i]}
                stroke="white"
                strokeWidth="1.5"
              />
            ))}
          </g>
        )}
      </svg>

      {/* Floating Tooltip */}
      {hoveredIdx !== null && (
        <div
          className="chart-tooltip-portal"
          style={{
            left: `${tooltipPos.x}px`,
            top: `${tooltipPos.y}px`,
          }}
        >
          <div className="tooltip-epoch">Epoch {data[hoveredIdx].epoch}</div>
          {keys.map((key, i) => (
            <div key={key} className="tooltip-val">
              <span style={{ color: colors[i], fontWeight: 600 }}>{labels[i]}:</span>
              <span>{data[hoveredIdx][key] !== null ? data[hoveredIdx][key].toFixed(4) : "N/A"}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// --- MAIN REACT APPLICATION ---
export default function App() {
  // Tabs & Modes
  const [activeTab, setActiveTab] = useState("restore");
  const [compareMode, setCompareMode] = useState("slider");
  const [sliderPosition, setSliderPosition] = useState(50);

  // Model selection states
  const [checkpointsList, setCheckpointsList] = useState([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState("");
  const [modelInfo, setModelInfo] = useState(null);

  // Image upload states
  const [inputImage, setInputImage] = useState(null);
  const [inputImagePreview, setInputImagePreview] = useState("");
  const [refImage, setRefImage] = useState(null);
  const [refImagePreview, setRefImagePreview] = useState("");

  const [inputDragActive, setInputDragActive] = useState(false);
  const [refDragActive, setRefDragActive] = useState(false);

  // Execution states
  const [loading, setLoading] = useState(false);
  const [restoredData, setRestoredData] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  // Refs for file inputs
  const inputRefFile = useRef(null);
  const refRefFile = useRef(null);

  // Fetch Checkpoints List & Model Info
  useEffect(() => {
    const fetchCheckpoints = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/checkpoints`);
        const data = await res.json();
        setCheckpointsList(data.checkpoints || []);
        if (data.checkpoints && data.checkpoints.length > 0) {
          // Select default
          setSelectedCheckpoint(data.checkpoints[0]);
        }
      } catch (err) {
        console.error("Failed to load checkpoints:", err);
      }
    };
    fetchCheckpoints();
  }, []);

  useEffect(() => {
    if (!selectedCheckpoint) return;
    const fetchModelInfo = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/model-info?checkpoint=${selectedCheckpoint}`);
        const data = await res.json();
        setModelInfo(data);
      } catch (err) {
        console.error("Failed to load model details:", err);
      }
    };
    fetchModelInfo();
  }, [selectedCheckpoint]);

  // Handle File Selections
  const handleFileChange = (e, target) => {
    const file = e.target.files[0];
    if (!file) return;
    processFile(file, target);
  };

  const processFile = (file, target) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (target === "input") {
        setInputImage(file);
        setInputImagePreview(reader.result);
      } else {
        setRefImage(file);
        setRefImagePreview(reader.result);
      }
    };
    reader.readAsDataURL(file);
  };

  // Drag and Drop Handling
  const handleDrag = (e, target, active) => {
    e.preventDefault();
    e.stopPropagation();
    if (target === "input") {
      setInputDragActive(active);
    } else {
      setRefDragActive(active);
    }
  };

  const handleDrop = (e, target) => {
    e.preventDefault();
    e.stopPropagation();
    if (target === "input") {
      setInputDragActive(false);
    } else {
      setRefDragActive(false);
    }

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0], target);
    }
  };

  const removeFile = (target) => {
    if (target === "input") {
      setInputImage(null);
      setInputImagePreview("");
    } else {
      setRefImage(null);
      setRefImagePreview("");
    }
  };

  // Perform Image Restoration
  const handleRestoreSubmit = async () => {
    if (!inputImage) return;

    setLoading(true);
    setErrorMsg("");
    setRestoredData(null);

    const formData = new FormData();
    formData.append("file", inputImage);
    if (refImage) {
      formData.append("ref_file", refImage);
    }
    formData.append("checkpoint", selectedCheckpoint);

    try {
      const res = await fetch(`${API_BASE}/api/restore`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Error restoring image");
      }

      const data = await res.json();
      setRestoredData(data);
    } catch (err) {
      console.error(err);
      setErrorMsg(err.message || "Something went wrong on the server.");
    } finally {
      setLoading(false);
    }
  };

  // Helper for UIqm progress coloring
  const getUiqmColor = (val) => {
    if (val < 1.0) return "#ef4444";
    if (val < 2.0) return "#f59e0b";
    return "#10b981";
  };

  return (
    <div className="app-container">
      {/* SIDEBAR / CONTROLS PANEL */}
      <aside className="sidebar">
        <div className="brand">
          <Compass className="brand-logo" size={32} />
          <h1 className="brand-name">DeepBlue</h1>
        </div>

        {/* Model Configurations */}
        <div className="panel">
          <h3 className="panel-title">
            <Cpu size={16} /> GAN Config
          </h3>
          <div className="form-group">
            <label>Model Checkpoint</label>
            <select
              className="select-input"
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
            >
              {checkpointsList.length > 0 ? (
                checkpointsList.map((cp) => (
                  <option key={cp} value={cp}>
                    {cp}
                  </option>
                ))
              ) : (
                <option value="">No checkpoints found</option>
              )}
            </select>
          </div>
        </div>

        {/* Input Image Upload Card */}
        <div className="panel">
          <h3 className="panel-title">
            <ImageIcon size={16} /> Input Image (Raw)
          </h3>
          
          {inputImagePreview ? (
            <div className="preview-thumbnail">
              <img src={inputImagePreview} alt="Underwater raw input" />
              <button className="remove-btn" onClick={() => removeFile("input")}>
                <Trash2 size={14} />
              </button>
            </div>
          ) : (
            <div
              className={`upload-zone ${inputDragActive ? "drag-active" : ""}`}
              onDragEnter={(e) => handleDrag(e, "input", true)}
              onDragOver={(e) => handleDrag(e, "input", true)}
              onDragLeave={(e) => handleDrag(e, "input", false)}
              onDrop={(e) => handleDrop(e, "input")}
              onClick={() => inputRefFile.current.click()}
            >
              <UploadCloud className="upload-icon" size={36} />
              <p className="upload-text">Drag & Drop or click</p>
              <p className="upload-subtext">Supports PNG, JPG, JPEG</p>
              <input
                ref={inputRefFile}
                type="file"
                style={{ display: "none" }}
                accept="image/*"
                onChange={(e) => handleFileChange(e, "input")}
              />
            </div>
          )}
        </div>

        {/* Reference Image Upload Card (Optional) */}
        <div className="panel">
          <h3 className="panel-title">
            <Eye size={16} /> Reference (Ground Truth)
          </h3>
          <p className="upload-subtext" style={{ marginTop: "-12px", marginBottom: "12px" }}>
            Optional. Used to compute PSNR and SSIM metrics.
          </p>

          {refImagePreview ? (
            <div className="preview-thumbnail">
              <img src={refImagePreview} alt="Reference target" />
              <button className="remove-btn" onClick={() => removeFile("reference")}>
                <Trash2 size={14} />
              </button>
            </div>
          ) : (
            <div
              className={`upload-zone ${refDragActive ? "drag-active" : ""}`}
              onDragEnter={(e) => handleDrag(e, "reference", true)}
              onDragOver={(e) => handleDrag(e, "reference", true)}
              onDragLeave={(e) => handleDrag(e, "reference", false)}
              onDrop={(e) => handleDrop(e, "reference")}
              onClick={() => refRefFile.current.click()}
            >
              <UploadCloud className="upload-icon" size={28} />
              <p className="upload-text">Upload reference image</p>
              <p className="upload-subtext">Will auto-detect from dataset if match found</p>
              <input
                ref={refRefFile}
                type="file"
                style={{ display: "none" }}
                accept="image/*"
                onChange={(e) => handleFileChange(e, "reference")}
              />
            </div>
          )}
        </div>

        {/* Action Button */}
        <button
          className="btn-primary"
          onClick={handleRestoreSubmit}
          disabled={loading || !inputImage}
        >
          {loading ? (
            <>
              <RefreshCw className="animate-spin" size={16} /> Processing...
            </>
          ) : (
            <>
              <Zap size={16} /> Restore Quality
            </>
          )}
        </button>

        {errorMsg && (
          <div className="panel" style={{ borderLeft: "4px solid #ef4444", background: "rgba(239,68,68,0.08)" }}>
            <span style={{ color: "#ef4444", fontSize: "13px", fontWeight: 600 }}>Error:</span>
            <p style={{ color: "#f8fafc", fontSize: "12px", marginTop: "4px" }}>{errorMsg}</p>
          </div>
        )}
      </aside>

      {/* MAIN SCREEN AREA */}
      <main className="main-content">
        {/* Navigation / Header Bar */}
        <header className="top-nav">
          <nav className="tabs">
            <button
              className={`tab ${activeTab === "restore" ? "active" : ""}`}
              onClick={() => setActiveTab("restore")}
            >
              <Sliders size={16} /> Restoration Playground
            </button>
            <button
              className={`tab ${activeTab === "training" ? "active" : ""}`}
              onClick={() => setActiveTab("training")}
            >
              <ChartIcon size={16} /> Training Curves
            </button>
          </nav>

          {modelInfo && (
            <div className="meta-badges">
              <div className="badge badge-device">
                <Cpu size={12} />
                <span>Running on: {modelInfo.device || "CPU"}</span>
              </div>
              <div className="badge badge-checkpoint">
                <Layers size={12} />
                <span>Model: {modelInfo.checkpoint || "Standard"}</span>
              </div>
            </div>
          )}
        </header>

        {/* TAB CONTENT: RESTORATION */}
        {activeTab === "restore" && (
          <div className="dashboard-grid">
            {/* LEFT AREA: Comparisons and histograms */}
            <div className="compare-container">
              {restoredData ? (
                <>
                  <div className="compare-header">
                    <h4 className="compare-title">Visual Comparison</h4>
                    <div className="btn-group">
                      <button
                        className={`btn-toggle ${compareMode === "slider" ? "active" : ""}`}
                        onClick={() => setCompareMode("slider")}
                      >
                        Slider Compare
                      </button>
                      <button
                        className={`btn-toggle ${compareMode === "side-by-side" ? "active" : ""}`}
                        onClick={() => setCompareMode("side-by-side")}
                      >
                        Side-by-Side
                      </button>
                    </div>
                  </div>

                  {compareMode === "slider" ? (
                    <div className="slider-wrapper">
                      {/* Before (Original) Image */}
                      <div className="image-before">
                        <img src={restoredData.original_image} alt="Original underwater" />
                        <span className="image-label before">Before</span>
                      </div>

                      {/* After (Restored) Image */}
                      <div
                        className="image-after"
                        style={{ clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)` }}
                      >
                        <img src={restoredData.restored_image} alt="Restored GAN output" />
                        <span className="image-label after">After (Restored)</span>
                      </div>

                      {/* Sliding Divider Bar */}
                      <div className="slider-line" style={{ left: `${sliderPosition}%` }} />
                      <div className="slider-button" style={{ left: `${sliderPosition}%` }}>
                        <SlidersHorizontal size={16} />
                      </div>

                      {/* Draggable transparent input overlay */}
                      <input
                        type="range"
                        className="compare-slider-input"
                        min="0"
                        max="100"
                        value={sliderPosition}
                        onChange={(e) => setSliderPosition(Number(e.target.value))}
                      />
                    </div>
                  ) : (
                    <div className="side-by-side-grid">
                      <div className="side-box">
                        <img src={restoredData.original_image} alt="Original raw" />
                        <span className="image-label before">Original Raw</span>
                      </div>
                      <div className="side-box">
                        <img src={restoredData.restored_image} alt="Restored GAN" />
                        <span className="image-label after">Restored Output</span>
                      </div>
                    </div>
                  )}

                  {/* RGB Histograms */}
                  <div className="panel histogram-container">
                    <h3 className="panel-title">
                      <TrendingUp size={16} /> RGB Color Distribution Analysis
                    </h3>
                    <div className="histogram-image-box">
                      <img src={restoredData.histogram} alt="RGB histograms comparison" />
                    </div>
                    <p className="histogram-desc">
                      The histograms show the distribution of RGB channels. Water quickly absorbs red wavelength light.
                      The GAN Generator model restores this lost red component, correcting the heavy green/blue tint to 
                      create a more balanced, natural spectrum (seen as matching rgb spreads on the restored side).
                    </p>
                  </div>
                </>
              ) : loading ? (
                <div className="slider-wrapper" style={{ position: "relative" }}>
                  <div className="loading-overlay">
                    <div className="sonar-wrapper">
                      <div className="sonar-wave"></div>
                      <div className="sonar-wave"></div>
                      <div className="sonar-wave"></div>
                      <div className="sonar-core"></div>
                    </div>
                    <div className="scanning-text">Restoring underwater scene</div>
                    <p style={{ fontSize: "12px", color: "var(--text-muted)", marginTop: "-10px" }}>
                      Processing through UNet Generator layers...
                    </p>
                  </div>
                </div>
              ) : (
                <div className="welcome-container">
                  <Compass className="welcome-icon" size={64} />
                  <h3 className="welcome-title">Underwater Image Restorator</h3>
                  <p className="welcome-desc">
                    Reconstruct and enhance degraded underwater photographs using a trained Generative Adversarial Network.
                    Instantly calculate color balances, sharpness, and standard reference metrics.
                  </p>
                  <div className="welcome-steps">
                    <div className="step-card">
                      <div className="step-number">1</div>
                      <h4>Select Checkpoint</h4>
                      <p>Select your model `.pth` parameters from the sidebar.</p>
                    </div>
                    <div className="step-card">
                      <div className="step-number">2</div>
                      <h4>Upload Image</h4>
                      <p>Drop an underwater photo. Upload ground truth if you have it.</p>
                    </div>
                    <div className="step-card">
                      <div className="step-number">3</div>
                      <h4>Click Restore</h4>
                      <p>Run inference on CPU/GPU and see immediate visual analysis.</p>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* RIGHT AREA: Metrics Cards and Downloader */}
            <div className="metrics-sidebar">
              {restoredData ? (
                <>
                  <div className="panel download-card">
                    <div className="download-text-box">
                      <h4>Restoration Complete</h4>
                      <p>Download the high-res restored PNG file</p>
                    </div>
                    <a
                      href={restoredData.restored_image}
                      download={`restored_${inputImage ? inputImage.name : "image.png"}`}
                    >
                      <button className="btn-secondary">
                        <Download size={14} /> Download PNG
                      </button>
                    </a>
                  </div>

                  <div className="panel">
                    <h3 className="panel-title">Restoration Analytics</h3>
                    
                    {/* UIQM Metric Card */}
                    <div className="metric-card">
                      <div className="metric-card-header">
                        <div className="metric-name-container">
                          <span className="metric-name">UIQM (Underwater Quality Index)</span>
                          <span className="info-tooltip-trigger" title="Underwater Image Quality Measure. Combines colorfulness, contrast, and sharpness. Higher values represent better visual quality.">
                            <Info size={13} />
                          </span>
                        </div>
                        <span className={`metric-badge-diff positive`}>
                          +{restoredData.metrics.uiqm_diff_percent.toFixed(1)}%
                        </span>
                      </div>
                      <div className="metric-value-row">
                        <span className="metric-main-value" style={{ color: getUiqmColor(restoredData.metrics.uiqm_after) }}>
                          {restoredData.metrics.uiqm_after.toFixed(3)}
                        </span>
                        <span className="metric-subtitle">Visual score</span>
                      </div>
                      
                      <div className="metric-comparison">
                        <div className="comparison-row">
                          <span>Original Input:</span>
                          <span>{restoredData.metrics.uiqm_before.toFixed(3)}</span>
                        </div>
                        <div className="bar-container">
                          <div className="bar-fill before" style={{ width: `${Math.min(100, (restoredData.metrics.uiqm_before / 4) * 100)}%` }} />
                        </div>
                        
                        <div className="comparison-row after">
                          <span>Restored Scene:</span>
                          <span>{restoredData.metrics.uiqm_after.toFixed(3)}</span>
                        </div>
                        <div className="bar-container">
                          <div className="bar-fill after" style={{ width: `${Math.min(100, (restoredData.metrics.uiqm_after / 4) * 100)}%` }} />
                        </div>
                      </div>
                    </div>

                    {/* PSNR & SSIM metrics (conditional on reference) */}
                    {restoredData.metrics.has_reference ? (
                      <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: "16px" }}>
                        {/* PSNR Card */}
                        <div className="metric-card">
                          <div className="metric-card-header">
                            <div className="metric-name-container">
                              <span className="metric-name">PSNR (Peak Signal-to-Noise Ratio)</span>
                              <span className="info-tooltip-trigger" title="Calculates log pixel error ratio compared to the clean Ground Truth. Higher is better (values > 25dB represent solid reconstruction).">
                                <Info size={13} />
                              </span>
                            </div>
                            {restoredData.metrics.psnr_after > restoredData.metrics.psnr_before && (
                              <span className="metric-badge-diff positive">
                                Improved
                              </span>
                            )}
                          </div>
                          <div className="metric-value-row">
                            <span className="metric-main-value" style={{ color: restoredData.metrics.psnr_after > 20 ? "#10b981" : "#f59e0b" }}>
                              {restoredData.metrics.psnr_after.toFixed(2)} dB
                            </span>
                            <span className="metric-subtitle">Error recovery</span>
                          </div>
                          <div className="metric-comparison" style={{ fontSize: "11.5px" }}>
                            <div className="comparison-row">
                              <span>Before:</span>
                              <span>{restoredData.metrics.psnr_before.toFixed(2)} dB</span>
                            </div>
                            <div className="comparison-row after">
                              <span>After:</span>
                              <span>{restoredData.metrics.psnr_after.toFixed(2)} dB</span>
                            </div>
                          </div>
                        </div>

                        {/* SSIM Card */}
                        <div className="metric-card">
                          <div className="metric-card-header">
                            <div className="metric-name-container">
                              <span className="metric-name">SSIM (Structural Similarity)</span>
                              <span className="info-tooltip-trigger" title="Computes similarity of structural features, textures, and luminance compared to reference. Range 0 to 1, higher is closer to perfect restoration.">
                                <Info size={13} />
                              </span>
                            </div>
                            {restoredData.metrics.ssim_after > restoredData.metrics.ssim_before && (
                              <span className="metric-badge-diff positive">
                                Improved
                              </span>
                            )}
                          </div>
                          <div className="metric-value-row">
                            <span className="metric-main-value" style={{ color: restoredData.metrics.ssim_after > 0.8 ? "#10b981" : "#f59e0b" }}>
                              {restoredData.metrics.ssim_after.toFixed(4)}
                            </span>
                            <span className="metric-subtitle">Structure match</span>
                          </div>
                          <div className="metric-comparison" style={{ fontSize: "11.5px" }}>
                            <div className="comparison-row">
                              <span>Before:</span>
                              <span>{restoredData.metrics.ssim_before.toFixed(4)}</span>
                            </div>
                            <div className="comparison-row after">
                              <span>After:</span>
                              <span>{restoredData.metrics.ssim_after.toFixed(4)}</span>
                            </div>
                          </div>
                        </div>
                        
                        <p style={{ fontSize: "11px", color: "var(--text-muted)", fontStyle: "italic", textAlign: "center" }}>
                          Metrics relative to ground truth reference: {restoredData.metrics.reference_filename}
                        </p>
                      </div>
                    ) : (
                      <div className="metric-card" style={{ marginTop: "16px", border: "1px dashed var(--border-color)", background: "transparent" }}>
                        <p style={{ fontSize: "12px", color: "var(--text-secondary)", textAlign: "center" }}>
                          PSNR & SSIM metrics are unavailable because no reference ground truth was uploaded. 
                          Upload an image in the optional sidebar block to see these.
                        </p>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <div className="panel" style={{ color: "var(--text-muted)" }}>
                  <h3 className="panel-title">Model Specifications</h3>
                  {modelInfo && modelInfo.config ? (
                    <div style={{ display: "flex", flexDirection: "column", gap: "12px", fontSize: "13px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Target Resolution:</span>
                        <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>
                          {modelInfo.config.image_size} x {modelInfo.config.image_size}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>L1 Loss weight (λ):</span>
                        <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{modelInfo.config.lambda_l1}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Base Learning Rate:</span>
                        <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{modelInfo.config.learning_rate}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Batch Size:</span>
                        <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{modelInfo.config.batch_size}</span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "space-between" }}>
                        <span>Total Epochs:</span>
                        <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{modelInfo.config.epochs}</span>
                      </div>
                    </div>
                  ) : (
                    <p style={{ fontSize: "12px" }}>Select a checkpoint on the sidebar to load model specifications.</p>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB CONTENT: TRAINING CURVES */}
        {activeTab === "training" && (
          <div className="training-analytics">
            {modelInfo && modelInfo.history && modelInfo.history.length > 0 ? (
              <>
                <div className="panel training-summary-panel">
                  <div className="summary-metric-box">
                    <div className="summary-metric-label">Training Length</div>
                    <div className="summary-metric-value">{modelInfo.history.length} Epochs</div>
                    <div className="summary-metric-sub">Iterations recorded</div>
                  </div>
                  <div className="summary-metric-box">
                    <div className="summary-metric-label">Best recorded PSNR</div>
                    <div className="summary-metric-value">
                      {Math.max(...modelInfo.history.map((h) => h.psnr || 0)).toFixed(2)} dB
                    </div>
                    <div className="summary-metric-sub">Highest epoch value</div>
                  </div>
                  <div className="summary-metric-box">
                    <div className="summary-metric-label">Best recorded SSIM</div>
                    <div className="summary-metric-value">
                      {Math.max(...modelInfo.history.map((h) => h.ssim || 0)).toFixed(4)}
                    </div>
                    <div className="summary-metric-sub">Highest similarity</div>
                  </div>
                  <div className="summary-metric-box">
                    <div className="summary-metric-label">Best UIQM (Restored)</div>
                    <div className="summary-metric-value">
                      {Math.max(...modelInfo.history.map((h) => h.uiqm || 0)).toFixed(4)}
                    </div>
                    <div className="summary-metric-sub">Marine quality score</div>
                  </div>
                </div>

                <div className="charts-grid">
                  {/* LOSS GRAPH CARD */}
                  <div className="chart-card">
                    <div className="chart-header">
                      <h4 className="chart-title">Adversarial loss history</h4>
                      <div className="chart-legends">
                        <div className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: "#00f0ff" }}></span>
                          <span>Generator</span>
                        </div>
                        <div className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: "#ef4444" }}></span>
                          <span>Discriminator</span>
                        </div>
                      </div>
                    </div>
                    <SVGLineChart
                      data={modelInfo.history}
                      keys={["loss_G", "loss_D"]}
                      colors={["#00f0ff", "#ef4444"]}
                      labels={["Generator Loss", "Discriminator Loss"]}
                      yLabel="Loss"
                    />
                  </div>

                  {/* PSNR/SSIM METRICS CURVE */}
                  <div className="chart-card">
                    <div className="chart-header">
                      <h4 className="chart-title">Training PSNR / SSIM curves</h4>
                      <div className="chart-legends">
                        <div className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: "#10b981" }}></span>
                          <span>PSNR (dB)</span>
                        </div>
                        <div className="legend-item">
                          <span className="legend-color" style={{ backgroundColor: "#f59e0b" }}></span>
                          <span>SSIM</span>
                        </div>
                      </div>
                    </div>
                    <SVGLineChart
                      data={modelInfo.history}
                      keys={["psnr", "ssim"]}
                      colors={["#10b981", "#f59e0b"]}
                      labels={["PSNR (dB)", "SSIM (0-1)"]}
                      yLabel="Score"
                    />
                  </div>
                </div>

                {/* SINGLE CHART FOR UIQM */}
                <div className="panel chart-card" style={{ maxWidth: "600px", margin: "0 auto", width: "100%" }}>
                  <div className="chart-header">
                    <h4 className="chart-title">UIQM Visual Score evolution</h4>
                    <div className="chart-legends">
                      <div className="legend-item">
                        <span className="legend-color" style={{ backgroundColor: "#a855f7" }}></span>
                        <span>UIQM Value</span>
                      </div>
                    </div>
                  </div>
                  <SVGLineChart
                    data={modelInfo.history}
                    keys={["uiqm"]}
                    colors={["#a855f7"]}
                    labels={["UIQM Score"]}
                    yLabel="UIQM"
                  />
                </div>
              </>
            ) : (
              <div className="panel" style={{ textAlign: "center", padding: "60px" }}>
                <ChartIcon size={48} style={{ color: "var(--text-muted)", marginBottom: "16px" }} />
                <h3>No Training History Found</h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "14px", marginTop: "8px" }}>
                  The active checkpoint does not contain training curves history data, or history logging is empty. 
                  Check training curves in the backend checkpoints file.
                </p>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
