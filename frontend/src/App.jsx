import { useState, useEffect, useRef } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:5001";
const API_URL = `${API_BASE}/api/predict`;
const PERFORMANCE_API_URL = `${API_BASE}/api/model-performance`;
const DISEASES_API_URL = `${API_BASE}/api/diseases`;
const SEARCH_API_URL = `${API_BASE}/api/search`;

function App() {
  // Core prediction state
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showOverlay, setShowOverlay] = useState(false);

  // Input Mode: "upload" vs "camera"
  const [inputMode, setInputMode] = useState("upload");
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [facingMode, setFacingMode] = useState("environment");
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  // Active treatment tab
  const [activeTreatmentTab, setActiveTreatmentTab] = useState("immediate");

  // Assessment History & Timeline state (loaded from localStorage)
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem("plantcare_assessment_history");
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      return [];
    }
  });

  // Disease Directory & Search state
  const [diseasesList, setDiseasesList] = useState([]);
  const [availableCrops, setAvailableCrops] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCrop, setSelectedCrop] = useState("all");
  const [selectedPathogen, setSelectedPathogen] = useState("all");
  const [activeDiseaseModal, setActiveDiseaseModal] = useState(null);
  const [modalLoading, setModalLoading] = useState(false);

  // Verified held-out test metrics as initial state; dynamically refreshed from backend API
  const [performance, setPerformance] = useState({
    model_name: "EfficientNetB0",
    test_images: 236,
    test_accuracy: 47.03,
    macro_precision: 53.09,
    macro_recall: 47.14,
    macro_f1: 45.21,
    weighted_f1: 45.64,
    validation_accuracy: 51.61,
    validation_macro_f1: 47.45,
    evaluation_notes: "Evaluated on 236 genuinely untouched held-out test images across 27 classes."
  });

  // Fetch performance and disease catalog on mount
  useEffect(() => {
    async function fetchPerformance() {
      try {
        const response = await fetch(PERFORMANCE_API_URL);
        if (response.ok) {
          const data = await response.json();
          if (data.status === "success") {
            setPerformance(data);
          }
        }
      } catch (err) {
        console.warn("Could not fetch dynamic model performance, using verified defaults.", err);
      }
    }

    async function fetchDiseases() {
      try {
        const response = await fetch(DISEASES_API_URL);
        if (response.ok) {
          const data = await response.json();
          if (data.status === "success") {
            setDiseasesList(data.diseases || []);
            setAvailableCrops(data.crops || []);
          }
        }
      } catch (err) {
        console.warn("Could not fetch disease catalog", err);
      }
    }

    fetchPerformance();
    fetchDiseases();
  }, []);

  // Clean up camera stream on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Camera Management
  async function startCamera(modeToUse = facingMode) {
    setCameraError("");
    stopCamera();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError("Camera access is not supported by your browser or environment.");
      return;
    }

    try {
      const constraints = {
        video: {
          facingMode: modeToUse,
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraActive(true);
    } catch (err) {
      console.error("Camera error:", err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        setCameraError("Camera permission was denied. Please allow camera access in your browser settings.");
      } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
        setCameraError("No camera device found on your hardware.");
      } else {
        setCameraError(`Camera initialization failed: ${err.message || "Unknown error"}`);
      }
      setIsCameraActive(false);
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  }

  function handleSwitchCamera() {
    const nextMode = facingMode === "environment" ? "user" : "environment";
    setFacingMode(nextMode);
    startCamera(nextMode);
  }

  function handleCapturePhoto() {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setCameraError("Failed to capture image frame.");
          return;
        }

        const capturedFile = new File(
          [blob],
          `camera_leaf_${Date.now()}.jpg`,
          { type: "image/jpeg" }
        );

        setFile(capturedFile);
        const previewUrl = URL.createObjectURL(blob);
        setPreview(previewUrl);
        setResult(null);
        setError("");
        setShowOverlay(false);
        stopCamera();
        setInputMode("upload");
      },
      "image/jpeg",
      0.92
    );
  }

  function handleFileChange(event) {
    const selectedFile = event.target.files[0];
    if (!selectedFile) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(selectedFile.type)) {
      setError("Please select a JPG, PNG, or WEBP image.");
      event.target.value = "";
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Please select an image smaller than 10 MB.");
      event.target.value = "";
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
    setShowOverlay(false);
  }

  async function handlePredict() {
    if (!file) {
      setError("Please upload or capture a crop leaf image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setShowOverlay(false);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.message || "Prediction failed.");
      }

      setResult(data);

      // Save to Assessment History & Timeline
      const newHistoryItem = {
        id: Date.now(),
        timestamp: new Date().toLocaleString([], {
          month: "short",
          day: "numeric",
          hour: "2-digit",
          minute: "2-digit"
        }),
        imageName: file.name || "Camera Capture",
        predictedDisease: data.predicted_disease,
        confidence: data.confidence,
        severityLevel: data.severity_assessment?.level || "Not assessed",
        affectedPct: data.severity_assessment?.affected_area_percentage,
        leafCoverage: data.severity_assessment?.leaf_coverage_percentage,
        overlayImage: data.severity_assessment?.overlay_image,
        previewUrl: preview
      };

      const updatedHistory = [newHistoryItem, ...history.slice(0, 19)];
      setHistory(updatedHistory);
      try {
        localStorage.setItem("plantcare_assessment_history", JSON.stringify(updatedHistory));
      } catch (err) {
        console.warn("Could not save to localStorage", err);
      }
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the backend. Check whether Flask is running."
      );
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
    stopCamera();
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setShowOverlay(false);
    setInputMode("upload");

    const input = document.getElementById("crop-image-input");
    if (input) input.value = "";
  }

  function handleClearHistory() {
    if (window.confirm("Are you sure you want to clear your assessment history?")) {
      setHistory([]);
      localStorage.removeItem("plantcare_assessment_history");
    }
  }

  async function openDiseaseDetail(diseaseId) {
    setModalLoading(true);
    try {
      const response = await fetch(`${API_BASE}/api/diseases/${encodeURIComponent(diseaseId)}`);
      if (response.ok) {
        const data = await response.json();
        setActiveDiseaseModal(data.disease);
      } else {
        alert("Disease details not found.");
      }
    } catch (e) {
      console.error("Error opening disease detail:", e);
    } finally {
      setModalLoading(false);
    }
  }

  // Filter diseases locally for instantaneous search
  const filteredDiseases = diseasesList.filter((item) => {
    const matchesCrop =
      selectedCrop === "all" || item.crop.toLowerCase() === selectedCrop.toLowerCase();
    const matchesPathogen =
      selectedPathogen === "all" ||
      item.pathogen_type.toLowerCase() === selectedPathogen.toLowerCase();

    if (!searchQuery.trim()) {
      return matchesCrop && matchesPathogen;
    }

    const q = searchQuery.toLowerCase();
    const searchSpace = `${item.display_name} ${item.class_name} ${item.crop} ${item.scientific_name} ${item.description}`.toLowerCase();
    return matchesCrop && matchesPathogen && searchSpace.includes(q);
  });

  const predictedDisease =
    result?.predicted_disease ||
    result?.prediction?.disease ||
    "Unknown";

  const confidence = Number(
    result?.confidence ?? result?.prediction?.confidence ?? 0
  );

  const topPredictions = result?.top_predictions || [];

  const confidencePercent = Math.min(100, Math.max(0, confidence));

  // Structured severity data from API with fallback
  const severityData = result?.severity_assessment || {
    status: "Not assessed",
    level: null,
    affected_area_percentage: null,
    leaf_coverage_percentage: null,
    overlay_image: null,
    explanation:
      result?.severity_note ||
      "Severity assessment provides an experimental visible-damage estimate based on leaf and lesion segmentation.",
    method: "Color-space adaptive segmentation (HSV / LAB / ExG)",
    disclaimer: "Experimental visible-damage estimate — not a validated agronomic severity diagnosis."
  };

  const treatmentData = result?.treatment_and_management;
  const diseaseInfo = result?.disease_info;

  return (
    <div className="app">
      {/* 5.1 Navigation Header */}
      <header className="navbar">
        <a href="#home" className="brand">
          <span className="brand-icon">🌿</span>
          <span>PlantCare <strong>AI</strong></span>
        </a>

        <nav className="nav-links">
          <a href="#home">Dashboard</a>
          <a href="#detection">Crop Analysis</a>
          <a href="#timeline">Severity Timeline</a>
          <a href="#search-section">Disease Directory</a>
          <a href="#performance">Model Benchmarks</a>
        </nav>

        <a href="#detection" className="nav-button">
          Analyze Crop ↗
        </a>
      </header>

      <main>
        {/* 5.1 Dashboard Hero */}
        <section className="hero" id="home">
          <div className="hero-content">
            <span className="eyebrow">
              🌱 PRECISION AGRICULTURE · CHAPTER 5 SYSTEM
            </span>

            <h1>
              Detect crop diseases.
              <br />
              <span>Assess severity & treat wisely.</span>
            </h1>

            <p>
              An integrated, AI-driven plant pathology dashboard combining 28-class
              disease identification, computer-vision severity assessment, sequential
              monitoring timelines, and verified agricultural treatment protocols.
            </p>

            <div className="hero-actions">
              <a href="#detection" className="primary-link">
                Analyze a Crop <span>→</span>
              </a>

              <a href="#search-section" className="secondary-link">
                🔍 Search Disease Database
              </a>
            </div>

            <div className="hero-note">
              <span className="status-dot"></span>
              Powered by fine-tuned {performance.model_name} on PlantDoc field benchmarks
            </div>
          </div>

          <div className="hero-visual">
            <div className="hero-image">
              <img
                src="https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=85"
                alt="Green agricultural fields"
              />
            </div>

            <div className="floating-card">
              <span className="floating-icon">🔬</span>
              <div>
                <strong>Complete Diagnostic Suite</strong>
                <p>Detection · Severity · Treatment · Search</p>
              </div>
            </div>
          </div>
        </section>

        {/* Dashboard Quick Navigation & Stats */}
        <section className="stats-strip">
          <div className="stat-item">
            <strong>28</strong>
            <span>Supported Classes</span>
          </div>

          <div className="stat-item">
            <strong>13</strong>
            <span>Major Crop Types</span>
          </div>

          <div className="stat-item">
            <strong>{performance.test_accuracy}%</strong>
            <span>Held-Out Test Accuracy</span>
          </div>

          <div className="stat-item">
            <strong>CV + ExG</strong>
            <span>Severity Engine</span>
          </div>
        </section>

        {/* 5.2 & 5.4 Image Upload and Real-Time Camera Detection */}
        <section className="detection-section" id="detection">
          <div className="section-intro">
            <span className="eyebrow">DIAGNOSTIC WORKSPACE</span>
            <h2>Analyze your crop leaf</h2>
            <p>
              Upload a high-resolution leaf photograph or capture live through your device camera
              for instant disease identification, severity estimation, and tailored management guidance.
            </p>
          </div>

          <div className="workspace">
            {/* Input Card */}
            <div className="upload-card">
              <div className="section-heading">
                <span className="step-number">01</span>
                <div>
                  <h3>Select input source</h3>
                  <p>Choose between file upload or live camera capture.</p>
                </div>
              </div>

              {/* Mode Switcher Tabs */}
              <div className="input-mode-tabs">
                <button
                  type="button"
                  className={`mode-tab-btn ${inputMode === "upload" ? "active" : ""}`}
                  onClick={() => {
                    setInputMode("upload");
                    stopCamera();
                  }}
                >
                  📁 File Upload
                </button>
                <button
                  type="button"
                  className={`mode-tab-btn ${inputMode === "camera" ? "active" : ""}`}
                  onClick={() => {
                    setInputMode("camera");
                    startCamera();
                  }}
                >
                  📷 Live Camera Capture
                </button>
              </div>

              {inputMode === "upload" ? (
                <>
                  <label
                    className={`upload-area ${preview ? "has-preview" : ""}`}
                    htmlFor="crop-image-input"
                  >
                    {preview ? (
                      <img
                        src={preview}
                        alt="Selected crop leaf"
                        className="preview-image"
                      />
                    ) : (
                      <div className="upload-placeholder">
                        <span className="upload-icon">📷</span>
                        <h4>Choose a crop leaf image</h4>
                        <p>Click here to browse files or drag & drop</p>
                        <span className="file-hint">JPG · PNG · WEBP · Max 10 MB</span>
                      </div>
                    )}
                  </label>

                  <input
                    id="crop-image-input"
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={handleFileChange}
                    hidden
                  />
                </>
              ) : (
                /* Live Camera Capture Section */
                <div className="camera-viewfinder-card">
                  {cameraError ? (
                    <div className="camera-error-box">
                      <span>⚠️</span>
                      <p>{cameraError}</p>
                      <button
                        type="button"
                        className="retry-camera-btn"
                        onClick={() => startCamera()}
                      >
                        Retry Camera
                      </button>
                    </div>
                  ) : (
                    <div className="camera-live-container">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="camera-video-feed"
                      />
                      <div className="camera-target-crosshairs">
                        <span className="corner top-left"></span>
                        <span className="corner top-right"></span>
                        <span className="corner bottom-left"></span>
                        <span className="corner bottom-right"></span>
                        <p className="crosshairs-label">Align Leaf within Frame</p>
                      </div>

                      <div className="camera-controls-bar">
                        <button
                          type="button"
                          className="camera-flip-btn"
                          onClick={handleSwitchCamera}
                          title="Switch front/back camera"
                        >
                          🔄 Switch
                        </button>
                        <button
                          type="button"
                          className="camera-snap-btn"
                          onClick={handleCapturePhoto}
                        >
                          📸 Capture Photo
                        </button>
                        <button
                          type="button"
                          className="camera-cancel-btn"
                          onClick={() => {
                            stopCamera();
                            setInputMode("upload");
                          }}
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {file && (
                <div className="file-info">
                  <span>🖼️</span>
                  <span className="file-name">{file.name}</span>
                  <button
                    className="remove-button"
                    onClick={resetForm}
                    type="button"
                  >
                    Remove
                  </button>
                </div>
              )}

              <div className="upload-note">
                <span>ⓘ</span>
                <p>
                  Ensure the leaf is in focus and well-lit. Avoid heavy shadows or holding
                  fingers over lesion regions for optimal segmentation.
                </p>
              </div>

              {error && <div className="error-message">{error}</div>}

              <button
                className="predict-button"
                onClick={handlePredict}
                disabled={!file || loading}
                type="button"
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing Crop & Computing Severity...
                  </>
                ) : (
                  "Analyze Crop Image 🌿"
                )}
              </button>
            </div>

            {/* Results Card */}
            <div className="results-card">
              <div className="section-heading">
                <span className="step-number">02</span>
                <div>
                  <h3>Diagnostic findings</h3>
                  <p>Classification, severity analysis, and treatment suggestions.</p>
                </div>
              </div>

              {result ? (
                <div className="results-content animate-fade-in">
                  {/* Disease Classification Header */}
                  <div className="prediction-hero">
                    <span className="result-label">Predicted Condition</span>
                    <h3 className="disease-title">{predictedDisease}</h3>

                    {diseaseInfo?.scientific_name && (
                      <p className="scientific-name">
                        <em>{diseaseInfo.scientific_name}</em>
                      </p>
                    )}

                    <div className="confidence-block">
                      <div className="confidence-header">
                        <span>Prediction Certainty</span>
                        <strong>{confidence}%</strong>
                      </div>

                      <div className="confidence-track">
                        <div
                          className="confidence-fill"
                          style={{ width: `${confidencePercent}%` }}
                        />
                      </div>
                    </div>

                    <p className="confidence-note">
                      Confidence represents the model's certainty for this specific image,
                      distinct from the overall test accuracy ({performance.test_accuracy}%).
                    </p>
                  </div>

                  {/* 5.3 Disease Severity Assessment Card */}
                  <div className="severity-assessment-card">
                    <div className="severity-card-header">
                      <div className="severity-icon-badge">🔬</div>
                      <div>
                        <h4>Disease Severity Assessment</h4>
                        <span className="severity-subtitle">
                          Computer Vision Visible Damage Estimation
                        </span>
                      </div>
                      <span
                        className={`status-badge ${
                          severityData.level === "Minimal / Healthy"
                            ? "minimal"
                            : severityData.level === "Mild"
                            ? "mild"
                            : severityData.level === "Moderate"
                            ? "moderate"
                            : severityData.level === "Severe"
                            ? "severe"
                            : "neutral"
                        }`}
                      >
                        {severityData.level || severityData.status}
                      </span>
                    </div>

                    {severityData.status === "Assessed" ? (
                      <div className="severity-active-result">
                        {/* Metrics Grid */}
                        <div className="severity-metrics-grid">
                          <div className="severity-metric-box">
                            <span className="severity-metric-label">Estimated Damage</span>
                            <span className="severity-metric-value highlight">
                              {severityData.affected_area_percentage}%
                            </span>
                            <span className="severity-metric-sub">of segmented leaf</span>
                          </div>

                          <div className="severity-metric-box">
                            <span className="severity-metric-label">Severity Level</span>
                            <span className="severity-metric-value">
                              {severityData.level}
                            </span>
                            <span className="severity-metric-sub">EPPO / Horsfall-Barratt</span>
                          </div>

                          <div className="severity-metric-box">
                            <span className="severity-metric-label">Leaf Area in Photo</span>
                            <span className="severity-metric-value">
                              {severityData.leaf_coverage_percentage}%
                            </span>
                            <span className="severity-metric-sub">of frame</span>
                          </div>
                        </div>

                        {/* Affected Area Progress Bar */}
                        <div className="severity-progress-section">
                          <div className="severity-progress-header">
                            <span>Affected Leaf Area</span>
                            <strong>{severityData.affected_area_percentage}%</strong>
                          </div>
                          <div className="severity-progress-track">
                            <div
                              className={`severity-progress-fill ${
                                severityData.level === "Minimal / Healthy"
                                  ? "fill-minimal"
                                  : severityData.level === "Mild"
                                  ? "fill-mild"
                                  : severityData.level === "Moderate"
                                  ? "fill-moderate"
                                  : "fill-severe"
                              }`}
                              style={{
                                width: `${Math.min(
                                  100,
                                  Math.max(3, severityData.affected_area_percentage)
                                )}%`
                              }}
                            />
                          </div>
                          <div className="severity-threshold-markers">
                            <span>Minimal (&lt;3%)</span>
                            <span>Mild (3-10%)</span>
                            <span>Moderate (10-25%)</span>
                            <span>Severe (&gt;25%)</span>
                          </div>
                        </div>

                        {/* Diagnostic Overlay Viewer Toggle */}
                        {severityData.overlay_image && (
                          <div className="severity-overlay-section">
                            <button
                              type="button"
                              className="severity-overlay-toggle-btn"
                              onClick={() => setShowOverlay(!showOverlay)}
                            >
                              <span>
                                {showOverlay
                                  ? "🙈 Hide Lesion Diagnostic Overlay"
                                  : "🔬 View Lesion Diagnostic Overlay"}
                              </span>
                            </button>

                            {showOverlay && (
                              <div className="severity-overlay-display">
                                <img
                                  src={severityData.overlay_image}
                                  alt="Lesion Segmentation Diagnostic Overlay"
                                  className="severity-overlay-img"
                                />
                                <div className="severity-overlay-legend">
                                  <span>
                                    <strong style={{ color: "#27ae60" }}>🟢 Green Outline:</strong> Segmented Leaf Perimeter
                                  </span>
                                  <span>
                                    <strong style={{ color: "#e74c3c" }}>🔴 Red Highlight:</strong> Detected Lesion Regions
                                  </span>
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        <p className="severity-details">{severityData.explanation}</p>

                        <div className="severity-scientific-disclaimer">
                          <span>⚠️</span>
                          <div>
                            <strong>Experimental visible-damage estimate — not a validated agronomic severity diagnosis.</strong>
                            <p style={{ margin: "4px 0 0 0", fontSize: "11px", color: "#635832" }}>
                              Calculated via adaptive color-space segmentation (HSV, LAB, and Excess Green Index).
                              Biological disease severity may vary depending on vascular penetration.
                            </p>
                          </div>
                        </div>
                      </div>
                    ) : severityData.status === "Unable to segment leaf" ? (
                      <div className="severity-notice-block warning">
                        <p className="severity-explanation">
                          <strong>⚠️ {severityData.status}:</strong> {severityData.explanation}
                        </p>
                        <p
                          className="severity-tip"
                          style={{ margin: "6px 0 0 0", fontSize: "11px", color: "#8a6d3b" }}
                        >
                          💡 <em>Tip: Ensure the leaf is in focus, occupies the central frame, and has distinct color contrast against the background.</em>
                        </p>
                      </div>
                    ) : (
                      <div className="severity-notice-block">
                        <p className="severity-explanation">{severityData.explanation}</p>
                        <div className="severity-scientific-disclaimer">
                          <span>⚠️</span>
                          <span>
                            <strong>Scientific Safeguard:</strong> Mild, Moderate, or Severe ratings
                            cannot be inferred from classification confidence scores or arbitrary rules.
                            Validated severity requires lesion segmentation or dedicated multi-stage datasets.
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* 5.6 Dedicated Treatment & Management Recommendations Card */}
                  <div className="treatment-card" id="treatment-card">
                    <div className="treatment-card-header">
                      <div className="treatment-icon-badge">🩺</div>
                      <div>
                        <h4>Treatment & Management Recommendations</h4>
                        <span className="treatment-subtitle">
                          Evidence-based agricultural extension protocols
                        </span>
                      </div>
                      {diseaseInfo && (
                        <div className="treatment-header-tags">
                          <span className="crop-pill">{diseaseInfo.crop}</span>
                          <span className={`pathogen-pill ${diseaseInfo.pathogen_type?.toLowerCase().replace(/[^a-z]/g, "")}`}>
                            {diseaseInfo.pathogen_type}
                          </span>
                        </div>
                      )}
                    </div>

                    {/* Uncertainty Precaution Banner */}
                    {result.uncertainty_warning && (
                      <div
                        className={`treatment-alert-banner ${
                          diseaseInfo?.is_healthy ? "healthy" : "warning"
                        }`}
                      >
                        <span className="alert-icon">
                          {diseaseInfo?.is_healthy ? "✅" : "⚠️"}
                        </span>
                        <div>
                          <strong>
                            {diseaseInfo?.is_healthy
                              ? "Healthy Leaf Verified"
                              : "Diagnostic Precaution:"}
                          </strong>
                          <p>{result.uncertainty_warning}</p>
                        </div>
                      </div>
                    )}

                    {treatmentData ? (
                      <div className="treatment-content-wrapper">
                        {/* Tab Switcher */}
                        <div className="treatment-tab-bar">
                          <button
                            type="button"
                            className={`treatment-tab ${activeTreatmentTab === "immediate" ? "active" : ""}`}
                            onClick={() => setActiveTreatmentTab("immediate")}
                          >
                            🚨 Immediate Actions
                          </button>
                          <button
                            type="button"
                            className={`treatment-tab ${activeTreatmentTab === "cultural" ? "active" : ""}`}
                            onClick={() => setActiveTreatmentTab("cultural")}
                          >
                            🌾 Cultural & Prevention
                          </button>
                          <button
                            type="button"
                            className={`treatment-tab ${activeTreatmentTab === "options" ? "active" : ""}`}
                            onClick={() => setActiveTreatmentTab("options")}
                          >
                            🧪 Treatment Options
                          </button>
                          <button
                            type="button"
                            className={`treatment-tab ${activeTreatmentTab === "expert" ? "active" : ""}`}
                            onClick={() => setActiveTreatmentTab("expert")}
                          >
                            👨‍🌾 Expert Advice
                          </button>
                        </div>

                        {/* Tab 1: Immediate Actions */}
                        {activeTreatmentTab === "immediate" && (
                          <div className="treatment-tab-pane animate-fade-in">
                            <h5>Immediate Sanitation & Containment Steps:</h5>
                            <ul className="treatment-checklist">
                              {treatmentData.immediate_actions?.map((act, idx) => (
                                <li key={idx}>
                                  <span className="bullet-num">{idx + 1}</span>
                                  <span>{act}</span>
                                </li>
                              ))}
                            </ul>
                            {diseaseInfo?.favorable_conditions && (
                              <div className="conditions-callout">
                                <strong>🌧️ Conditions Favoring Spread:</strong>{" "}
                                {diseaseInfo.favorable_conditions}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Tab 2: Cultural Practices & Prevention */}
                        {activeTreatmentTab === "cultural" && (
                          <div className="treatment-tab-pane animate-fade-in">
                            <h5>Cultural Management & Field Hygiene:</h5>
                            <ul className="treatment-bullet-list">
                              {treatmentData.cultural_practices?.map((cp, idx) => (
                                <li key={idx}>🌱 {cp}</li>
                              ))}
                            </ul>

                            <h5 style={{ marginTop: "16px" }}>Preventive Measures for Next Cycle:</h5>
                            <ul className="treatment-bullet-list">
                              {treatmentData.preventive_measures?.map((pm, idx) => (
                                <li key={idx}>🛡️ {pm}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Tab 3: Treatment Options */}
                        {activeTreatmentTab === "options" && (
                          <div className="treatment-tab-pane animate-fade-in">
                            <h5>Evidence-Based Treatment Categories:</h5>
                            <ul className="treatment-bullet-list">
                              {treatmentData.treatment_options?.map((opt, idx) => (
                                <li key={idx}>🔬 {opt}</li>
                              ))}
                            </ul>

                            <div className="safety-regulatory-box">
                              <strong>⚖️ Regulatory & Safety Mandate:</strong>
                              <p>
                                {treatmentData.disclaimer ||
                                  "Pesticide applications must follow product labels and local agricultural regulations strictly. Do not apply chemicals without label authorization for your crop."}
                              </p>
                            </div>
                          </div>
                        )}

                        {/* Tab 4: Expert Advice */}
                        {activeTreatmentTab === "expert" && (
                          <div className="treatment-tab-pane animate-fade-in">
                            <h5>When to Seek Professional Consultation:</h5>
                            <div className="expert-advice-quote">
                              <span>👨‍🏫</span>
                              <p>{treatmentData.expert_advice}</p>
                            </div>
                            <p className="expert-footer-note">
                              Local university agricultural extension offices provide subsidized
                              microscopic analysis and race identification for aggressive blights.
                            </p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="treatment-notice-empty">
                        <p>
                          Treatment recommendations are unavailable for this disease. Please consult a qualified agricultural expert.
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Differential Diagnoses */}
                  {topPredictions.length > 0 && (
                    <div className="differential-block">
                      <h4 className="top-heading">Top Differential Diagnoses</h4>
                      <div className="prediction-list">
                        {topPredictions.map((item, index) => (
                          <div key={index} className="prediction-row">
                            <span className="rank-badge">#{index + 1}</span>
                            <span className="prediction-name">{item.disease}</span>
                            <span className="prediction-percentage">{item.confidence}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="empty-results">
                  <div className="empty-icon">🌱</div>
                  <h4>No crop leaf analyzed yet</h4>
                  <p>
                    Upload an image or capture via your camera on the left, then click
                    <strong> Analyze Crop Image</strong> to generate comprehensive findings.
                  </p>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* 5.3 Dedicated Disease Severity Timeline Section */}
        <section className="timeline-section" id="timeline">
          <div className="section-intro">
            <span className="eyebrow">MONITORING LOG</span>
            <h2>Assessment History & Severity Timeline</h2>
            <p>
              Compare sequential plant samples over time. Track affected leaf area percentages,
              visual damage ratings, and disease diagnoses across your field evaluations.
            </p>
          </div>

          <div className="timeline-container">
            {history.length > 0 ? (
              <>
                <div className="timeline-toolbar">
                  <span className="timeline-count">
                    Recorded Assessments: <strong>{history.length}</strong>
                  </span>
                  <button
                    type="button"
                    className="clear-history-btn"
                    onClick={handleClearHistory}
                  >
                    🗑️ Clear History
                  </button>
                </div>

                <div className="timeline-cards-grid">
                  {history.map((entry, index) => (
                    <div key={entry.id || index} className="timeline-card">
                      <div className="timeline-card-header">
                        <span className="timeline-time">🕒 {entry.timestamp}</span>
                        <span
                          className={`status-badge ${
                            entry.severityLevel === "Minimal / Healthy"
                              ? "minimal"
                              : entry.severityLevel === "Mild"
                              ? "mild"
                              : entry.severityLevel === "Moderate"
                              ? "moderate"
                              : entry.severityLevel === "Severe"
                              ? "severe"
                              : "neutral"
                          }`}
                        >
                          {entry.severityLevel}
                        </span>
                      </div>

                      <div className="timeline-body">
                        <h4 className="timeline-disease">{entry.predictedDisease}</h4>
                        <div className="timeline-stats">
                          <div>
                            <span>Certainty:</span>
                            <strong>{entry.confidence}%</strong>
                          </div>
                          <div>
                            <span>Affected Area:</span>
                            <strong>
                              {entry.affectedPct !== null && entry.affectedPct !== undefined
                                ? `${entry.affectedPct}%`
                                : "N/A"}
                            </strong>
                          </div>
                        </div>

                        {entry.overlayImage ? (
                          <div className="timeline-thumbnail-wrapper">
                            <img
                              src={entry.overlayImage}
                              alt="Diagnostic overlay"
                              className="timeline-thumb-img"
                            />
                            <span className="thumb-tag">CV Overlay</span>
                          </div>
                        ) : entry.previewUrl ? (
                          <div className="timeline-thumbnail-wrapper">
                            <img
                              src={entry.previewUrl}
                              alt="Sample thumbnail"
                              className="timeline-thumb-img"
                            />
                            <span className="thumb-tag">Original</span>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="timeline-scientific-note">
                  <span>ℹ️</span>
                  <p>
                    <strong>Chronological Note:</strong> This timeline organizes sequential samples
                    evaluated by the user. True biological progression cannot be modeled from a single snapshot
                    without repeated longitudinal tracking of the same flagged leaf tag over consecutive days.
                  </p>
                </div>
              </>
            ) : (
              <div className="timeline-empty-card">
                <span className="empty-clock">⏳</span>
                <h4>No historical assessments recorded yet</h4>
                <p>
                  As you analyze crop samples using the detection workspace above, your assessments,
                  severity ratings, and diagnostic overlays will automatically appear in this timeline.
                </p>
              </div>
            )}
          </div>
        </section>

        {/* 5.5 Dedicated Disease Search & Crop Disease Information Section */}
        <section className="search-section" id="search-section">
          <div className="section-intro">
            <span className="eyebrow">CROP DISEASE ENCYCLOPEDIA</span>
            <h2>Disease Search & Knowledge Directory</h2>
            <p>
              Explore the complete 28-class PlantDoc plant pathology database. Browse symptoms,
              favorable weather conditions, and certified treatment recommendations by crop or disease name.
            </p>
          </div>

          <div className="search-container">
            {/* Search Input Bar */}
            <div className="search-bar-wrapper">
              <span className="search-icon">🔍</span>
              <input
                type="text"
                className="search-input-field"
                placeholder="Search by disease name (e.g. Blight, Scab, Rust, Mites) or crop..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
              {searchQuery && (
                <button
                  type="button"
                  className="search-clear-btn"
                  onClick={() => setSearchQuery("")}
                >
                  ✕
                </button>
              )}
            </div>

            {/* Crop Filter Chips */}
            <div className="filter-chips-row">
              <span className="filter-label">Filter by Crop:</span>
              <button
                type="button"
                className={`filter-chip ${selectedCrop === "all" ? "active" : ""}`}
                onClick={() => setSelectedCrop("all")}
              >
                All Crops ({diseasesList.length})
              </button>
              {availableCrops.map((crop) => (
                <button
                  key={crop}
                  type="button"
                  className={`filter-chip ${selectedCrop.toLowerCase() === crop.toLowerCase() ? "active" : ""}`}
                  onClick={() => setSelectedCrop(crop.toLowerCase())}
                >
                  {crop}
                </button>
              ))}
            </div>

            {/* Pathogen Type Filter Chips */}
            <div className="filter-chips-row" style={{ marginTop: "8px" }}>
              <span className="filter-label">Pathology Type:</span>
              {["all", "Fungal", "Bacterial", "Viral", "Pest / Mite", "None (Healthy)"].map((type) => (
                <button
                  key={type}
                  type="button"
                  className={`filter-chip pathogen ${selectedPathogen.toLowerCase() === type.toLowerCase() ? "active" : ""}`}
                  onClick={() => setSelectedPathogen(type.toLowerCase())}
                >
                  {type === "all" ? "All Pathologies" : type}
                </button>
              ))}
            </div>

            {/* Disease Cards Grid */}
            <div className="diseases-grid">
              {filteredDiseases.length > 0 ? (
                filteredDiseases.map((item) => (
                  <div key={item.id} className="disease-catalog-card">
                    <div className="catalog-card-header">
                      <span className="catalog-crop-badge">{item.crop}</span>
                      <span className={`pathogen-pill ${item.pathogen_type.toLowerCase().replace(/[^a-z]/g, "")}`}>
                        {item.pathogen_type}
                      </span>
                    </div>

                    <h4 className="catalog-disease-name">{item.display_name}</h4>
                    {item.scientific_name && (
                      <p className="catalog-scientific">
                        <em>{item.scientific_name}</em>
                      </p>
                    )}

                    <p className="catalog-desc">
                      {item.description.length > 130
                        ? `${item.description.substring(0, 130)}...`
                        : item.description}
                    </p>

                    <button
                      type="button"
                      className="catalog-explore-btn"
                      onClick={() => openDiseaseDetail(item.id)}
                    >
                      View Treatment & Guide 📖
                    </button>
                  </div>
                ))
              ) : (
                <div className="catalog-no-results">
                  <span className="no-result-icon">🔬</span>
                  <h4>No matching diseases found for &ldquo;{searchQuery}&rdquo;</h4>
                  <p>
                    The current benchmark specifically covers 28 conditions across 13 crops
                    (Apple, Bell Pepper, Blueberry, Cherry, Corn, Grape, Peach, Potato, Raspberry,
                    Soybean, Squash, Strawberry, and Tomato).
                  </p>
                  <button
                    type="button"
                    className="reset-search-btn"
                    onClick={() => {
                      setSearchQuery("");
                      setSelectedCrop("all");
                      setSelectedPathogen("all");
                    }}
                  >
                    Reset All Filters
                  </button>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* Modal for Disease Details from Catalog Search */}
        {activeDiseaseModal && (
          <div className="modal-backdrop" onClick={() => setActiveDiseaseModal(null)}>
            <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <div>
                  <span className="modal-crop-badge">{activeDiseaseModal.crop}</span>
                  <h3>{activeDiseaseModal.display_name}</h3>
                  {activeDiseaseModal.scientific_name && (
                    <p className="modal-scientific">
                      <em>{activeDiseaseModal.scientific_name}</em>
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  className="modal-close-btn"
                  onClick={() => setActiveDiseaseModal(null)}
                >
                  ✕
                </button>
              </div>

              <div className="modal-body">
                <div className="modal-section">
                  <h5>📖 Overview</h5>
                  <p>{activeDiseaseModal.description}</p>
                </div>

                {activeDiseaseModal.symptoms && activeDiseaseModal.symptoms.length > 0 && (
                  <div className="modal-section">
                    <h5>🔍 Typical Symptoms</h5>
                    <ul className="modal-bullet-list">
                      {activeDiseaseModal.symptoms.map((s, idx) => (
                        <li key={idx}>{s}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {activeDiseaseModal.favorable_conditions && (
                  <div className="modal-section">
                    <h5>🌧️ Favorable Weather Conditions</h5>
                    <p>{activeDiseaseModal.favorable_conditions}</p>
                  </div>
                )}

                {activeDiseaseModal.treatment_and_management && (
                  <div className="modal-section treatment-modal-block">
                    <h5>🩺 Treatment & Management Protocol</h5>

                    <h6>Immediate Actions:</h6>
                    <ul className="modal-bullet-list">
                      {activeDiseaseModal.treatment_and_management.immediate_actions?.map((a, i) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ul>

                    <h6>Cultural & Preventive Practices:</h6>
                    <ul className="modal-bullet-list">
                      {activeDiseaseModal.treatment_and_management.cultural_practices?.map((c, i) => (
                        <li key={i}>{c}</li>
                      ))}
                    </ul>

                    <h6>Treatment Options:</h6>
                    <ul className="modal-bullet-list">
                      {activeDiseaseModal.treatment_and_management.treatment_options?.map((t, i) => (
                        <li key={i}>{t}</li>
                      ))}
                    </ul>

                    <h6>Expert Consultation:</h6>
                    <p>{activeDiseaseModal.treatment_and_management.expert_advice}</p>

                    <div className="safety-regulatory-box" style={{ marginTop: "12px" }}>
                      <p>{activeDiseaseModal.treatment_and_management.disclaimer}</p>
                    </div>
                  </div>
                )}
              </div>

              <div className="modal-footer">
                <button
                  type="button"
                  className="modal-dismiss-btn"
                  onClick={() => setActiveDiseaseModal(null)}
                >
                  Close Guide
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Model Performance Section */}
        <section className="performance-section" id="performance">
          <div className="section-intro">
            <span className="eyebrow">VERIFIED BENCHMARKS</span>
            <h2>Model Performance & Evaluation</h2>
            <p>
              Grounded, scientifically audited performance metrics evaluated on the
              held-out test set of 236 images across 27 classes.
            </p>
          </div>

          <div className="performance-container">
            <div className="performance-grid">
              <div className="metric-card highlight-card">
                <div className="metric-header">
                  <span className="metric-icon">🎯</span>
                  <span className="metric-tag">Primary Metric</span>
                </div>
                <div className="metric-value">{performance.test_accuracy}%</div>
                <div className="metric-label">Held-out Test Accuracy</div>
                <p className="metric-desc">
                  Overall top-1 classification accuracy across 236 untouched test images.
                </p>
              </div>

              <div className="metric-card">
                <div className="metric-header">
                  <span className="metric-icon">⚖️</span>
                  <span className="metric-tag">Balanced</span>
                </div>
                <div className="metric-value">{performance.macro_f1}%</div>
                <div className="metric-label">Macro-Averaged F1-Score</div>
                <p className="metric-desc">
                  Harmonic mean of precision and recall giving equal weight to all classes.
                </p>
              </div>

              <div className="metric-card">
                <div className="metric-header">
                  <span className="metric-icon">📊</span>
                  <span className="metric-tag">Weighted</span>
                </div>
                <div className="metric-value">{performance.weighted_f1}%</div>
                <div className="metric-label">Weighted F1-Score</div>
                <p className="metric-desc">
                  F1-Score weighted by class representation in the test set.
                </p>
              </div>

              <div className="metric-card">
                <div className="metric-header">
                  <span className="metric-icon">🔍</span>
                  <span className="metric-tag">Precision</span>
                </div>
                <div className="metric-value">{performance.macro_precision}%</div>
                <div className="metric-label">Macro Precision</div>
                <p className="metric-desc">
                  Average precision across all 27 evaluation classes.
                </p>
              </div>

              <div className="metric-card">
                <div className="metric-header">
                  <span className="metric-icon">📈</span>
                  <span className="metric-tag">Recall</span>
                </div>
                <div className="metric-value">{performance.macro_recall}%</div>
                <div className="metric-label">Macro Recall</div>
                <p className="metric-desc">
                  Average sensitivity across all classes.
                </p>
              </div>

              <div className="metric-card">
                <div className="metric-header">
                  <span className="metric-icon">🧪</span>
                  <span className="metric-tag">Validation</span>
                </div>
                <div className="metric-value">{performance.validation_accuracy}%</div>
                <div className="metric-label">Validation Accuracy</div>
                <p className="metric-desc">
                  Peak accuracy on the 465-image stratified validation split.
                </p>
              </div>
            </div>

            <div className="performance-insights">
              <div className="insight-card">
                <div className="insight-header">
                  <span className="insight-icon">🔬</span>
                  <h4>Why PlantDoc Differs from Lab Datasets</h4>
                </div>
                <p>
                  PlantDoc contains in-the-wild photography with complex backgrounds, multiple overlapping leaves,
                  and natural lighting variations. Published academic benchmarks achieve 45%–55% for whole-image
                  classifiers. The 95%+ claims in literature apply strictly to lab-isolated datasets like PlantVillage.
                </p>
              </div>

              <div className="insight-card">
                <div className="insight-header">
                  <span className="insight-icon">🔍</span>
                  <h4>Distinguishing Accuracy from Prediction Confidence</h4>
                </div>
                <p>
                  Prediction confidence on an uploaded image (e.g. 70% or 85%) is the model&apos;s certainty for
                  that single instance. It must not be conflated with the overall model accuracy (<strong>{performance.test_accuracy}%</strong>),
                  which is the empirical benchmark across the complete held-out test suite.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Disclaimer */}
        <section className="disclaimer">
          <strong>Important Agricultural Disclaimer</strong>
          <p>
            PlantCare AI provides computer-vision assisted plant disease predictions and visible damage estimates.
            Predictions should never replace on-site physical diagnosis by certified plant pathologists or county extension specialists.
            Always confirm product registrations, pesticide safety guidelines, and pre-harvest intervals before chemical application.
          </p>
        </section>
      </main>

      <footer className="footer">
        <a href="#home" className="brand">
          <span className="brand-icon">🌿</span>
          <span>PlantCare <strong>AI</strong></span>
        </a>

        <p>
          PlantCare AI · Drone-Based Crop Disease Severity Detection · Complete Chapter 5 Diagnostic Suite
        </p>

        <a href="#home" className="back-to-top">
          Back to top ↑
        </a>
      </footer>
    </div>
  );
}

export default App;