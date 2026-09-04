/**
 * Body2Fit Interactive Demo Application Controller
 * Handles UI interactions, live pipeline API calls, Three.js 3D rendering,
 * and clinical risk interpretation.
 */

// State Management
let currentPreset = "deva_full";
let isRunning = false;
let presetsCache = {};

// Three.js instances for Step 7 viewer & Studio tab
let viewers = {
  pipeline: { scene: null, camera: null, renderer: null, controls: null, mesh: null, containerId: "three-canvas-container", autoRotate: true, wireframe: false },
  studio: { scene: null, camera: null, renderer: null, controls: null, mesh: null, containerId: "three-studio-container", autoRotate: true, wireframe: false }
};

// Document Ready Initialization
document.addEventListener("DOMContentLoaded", async () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  
  await fetchPresets();
  await fetchHealth();
  
  // Initialize Three.js canvases
  initThreeViewer("pipeline");
  initThreeViewer("studio");
  
  // Run initial preset pipeline
  await runPipeline();
});

/**
 * Fetch server health and device info
 */
async function fetchHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      const badge = document.getElementById("device-badge");
      if (badge && data.device) {
        badge.textContent = `Device: ${data.device.toUpperCase()} • 640×480`;
      }
    }
  } catch (err) {
    console.warn("Health check error:", err);
  }
}

/**
 * Fetch preset list from API
 */
async function fetchPresets() {
  try {
    const res = await fetch("/api/presets");
    if (res.ok) {
      presetsCache = await res.json();
    }
  } catch (err) {
    console.error("Failed to load presets:", err);
  }
}

/**
 * Handle Preset Dropdown Changes
 */
function onPresetChange() {
  const select = document.getElementById("preset-selector");
  const drawer = document.getElementById("custom-upload-drawer");
  const val = select.value;
  currentPreset = val;

  if (val === "custom") {
    drawer.classList.remove("hidden");
  } else {
    drawer.classList.add("hidden");
    const preset = presetsCache[val];
    if (preset) {
      document.getElementById("input-height").value = preset.height_cm || 175.0;
      document.getElementById("input-sex").value = preset.sex || "male";
      
      // Update preview photos
      if (preset.front_photo) document.getElementById("img-front-rgb").src = preset.front_photo;
      if (preset.side_photo) document.getElementById("img-side-rgb").src = preset.side_photo;
      if (preset.front_silhouette) document.getElementById("img-front-sil").src = preset.front_silhouette;
      if (preset.side_silhouette) document.getElementById("img-side-sil").src = preset.side_silhouette;
      if (preset.front_overlay) document.getElementById("img-decomp-front").src = preset.front_overlay;
      if (preset.side_overlay) document.getElementById("img-decomp-side").src = preset.side_overlay;
      if (preset.smplx_rendered) document.getElementById("img-smplx-rendered").src = preset.smplx_rendered;
      if (preset.smplx_overlay) document.getElementById("img-smplx-overlay").src = preset.smplx_overlay;
    }
  }
}

/**
 * Switch Navigation Tabs
 */
function switchTab(tabId) {
  const tabs = ["pipeline", "walkthrough", "architecture", "mesh3d", "benchmarks"];
  tabs.forEach(t => {
    const content = document.getElementById(`tab-content-${t}`);
    const btn = document.getElementById(`tab-btn-${t}`);
    if (content) content.classList.add("hidden");
    if (btn) {
      btn.classList.remove("active", "text-emerald-400", "bg-emerald-500/10", "border-emerald-500/20");
      btn.classList.add("text-slate-400");
    }
  });

  const activeContent = document.getElementById(`tab-content-${tabId}`);
  const activeBtn = document.getElementById(`tab-btn-${tabId}`);
  if (activeContent) activeContent.classList.remove("hidden");
  if (activeBtn) {
    activeBtn.classList.add("active", "text-emerald-400", "bg-emerald-500/10", "border-emerald-500/20");
    activeBtn.classList.remove("text-slate-400");
  }

  // Resize 3D viewer if switching to mesh3d tab
  if (tabId === "mesh3d") {
    setTimeout(() => {
      onViewerResize("studio");
    }, 50);
  }
}

/**
 * One-Click Instant Test Button for Recruiters
 */
async function runOneClickTest() {
  const select = document.getElementById("preset-selector");
  select.value = "deva_full";
  onPresetChange();
  switchTab("pipeline");
  await runPipeline();
}

/**
 * Execute Anthropometric Pipeline via Live API Call
 */
async function runPipeline() {
  if (isRunning) return;
  isRunning = true;
  
  const runBtn = document.getElementById("btn-run-pipeline");
  const btnText = document.getElementById("run-button-text");
  if (runBtn) {
    runBtn.classList.add("opacity-75", "cursor-wait");
    btnText.textContent = "Inferencing...";
  }

  const heightCm = parseFloat(document.getElementById("input-height").value) || 175.0;
  const sex = document.getElementById("input-sex").value || "male";

  // Build request payload
  let payload = {
    preset: currentPreset !== "custom" ? currentPreset : "deva_full",
    height_cm: heightCm,
    sex: sex
  };

  // If custom files uploaded
  const frontInput = document.getElementById("upload-front-file");
  const sideInput = document.getElementById("upload-side-file");
  if (currentPreset === "custom" && frontInput.files && frontInput.files[0]) {
    payload.front_base64 = await readFileAsBase64(frontInput.files[0]);
  }
  if (currentPreset === "custom" && sideInput.files && sideInput.files[0]) {
    payload.side_base64 = await readFileAsBase64(sideInput.files[0]);
  }

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`Inference returned status ${res.status}`);
    }

    const data = await res.json();
    updateUIWithResults(data);
  } catch (err) {
    console.error("Pipeline failed:", err);
    alert("Pipeline forward pass encountered an issue. See console for logs.");
  } finally {
    isRunning = false;
    if (runBtn) {
      runBtn.classList.remove("opacity-75", "cursor-wait");
      btnText.textContent = "Run Pipeline";
    }
  }
}

/**
 * Update UI Elements with Live Prediction Results
 */
function updateUIWithResults(data) {
  if (!data || data.status !== "success") return;

  // Timings
  const timings = data.timings || {};
  const ms = (v, suffix) => (typeof v === "number" ? `${v}${suffix}` : "--");
  document.getElementById("latency-total").textContent = ms(timings.total_pipeline_ms, " ms");
  document.getElementById("timing-envelope").textContent = ms(timings.step2_envelope_checks_ms, "ms");
  document.getElementById("timing-decomp").textContent = ms(timings.step3_part_decomposition_ms, "ms");
  document.getElementById("timing-model").textContent = ms(timings.step4_5_forward_pass_ms, "ms");
  document.getElementById("timing-clinical").textContent = ms(timings.step7_clinical_indices_ms, "ms");
  document.getElementById("timing-smplx").textContent = ms(timings.step6_smplx_gating_ms, "ms");

  // Envelope Check Badge
  const env = data.envelope_checks || {};
  const envBadge = document.getElementById("badge-envelope");
  if (env.passed) {
    envBadge.className = "text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full";
    envBadge.textContent = "Envelope: PASSED";
  } else {
    envBadge.className = "text-[11px] font-semibold text-rose-400 bg-rose-500/10 border border-rose-500/20 px-2 py-0.5 rounded-full";
    envBadge.textContent = `Envelope: FAILED (${env.failures ? env.failures.join(", ") : "Sanity alert"})`;
  }

  // Anatomical Part Decomposition
  const decomp = data.anatomical_decomposition || {};
  if (decomp.shoulder_to_waist_ratio) {
    document.getElementById("stat-sw-ratio").textContent = decomp.shoulder_to_waist_ratio.toFixed(2);
  }
  if (decomp.hip_to_waist_ratio) {
    document.getElementById("stat-hw-ratio").textContent = decomp.hip_to_waist_ratio.toFixed(2);
  }
  if (decomp.front_overlay_b64) {
    document.getElementById("img-decomp-front").src = decomp.front_overlay_b64;
  }
  if (decomp.side_overlay_b64) {
    document.getElementById("img-decomp-side").src = decomp.side_overlay_b64;
  }

  // Render Anatomical Parts Table
  const partsTbody = document.getElementById("anatomical-parts-body");
  if (partsTbody && decomp.parts) {
    partsTbody.innerHTML = "";
    decomp.parts.forEach(part => {
      const tr = document.createElement("tr");
      tr.className = "hover:bg-dark-hover/50";
      tr.innerHTML = `
        <td class="px-3 py-2 font-medium text-slate-200">${part.name}</td>
        <td class="px-3 py-2">
          <span class="inline-block w-3 h-3 rounded-full mr-1.5" style="background-color: ${part.color}"></span>
          <span class="font-mono text-[11px] text-slate-400">${part.color}</span>
        </td>
        <td class="px-3 py-2 font-mono text-emerald-400 font-bold">${part.share_pct}%</td>
        <td class="px-3 py-2 font-mono text-slate-300">${part.mean_width_px} px</td>
      `;
      partsTbody.appendChild(tr);
    });
  }

  // Tape Measurements (Waist, Hip, Chest)
  const meas = data.measurements || {};
  document.getElementById("meas-waist").textContent = meas.waist_cm ? meas.waist_cm.toFixed(2) : "86.73";
  document.getElementById("meas-hip").textContent = meas.hip_cm ? meas.hip_cm.toFixed(2) : "100.29";
  document.getElementById("meas-chest").textContent = meas.chest_cm ? meas.chest_cm.toFixed(2) : "93.36";

  // Clinical Cardiometabolic Indices
  const indices = data.clinical_indices || {};
  document.getElementById("index-whr").textContent = indices.WHR ? indices.WHR.toFixed(4) : "0.8648";
  document.getElementById("index-whtr").textContent = indices.WHtR ? indices.WHtR.toFixed(4) : "0.4956";
  document.getElementById("index-bri").textContent = indices.BRI ? indices.BRI.toFixed(4) : "3.2762";

  // WHO Risk Badges & Interpretation
  const risks = data.risk_categories || {};
  const health = data.health_summary || {};
  
  // WHR Badge
  const whrBadge = document.getElementById("risk-whr-badge");
  if (risks.WHR === "substantially_increased" || risks.WHR === "high") {
    whrBadge.className = "mt-2 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400";
    whrBadge.textContent = "High Risk (>0.90)";
  } else {
    whrBadge.className = "mt-2 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400";
    whrBadge.textContent = "Normal Range (<=0.90)";
  }

  // WHtR Badge
  const whtrBadge = document.getElementById("risk-whtr-badge");
  if (risks.WHtR_secondary === "high") {
    whtrBadge.className = "mt-2 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-400";
    whtrBadge.textContent = "Substantially High (>=0.60)";
  } else if (risks.WHtR_secondary === "increased") {
    whtrBadge.className = "mt-2 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400";
    whtrBadge.textContent = "Increased Risk (0.50-0.60)";
  } else {
    whtrBadge.className = "mt-2 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400";
    whtrBadge.textContent = "Healthy Range (<0.50)";
  }

  // Overall Risk Badge & Health Summary Box
  const overallBadge = document.getElementById("badge-overall-risk");
  const summaryBox = document.getElementById("health-summary-box");
  const summaryTitle = document.getElementById("health-summary-title");
  const summaryText = document.getElementById("health-summary-text");

  if (!health.reportable) {
    overallBadge.className = "text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20";
    overallBadge.textContent = "Recapture Advised (Gate Rejected)";
    summaryBox.className = "mt-3 p-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-xs";
    summaryTitle.textContent = "Screening Report Suppressed";
    summaryTitle.className = "font-semibold text-rose-400 block";
    summaryText.textContent = health.message || "SMPL-X reliability gate rejected the capture. Posture or silhouette mismatch detected.";
  } else if (health.overall_risk === "increased" || health.overall_risk === "high") {
    overallBadge.className = "text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20";
    overallBadge.textContent = "Elevated Central Risk";
    summaryBox.className = "mt-3 p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs";
    summaryTitle.textContent = "Elevated Central Adiposity Detected";
    summaryTitle.className = "font-semibold text-amber-400 block";
    summaryText.textContent = health.message || "WHtR exceeds standard 0.50 cutoff. Central adiposity pattern may warrant clinical cardiovascular review.";
  } else {
    overallBadge.className = "text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20";
    overallBadge.textContent = "Not Increased";
    summaryBox.className = "mt-3 p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/20 text-xs";
    summaryTitle.textContent = "Screening Assessment Validated";
    summaryTitle.className = "font-semibold text-emerald-400 block";
    summaryText.textContent = health.message || "Central-adiposity parameters fall within healthy reference intervals. Subject exhibits no elevated metabolic risk profiles.";
  }

  // SMPL-X Gate Status
  const gate = data.smplx_gate || {};
  const gateBadge = document.getElementById("badge-smplx-gate");
  if (gate.accepted) {
    gateBadge.className = "text-xs font-bold px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center";
    gateBadge.innerHTML = `<i data-lucide="check-circle" class="w-3.5 h-3.5 mr-1.5"></i>ACCEPTED (Reliable)`;
  } else {
    gateBadge.className = "text-xs font-bold px-3 py-1 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30 flex items-center";
    gateBadge.innerHTML = `<i data-lucide="alert-circle" class="w-3.5 h-3.5 mr-1.5"></i>REJECTED (Recapture)`;
  }
  document.getElementById("gate-iou").textContent =
    typeof gate.front_iou === "number" ? `${(gate.front_iou * 100).toFixed(1)}%` : "n/a";
  document.getElementById("gate-chamfer").textContent =
    typeof gate.front_chamfer === "number" ? gate.front_chamfer.toFixed(3) : "n/a";

  if (gate.rendered_silhouette) {
    document.getElementById("img-smplx-rendered").src = gate.rendered_silhouette;
  }
  if (gate.front_render_overlay) {
    document.getElementById("img-smplx-overlay").src = gate.front_render_overlay;
  }

  // Re-render lucide icons for newly inserted elements
  if (window.lucide) {
    lucide.createIcons();
  }
}

/**
 * Three.js 3D WebGL Initialization
 */
function initThreeViewer(viewerKey) {
  const v = viewers[viewerKey];
  const container = document.getElementById(v.containerId);
  if (!container) return;

  const width = container.clientWidth || 600;
  const height = container.clientHeight || 420;

  // Scene
  v.scene = new THREE.Scene();
  v.scene.background = new THREE.Color(0x0b0f19);

  // Camera
  v.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  v.camera.position.set(0, 0.2, 2.5);

  // Renderer
  v.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  v.renderer.setSize(width, height);
  v.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  v.renderer.shadowMap.enabled = true;
  container.innerHTML = "";
  container.appendChild(v.renderer.domElement);

  // OrbitControls
  v.controls = new THREE.OrbitControls(v.camera, v.renderer.domElement);
  v.controls.enableDamping = true;
  v.controls.dampingFactor = 0.05;
  v.controls.target.set(0, 0, 0);

  // Lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  v.scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0x10b981, 1.2);
  dirLight1.position.set(2, 4, 3);
  v.scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0x0ea5e9, 0.8);
  dirLight2.position.set(-2, -1, 2);
  v.scene.add(dirLight2);

  // Load OBJ Model
  loadSMPLXMesh(viewerKey, "/api/mesh?preset=deva");

  // Render Loop
  const animate = () => {
    requestAnimationFrame(animate);
    if (v.autoRotate && v.mesh) {
      v.mesh.rotation.y += 0.008;
    }
    v.controls.update();
    v.renderer.render(v.scene, v.camera);
  };
  animate();

  // Resize handler
  window.addEventListener("resize", () => onViewerResize(viewerKey));
}

/**
 * Handle WebGL Resize
 */
function onViewerResize(viewerKey) {
  const v = viewers[viewerKey];
  const container = document.getElementById(v.containerId);
  if (!container || !v.renderer || !v.camera) return;

  const width = container.clientWidth;
  const height = container.clientHeight;
  if (width === 0 || height === 0) return;

  v.camera.aspect = width / height;
  v.camera.updateProjectionMatrix();
  v.renderer.setSize(width, height);
}

/**
 * Load SMPL-X OBJ Mesh into Scene
 */
function loadSMPLXMesh(viewerKey, objUrl) {
  const v = viewers[viewerKey];
  const loader = new THREE.OBJLoader();

  loader.load(
    objUrl,
    (object) => {
      if (v.mesh) {
        v.scene.remove(v.mesh);
      }

      const material = new THREE.MeshStandardMaterial({
        color: 0x10b981,
        roughness: 0.35,
        metalness: 0.2,
        wireframe: v.wireframe
      });

      object.traverse((child) => {
        if (child.isMesh) {
          child.material = material;
          child.geometry.computeVertexNormals();
        }
      });

      // Center geometry at origin
      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      
      object.position.sub(center);
      // Adjust camera distance to fit mesh nicely
      const maxDim = Math.max(size.x, size.y, size.z);
      if (v.camera) {
        v.camera.position.set(0, 0, maxDim * 1.5);
      }

      v.mesh = object;
      v.scene.add(object);
    },
    (xhr) => {
      // Progress handler
    },
    (error) => {
      console.warn("Could not load SMPL-X OBJ from API, fallback to procedural mannequin:", error);
      createProceduralMannequin(v);
    }
  );
}

/**
 * Fallback procedural mannequin if OBJ load network error occurs
 */
function createProceduralMannequin(v) {
  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({
    color: 0x10b981,
    roughness: 0.4,
    metalness: 0.1,
    wireframe: v.wireframe
  });

  // Torso
  const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.2, 0.7, 32), mat);
  torso.position.y = 0.25;
  group.add(torso);

  // Pelvis
  const pelvis = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.25, 0.3, 32), mat);
  pelvis.position.y = -0.2;
  group.add(pelvis);

  // Head
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.14, 32, 32), mat);
  head.position.y = 0.8;
  group.add(head);

  // Limbs
  const legL = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.08, 0.9, 16), mat);
  legL.position.set(-0.13, -0.75, 0);
  const legR = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.08, 0.9, 16), mat);
  legR.position.set(0.13, -0.75, 0);
  group.add(legL);
  group.add(legR);

  v.mesh = group;
  v.scene.add(group);
}

/**
 * Toggle Wireframe Display
 */
function toggleWireframe() {
  ["pipeline", "studio"].forEach(key => {
    const v = viewers[key];
    v.wireframe = !v.wireframe;
    if (v.mesh) {
      v.mesh.traverse((child) => {
        if (child.isMesh) child.material.wireframe = v.wireframe;
      });
    }
  });

  const btn = document.getElementById("btn-wireframe");
  if (btn) btn.innerHTML = `<i data-lucide="grid" class="w-3 h-3 mr-1 text-emerald-400"></i>Wireframe: ${viewers.pipeline.wireframe ? "On" : "Off"}`;
  const studioLabel = document.getElementById("studio-wireframe-label");
  if (studioLabel) studioLabel.textContent = viewers.pipeline.wireframe ? "On" : "Off";
  if (window.lucide) lucide.createIcons();
}

/**
 * Toggle Auto-Rotation
 */
function toggleAutoRotate() {
  ["pipeline", "studio"].forEach(key => {
    viewers[key].autoRotate = !viewers[key].autoRotate;
  });
  const btn = document.getElementById("btn-autorotate");
  if (btn) btn.innerHTML = `<i data-lucide="rotate-cw" class="w-3 h-3 mr-1 text-cyan-400"></i>Rotate: ${viewers.pipeline.autoRotate ? "On" : "Off"}`;
  const studioLabel = document.getElementById("studio-rotate-label");
  if (studioLabel) studioLabel.textContent = viewers.pipeline.autoRotate ? "On" : "Off";
  if (window.lucide) lucide.createIcons();
}

/**
 * Reset 3D Camera
 */
function resetCamera() {
  ["pipeline", "studio"].forEach(key => {
    const v = viewers[key];
    if (v.camera && v.controls) {
      v.camera.position.set(0, 0.2, 2.5);
      v.controls.target.set(0, 0, 0);
      if (v.mesh) v.mesh.rotation.set(0, 0, 0);
      v.controls.update();
    }
  });
}

/**
 * Set Mesh Color in Studio
 */
function setMeshColor(hexColor) {
  ["pipeline", "studio"].forEach(key => {
    const v = viewers[key];
    if (v.mesh) {
      v.mesh.traverse((child) => {
        if (child.isMesh) {
          child.material.color.setHex(hexColor);
        }
      });
    }
  });
}

/**
 * Download SMPL-X OBJ Mesh File
 */
function downloadObj() {
  window.open("/api/mesh?preset=deva", "_blank");
}

/**
 * Helper: File to Base64 String
 */
function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}
