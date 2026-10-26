"""
Interactive Web Application: Autonomous Driving Adversarial Attack Simulator
Runs locally with zero external web framework dependencies (built on Python's http.server).
"""

import os
import sys
import json
import base64
import time
import webbrowser
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Ensure root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torchattacks

class NormalizedModel(torch.nn.Module):
    def __init__(self, base_model, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        norm_x = (x - self.mean) / self.std
        return self.base_model(norm_x)

print("Initializing Vision Model (ResNet-18)...")
weights = models.ResNet18_Weights.DEFAULT
base_model = models.resnet18(weights=weights).eval()
categories = weights.meta["categories"]
model = NormalizedModel(base_model).eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")
SAMPLE_MAP = {
    "closeup": os.path.join(SAMPLE_DIR, "stop_sign_closeup.jpg"),
    "dashcam": os.path.join(SAMPLE_DIR, "dashcam_stop_sign.jpg"),
}

def to_base64(numpy_img):
    buf = BytesIO()
    clamped = np.clip(numpy_img * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(clamped).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def execute_attack(image_source, attack_type="PGD", eps=8/255, steps=15):
    t0 = time.time()
    
    # Load Image
    if image_source in SAMPLE_MAP:
        img_path = SAMPLE_MAP[image_source]
        pil_img = Image.open(img_path).convert("RGB")
    elif image_source.startswith("data:image"):
        data = image_source.split(",")[1]
        pil_img = Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    else:
        pil_img = Image.open(SAMPLE_MAP["closeup"]).convert("RGB")

    tensor_img = preprocess(pil_img).unsqueeze(0)

    # Clean Prediction
    with torch.no_grad():
        clean_logits = model(tensor_img)
        clean_probs = torch.softmax(clean_logits, dim=1)
        clean_top_probs, clean_top_idxs = torch.topk(clean_probs, 5)

    original_label = clean_top_idxs[0][0].unsqueeze(0)

    # Select attack
    attack_name = attack_type.upper()
    if attack_name == "PGD":
        atk = torchattacks.PGD(model, eps=eps, alpha=max(eps/4, 1/255), steps=steps, random_start=True)
    elif attack_name == "FGSM":
        atk = torchattacks.FGSM(model, eps=eps)
    elif attack_name == "BIM":
        atk = torchattacks.BIM(model, eps=eps, alpha=max(eps/4, 1/255), steps=steps)
    elif attack_name == "CW":
        atk = torchattacks.CW(model, c=1, steps=steps, lr=0.05)
    elif attack_name == "DEEPFOOL":
        atk = torchattacks.DeepFool(model, steps=min(steps, 20))
    else:
        atk = torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=steps)

    adv_tensor = atk(tensor_img, original_label)

    # Post-attack prediction
    with torch.no_grad():
        adv_logits = model(adv_tensor)
        adv_probs = torch.softmax(adv_logits, dim=1)
        adv_top_probs, adv_top_idxs = torch.topk(adv_probs, 5)

    clean_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    
    perturbation = adv_np - clean_np
    perturbation_amplified = (perturbation * 15.0) + 0.5

    clean_preds = [
        {"label": categories[clean_top_idxs[0][i].item()].title(), "prob": float(clean_top_probs[0][i].item())}
        for i in range(5)
    ]
    adv_preds = [
        {"label": categories[adv_top_idxs[0][i].item()].title(), "prob": float(adv_top_probs[0][i].item())}
        for i in range(5)
    ]

    max_linf = float(np.max(np.abs(perturbation)))
    l2_dist = float(np.linalg.norm(perturbation))
    elapsed = time.time() - t0

    return {
        "clean_image": to_base64(clean_np),
        "adv_image": to_base64(adv_np),
        "pert_image": to_base64(perturbation_amplified),
        "clean_preds": clean_preds,
        "adv_preds": adv_preds,
        "metrics": {
            "attack_type": attack_name,
            "epsilon_255": round(max_linf * 255, 2),
            "l2_distance": round(l2_dist, 3),
            "time_seconds": round(elapsed, 2)
        }
    }

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Vision Threat Lab: Stop Sign Hijack</title>
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --border: #1f293d;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --accent-cyan: #06b6d4;
            --text-main: #f8fafc;
            --text-sub: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg-color); color: var(--text-main); line-height: 1.5; min-height: 100vh; padding: 2rem 1.5rem; }
        .wrapper { max-width: 1300px; margin: 0 auto; }
        
        /* Header */
        header { text-align: center; margin-bottom: 2rem; }
        .badge { display: inline-block; padding: 0.35rem 0.85rem; border-radius: 9999px; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.75rem; }
        h1 { font-size: 2.3rem; font-weight: 800; background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        header p { color: var(--text-sub); font-size: 1.05rem; margin-top: 0.5rem; max-width: 750px; margin-inline: auto; }

        /* Main Grid */
        .layout { display: grid; grid-template-columns: 340px 1fr; gap: 2rem; }
        @media (max-width: 960px) { .layout { grid-template-columns: 1fr; } }

        /* Sidebar Controls */
        .panel { background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .section-title { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem; }
        .form-group { margin-bottom: 1.25rem; }
        label { display: block; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; color: var(--text-sub); margin-bottom: 0.5rem; }
        select, input[type="file"] { width: 100%; background: #0b0f19; border: 1px solid var(--border); color: #fff; padding: 0.75rem; border-radius: 10px; font-size: 0.95rem; outline: none; }
        select:focus { border-color: var(--accent-cyan); }
        .slider-val { float: right; color: var(--accent-cyan); font-weight: 700; }
        input[type="range"] { width: 100%; accent-color: var(--accent-cyan); margin-top: 0.4rem; }
        .btn-launch { width: 100%; background: linear-gradient(135deg, #ef4444, #dc2626); color: #fff; border: none; padding: 0.9rem; font-size: 1rem; font-weight: 700; border-radius: 10px; cursor: pointer; transition: all 0.2s; display: flex; justify-content: center; align-items: center; gap: 0.5rem; margin-top: 1rem; }
        .btn-launch:hover { background: linear-gradient(135deg, #dc2626, #b91c1c); transform: translateY(-1px); box-shadow: 0 6px 20px rgba(239,68,68,0.4); }
        .btn-launch:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Results Area */
        .status-banner { border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
        .status-danger { background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); }
        .status-safe { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.35); }

        /* Visual Inspection 3-Grid */
        .visual-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.25rem; margin-bottom: 2rem; }
        @media (max-width: 800px) { .visual-grid { grid-template-columns: 1fr; } }
        .vis-card { background: #0b0f19; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
        .vis-header { padding: 0.75rem 1rem; font-size: 0.85rem; font-weight: 700; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; }
        .vis-img-box { aspect-ratio: 1; width: 100%; background: #000; display: flex; align-items: center; justify-content: center; position: relative; }
        .vis-img-box img { width: 100%; height: 100%; object-fit: cover; }
        .vis-footer { padding: 0.85rem 1rem; }
        .vis-pred { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.25rem; }
        .vis-sub { font-size: 0.85rem; color: var(--text-sub); }

        /* Split Comparison Slider */
        .slider-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem; }
        .split-wrap { position: relative; max-width: 440px; aspect-ratio: 1; margin: 1rem auto; overflow: hidden; border-radius: 12px; border: 2px solid var(--border); }
        .split-wrap img { position: absolute; top:0; left:0; width:100%; height:100%; object-fit: cover; }
        .split-clip { position: absolute; top:0; left:0; width: 50%; height:100%; overflow: hidden; border-right: 3px solid var(--accent-cyan); }
        .split-clip img { width: 440px; height: 440px; max-width: none; }

        /* Prediction Comparison Bar Chart */
        .bars-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        @media (max-width: 700px) { .bars-grid { grid-template-columns: 1fr; } }
        .bar-row { margin-bottom: 0.75rem; }
        .bar-label-line { display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.25rem; }
        .bar-track { width: 100%; height: 10px; background: #1e293b; border-radius: 999px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
        .bar-clean { background: var(--accent-green); }
        .bar-adv { background: var(--accent-red); }

        /* Loader */
        .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.3); border-radius: 50%; border-top-color: #fff; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="wrapper">
        <header>
            <div class="badge">Autonomous Safety Lab</div>
            <h1>Stop Sign Adversarial Attack Simulator</h1>
            <p>Experience how state-of-the-art vision models are completely subverted by imperceptible perturbations crafted with Torchattacks.</p>
        </header>

        <div class="layout">
            <!-- Left Controls -->
            <aside class="panel">
                <div class="section-title">⚙️ Attack Configuration</div>
                
                <div class="form-group">
                    <label>Target Image</label>
                    <select id="imageSelect">
                        <option value="closeup">Stop Sign (Close-up Photo)</option>
                        <option value="dashcam">Stop Sign (Autonomous Dashcam View)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Attack Algorithm</label>
                    <select id="attackSelect">
                        <option value="PGD" selected>PGD (Projected Gradient Descent)</option>
                        <option value="FGSM">FGSM (Fast Gradient Sign Method)</option>
                        <option value="BIM">BIM (Basic Iterative Method)</option>
                        <option value="CW">CW (Carlini & Wagner L2)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Perturbation Bound (&epsilon;)</label>
                    <div><span class="slider-val" id="epsVal">8 / 255</span></div>
                    <input type="range" id="epsRange" min="1" max="24" value="8">
                </div>

                <div class="form-group">
                    <label>Attack Iterations / Steps</label>
                    <div><span class="slider-val" id="stepsVal">15</span></div>
                    <input type="range" id="stepsRange" min="5" max="30" value="15">
                </div>

                <button class="btn-launch" id="runBtn" onclick="runAttack()">
                    <span id="btnText">⚡ Launch Attack</span>
                </button>

                <div style="margin-top: 1.5rem; padding-top: 1.25rem; border-top: 1px solid var(--border); font-size: 0.85rem; color: var(--text-sub);">
                    <strong style="color: #cbd5e1;">How it works:</strong> The algorithm computes gradients of the loss function with respect to the input pixels and adds subtle adversarial directions that blind the neural network.
                </div>
            </aside>

            <!-- Main Interactive Display -->
            <main>
                <!-- Autonomous Threat Status Banner -->
                <div class="status-banner status-danger" id="threatBanner">
                    <div>
                        <strong style="font-size: 1.05rem;" id="threatTitle">🚨 CRITICAL PERCEPTION FAILURE: Stop Sign Blindness</strong>
                        <p style="font-size: 0.9rem; color: #fca5a5; margin-top: 0.2rem;" id="threatDesc">
                            The autonomous vehicle fails to detect the Stop Sign and misidentifies the obstacle as a Mailbox. The vehicle would not stop at this intersection!
                        </p>
                    </div>
                </div>

                <!-- 3-Panel Inspection -->
                <div class="visual-grid">
                    <!-- Clean Panel -->
                    <div class="vis-card">
                        <div class="vis-header">
                            <span style="color: var(--accent-green);">1. Clean Image</span>
                            <span style="color: var(--text-sub);">Normal Feed</span>
                        </div>
                        <div class="vis-img-box">
                            <img id="cleanImg" src="" alt="Clean image">
                        </div>
                        <div class="vis-footer">
                            <div class="vis-pred" style="color: var(--accent-green);" id="cleanTopPred">Loading...</div>
                            <div class="vis-sub" id="cleanTopConf">Confidence: --%</div>
                        </div>
                    </div>

                    <!-- Perturbation Noise Panel -->
                    <div class="vis-card">
                        <div class="vis-header">
                            <span style="color: var(--accent-amber);">2. Adversarial Noise</span>
                            <span style="color: var(--accent-amber);" id="noiseMetric">&times;15 Amplified</span>
                        </div>
                        <div class="vis-img-box">
                            <img id="pertImg" src="" alt="Adversarial noise">
                        </div>
                        <div class="vis-footer">
                            <div class="vis-pred" style="color: var(--accent-amber);" id="noiseEps">&epsilon; = 8.0/255</div>
                            <div class="vis-sub" id="noiseL2">L2 Norm: --</div>
                        </div>
                    </div>

                    <!-- Adversarial Result Panel -->
                    <div class="vis-card">
                        <div class="vis-header">
                            <span style="color: var(--accent-red);">3. Adversarial Result</span>
                            <span style="color: var(--accent-red);">Hijacked Vision</span>
                        </div>
                        <div class="vis-img-box">
                            <img id="advImg" src="" alt="Adversarial image">
                        </div>
                        <div class="vis-footer">
                            <div class="vis-pred" style="color: var(--accent-red);" id="advTopPred">Loading...</div>
                            <div class="vis-sub" id="advTopConf">Confidence: --%</div>
                        </div>
                    </div>
                </div>

                <!-- Split-screen slider -->
                <div class="slider-card">
                    <h2 style="font-size: 1.15rem; font-weight: 700; margin-bottom: 0.5rem;">Interactive Pixel Reveal Lens</h2>
                    <p style="font-size: 0.9rem; color: var(--text-sub);">Slide back and forth to inspect that to biological human vision, the image appears 100% unaltered.</p>
                    
                    <div class="split-wrap">
                        <img id="splitAdv" src="" alt="Adversarial">
                        <div class="split-clip" id="splitClip">
                            <img id="splitClean" src="" alt="Clean">
                        </div>
                    </div>
                    <div style="max-width: 440px; margin: 0 auto; text-align: center;">
                        <input type="range" id="splitSlider" min="0" max="100" value="50">
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-sub); margin-top: 0.25rem;">
                            <span>← Clean Image</span>
                            <span>Adversarial Image →</span>
                        </div>
                    </div>
                </div>

                <!-- Prediction Bars -->
                <div class="panel">
                    <div class="section-title">📊 Top-5 Predictions Distribution Comparison</div>
                    <div class="bars-grid">
                        <div>
                            <h3 style="font-size: 0.95rem; color: var(--accent-green); margin-bottom: 1rem;">Clean Image Prediction</h3>
                            <div id="cleanBars"></div>
                        </div>
                        <div>
                            <h3 style="font-size: 0.95rem; color: var(--accent-red); margin-bottom: 1rem;">Adversarial Image Prediction</h3>
                            <div id="advBars"></div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    </div>

    <script>
        const epsRange = document.getElementById('epsRange');
        const epsVal = document.getElementById('epsVal');
        const stepsRange = document.getElementById('stepsRange');
        const stepsVal = document.getElementById('stepsVal');
        const splitSlider = document.getElementById('splitSlider');
        const splitClip = document.getElementById('splitClip');

        epsRange.addEventListener('input', (e) => {
            epsVal.textContent = e.target.value + ' / 255';
        });

        stepsRange.addEventListener('input', (e) => {
            stepsVal.textContent = e.target.value;
        });

        splitSlider.addEventListener('input', (e) => {
            splitClip.style.width = e.target.value + '%';
        });

        function renderBars(containerId, preds, isClean) {
            const container = document.getElementById(containerId);
            container.innerHTML = '';
            preds.forEach(p => {
                const percent = (p.prob * 100).toFixed(1);
                const fillClass = isClean ? 'bar-clean' : 'bar-adv';
                const row = document.createElement('div');
                row.className = 'bar-row';
                row.innerHTML = `
                    <div class="bar-label-line">
                        <span>${p.label}</span>
                        <strong style="color: ${isClean ? '#10b981' : '#ef4444'}">${percent}%</strong>
                    </div>
                    <div class="bar-track">
                        <div class="bar-fill ${fillClass}" style="width: ${percent}%"></div>
                    </div>
                `;
                container.appendChild(row);
            });
        }

        async function runAttack() {
            const btn = document.getElementById('runBtn');
            const btnText = document.getElementById('btnText');
            btn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span> Running Attack...';

            const payload = {
                image_source: document.getElementById('imageSelect').value,
                attack_type: document.getElementById('attackSelect').value,
                eps: parseFloat(epsRange.value) / 255.0,
                steps: parseInt(stepsRange.value)
            };

            try {
                const res = await fetch('/api/attack', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();

                // Update Images
                const cleanSrc = 'data:image/png;base64,' + data.clean_image;
                const advSrc = 'data:image/png;base64,' + data.adv_image;
                const pertSrc = 'data:image/png;base64,' + data.pert_image;

                document.getElementById('cleanImg').src = cleanSrc;
                document.getElementById('advImg').src = advSrc;
                document.getElementById('pertImg').src = pertSrc;
                document.getElementById('splitClean').src = cleanSrc;
                document.getElementById('splitAdv').src = advSrc;

                // Update Labels
                const topClean = data.clean_preds[0];
                const topAdv = data.adv_preds[0];

                document.getElementById('cleanTopPred').textContent = topClean.label;
                document.getElementById('cleanTopConf').textContent = `Confidence: ${(topClean.prob * 100).toFixed(1)}%`;

                document.getElementById('advTopPred').textContent = topAdv.label;
                document.getElementById('advTopConf').textContent = `Confidence: ${(topAdv.prob * 100).toFixed(1)}%`;

                document.getElementById('noiseEps').textContent = `&epsilon; = ${data.metrics.epsilon_255}/255`;
                document.getElementById('noiseL2').textContent = `L2 Norm: ${data.metrics.l2_distance} | ${data.metrics.time_seconds}s`;

                // Update Threat Banner
                const threatTitle = document.getElementById('threatTitle');
                const threatDesc = document.getElementById('threatDesc');
                if (topAdv.label.toLowerCase() !== topClean.label.toLowerCase()) {
                    threatTitle.textContent = `🚨 CRITICAL SAFETY THREAT: Blind to Stop Sign`;
                    threatDesc.textContent = `Autonomous vehicle model hijacked from "${topClean.label}" to "${topAdv.label}". The vehicle perception layer failed to identify the regulatory sign!`;
                    document.getElementById('threatBanner').className = 'status-banner status-danger';
                } else {
                    threatTitle.textContent = `🛡️ Attack Resisted`;
                    threatDesc.textContent = `Model maintained "${topClean.label}" prediction despite perturbation.`;
                    document.getElementById('threatBanner').className = 'status-banner status-safe';
                }

                // Render Bars
                renderBars('cleanBars', data.clean_preds, true);
                renderBars('advBars', data.adv_preds, false);

            } catch (err) {
                alert('Attack execution failed: ' + err.message);
            } finally {
                btn.disabled = false;
                btnText.innerHTML = '⚡ Launch Attack';
            }
        }

        // Run automatically once on page load
        window.onload = () => {
            runAttack();
        };
    </script>
</body>
</html>
"""

class ThreatLabHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet standard HTTP request logs
        return

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/attack":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            req = json.loads(body)
            
            img_source = req.get("image_source", "closeup")
            atk_type = req.get("attack_type", "PGD")
            eps = float(req.get("eps", 8/255))
            steps = int(req.get("steps", 15))

            result = execute_attack(img_source, attack_type=atk_type, eps=eps, steps=steps)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8501):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, ThreatLabHandler)
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 70)
    print(" 🚗 AUTONOMOUS DRIVING THREAT LAB IS RUNNING!")
    print(f" URL: {url}")
    print(" Press Ctrl+C in terminal to stop.")
    print("=" * 70 + "\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == "__main__":
    port = 8501
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
