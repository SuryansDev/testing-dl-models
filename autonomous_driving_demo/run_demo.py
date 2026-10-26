"""
Autonomous Driving Safety Demo: Adversarial Attack on Stop Sign Recognition
Demonstrates how imperceptible perturbations fool vision models in autonomous driving systems.
"""

import os
import sys
import base64
import webbrowser
from io import BytesIO

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path to access torchattacks
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torchattacks

class NormalizedModel(torch.nn.Module):
    """Wraps model to handle ImageNet normalization internally so attacks run on [0, 1] images."""
    def __init__(self, base_model, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        norm_x = (x - self.mean) / self.std
        return self.base_model(norm_x)

def load_system():
    print("[1/5] Loading vision model (ResNet-18 pretrained on ImageNet)...")
    weights = models.ResNet18_Weights.DEFAULT
    base_model = models.resnet18(weights=weights).eval()
    categories = weights.meta["categories"]
    norm_model = NormalizedModel(base_model).eval()
    return norm_model, categories

def load_and_preprocess(image_path):
    print(f"[2/5] Loading traffic sign image: {os.path.basename(image_path)}...")
    img = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor_img = preprocess(img).unsqueeze(0) # [1, 3, 224, 224] in [0, 1]
    return img, tensor_img

def run_adversarial_attack(model, tensor_img, attack_type="PGD", eps=8/255, steps=20):
    print(f"[3/5] Evaluating original prediction...")
    with torch.no_grad():
        clean_logits = model(tensor_img)
        clean_probs = torch.softmax(clean_logits, dim=1)
        clean_top_probs, clean_top_idxs = torch.topk(clean_probs, 5)

    original_label = clean_top_idxs[0][0].unsqueeze(0)

    print(f"[4/5] Crafting adversarial perturbation using {attack_type} (epsilon={eps:.4f} = {int(eps*255)}/255)...")
    if attack_type.upper() == "PGD":
        atk = torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=steps, random_start=True)
    elif attack_type.upper() == "FGSM":
        atk = torchattacks.FGSM(model, eps=eps)
    elif attack_type.upper() == "BIM":
        atk = torchattacks.BIM(model, eps=eps, alpha=eps/4, steps=steps)
    else:
        atk = torchattacks.PGD(model, eps=eps, alpha=eps/4, steps=steps)

    adv_img = atk(tensor_img, original_label)

    with torch.no_grad():
        adv_logits = model(adv_img)
        adv_probs = torch.softmax(adv_logits, dim=1)
        adv_top_probs, adv_top_idxs = torch.topk(adv_probs, 5)

    return clean_top_probs[0].tolist(), clean_top_idxs[0].tolist(), adv_img, adv_top_probs[0].tolist(), adv_top_idxs[0].tolist()

def generate_visual_artifacts(clean_tensor, adv_img, categories, clean_probs, clean_idxs, adv_probs, adv_idxs, output_dir):
    print("[5/5] Generating side-by-side visualization and HTML report...")
    
    clean_np = clean_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = adv_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    
    # Perturbation amplified
    perturbation = adv_np - clean_np
    perturbation_amplified = np.clip((perturbation * 15.0) + 0.5, 0.0, 1.0)
    
    fig, (top_row, bot_row) = plt.subplots(2, 2, figsize=(13, 9), facecolor="#0f172a")
    
    # We can use a 2x3 GridSpec for clean layout
    fig.clf()
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1.0])
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(clean_np)
    ax1.set_title(f"CLEAN INPUT\n{categories[clean_idxs[0]].upper()} ({clean_probs[0]*100:.1f}%)", 
                  color="#22c55e", fontsize=11, fontweight="bold", pad=8)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(perturbation_amplified)
    max_linf = np.max(np.abs(perturbation))
    l2_norm = np.linalg.norm(perturbation)
    ax2.set_title(f"ADVERSARIAL NOISE (15x)\nMax L-inf: {max_linf*255:.1f}/255 | L2: {l2_norm:.2f}", 
                  color="#f59e0b", fontsize=11, fontweight="bold", pad=8)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(adv_np)
    ax3.set_title(f"ADVERSARIAL ATTACK\nFOOLED: {categories[adv_idxs[0]].upper()} ({adv_probs[0]*100:.1f}%)", 
                  color="#ef4444", fontsize=11, fontweight="bold", pad=8)
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 0:2])
    clean_labels = [categories[idx][:22] for idx in clean_idxs[:5]]
    y_pos = np.arange(len(clean_labels))
    ax4.barh(y_pos, [p * 100 for p in clean_probs[:5]], color="#22c55e", alpha=0.85)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(clean_labels, color="#e2e8f0", fontsize=10)
    ax4.invert_yaxis()
    ax4.set_xlabel("Confidence (%)", color="#94a3b8")
    ax4.set_title("Original Model Top Predictions", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax4.set_xlim(0, 105)
    ax4.set_facecolor("#1e293b")
    ax4.tick_params(colors="#94a3b8")

    ax5 = fig.add_subplot(gs[1, 2])
    adv_labels = [categories[idx][:18] for idx in adv_idxs[:5]]
    y_pos2 = np.arange(len(adv_labels))
    ax5.barh(y_pos2, [p * 100 for p in adv_probs[:5]], color="#ef4444", alpha=0.85)
    ax5.set_yticks(y_pos2)
    ax5.set_yticklabels(adv_labels, color="#e2e8f0", fontsize=10)
    ax5.invert_yaxis()
    ax5.set_xlabel("Confidence (%)", color="#94a3b8")
    ax5.set_title("Post-Attack Top Predictions", color="#e2e8f0", fontsize=12, fontweight="bold")
    ax5.set_xlim(0, 105)
    ax5.set_facecolor("#1e293b")
    ax5.tick_params(colors="#94a3b8")

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
    chart_path = os.path.join(output_dir, "attack_comparison.png")
    plt.savefig(chart_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"-> Saved visual comparison chart: {chart_path}")

    # Convert images to base64 for standalone HTML report
    def to_b64(img_array):
        buf = BytesIO()
        pil_im = Image.fromarray((img_array * 255).astype(np.uint8))
        pil_im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    clean_b64 = to_b64(clean_np)
    adv_b64 = to_b64(adv_np)
    pert_b64 = to_b64(perturbation_amplified)

    # Generate interactive HTML report
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Driving Adversarial Attack Demo</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --surface-hover: #1f2937;
            --border: #2d3748;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background: var(--bg); color: var(--text-primary); line-height: 1.6; padding: 2rem; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{ text-align: center; margin-bottom: 2.5rem; }}
        header h1 {{ font-size: 2.25rem; font-weight: 800; background: linear-gradient(135deg, #ef4444, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }}
        header p {{ color: var(--text-secondary); font-size: 1.1rem; max-width: 750px; margin: 0 auto; }}
        .alert-box {{ background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 12px; padding: 1.25rem; margin-bottom: 2rem; display: flex; align-items: center; gap: 1rem; }}
        .alert-icon {{ font-size: 2rem; }}
        .grid-3 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; margin-bottom: 2.5rem; }}
        .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; box-shadow: 0 10px 25px rgba(0,0,0,0.4); }}
        .card-header {{ padding: 1rem 1.25rem; font-weight: 700; font-size: 1.05rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); }}
        .tag-safe {{ color: var(--accent-green); background: rgba(16, 185, 129, 0.15); padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.85rem; }}
        .tag-danger {{ color: var(--accent-red); background: rgba(239, 68, 68, 0.15); padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.85rem; }}
        .tag-warn {{ color: var(--accent-amber); background: rgba(245, 158, 11, 0.15); padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.85rem; }}
        .card-img-wrap {{ width: 100%; aspect-ratio: 1; position: relative; background: #000; display: flex; align-items: center; justify-content: center; }}
        .card-img-wrap img {{ width: 100%; height: 100%; object-fit: cover; }}
        .card-body {{ padding: 1.25rem; }}
        .metric-title {{ font-size: 0.9rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }}
        .metric-val {{ font-size: 1.4rem; font-weight: 800; margin-bottom: 0.5rem; }}
        .interactive-compare {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 2rem; margin-bottom: 2.5rem; text-align: center; }}
        .interactive-compare h2 {{ margin-bottom: 1rem; font-size: 1.4rem; }}
        .compare-slider-container {{ position: relative; max-width: 450px; aspect-ratio: 1; margin: 0 auto 1.5rem; overflow: hidden; border-radius: 12px; border: 2px solid var(--border); }}
        .compare-img {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; }}
        .compare-overlay {{ position: absolute; top: 0; left: 0; width: 50%; height: 100%; overflow: hidden; border-right: 3px solid #38bdf8; }}
        .compare-overlay img {{ width: 450px; height: 450px; max-width: none; object-fit: cover; }}
        .slider-range {{ width: 100%; max-width: 450px; cursor: pointer; }}
        .takeaway-section {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem; }}
        .takeaway-section h3 {{ margin-bottom: 0.75rem; color: var(--accent-amber); }}
        ul {{ padding-left: 1.5rem; color: var(--text-secondary); }}
        li {{ margin-bottom: 0.5rem; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Autonomous Driving Threat Demo: Stop Sign Hijack</h1>
            <p>Demonstrating how imperceptible mathematical perturbations in computer vision models can lead to catastrophic misclassifications in autonomous vehicle perception.</p>
        </header>

        <div class="alert-box">
            <div class="alert-icon">⚠️</div>
            <div>
                <strong>Autonomous Safety Hazard:</strong> The vehicle vision system identifies the input stop sign as a 
                <span style="color: var(--accent-green); font-weight: bold;">"{categories[clean_idxs[0]]}" ({clean_probs[0]*100:.1f}%)</span> under normal conditions. 
                With an imperceptible L-inf perturbation of &epsilon; = {int(max_linf*255)}/255, the classifier shifts entirely to 
                <span style="color: var(--accent-red); font-weight: bold;">"{categories[adv_idxs[0]]}" ({adv_probs[0]*100:.1f}%)</span>.
            </div>
        </div>

        <div class="grid-3">
            <div class="card">
                <div class="card-header">
                    <span>1. Clean Image</span>
                    <span class="tag-safe">Normal View</span>
                </div>
                <div class="card-img-wrap">
                    <img src="data:image/png;base64,{clean_b64}" alt="Clean Stop Sign">
                </div>
                <div class="card-body">
                    <div class="metric-title">Detected Label</div>
                    <div class="metric-val" style="color: var(--accent-green);">{categories[clean_idxs[0]].title()}</div>
                    <p style="color: var(--text-secondary);">Model Confidence: <strong>{clean_probs[0]*100:.1f}%</strong></p>
                    <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.5rem;">Vehicle action: Decelerates and stops safely at junction.</p>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span>2. Adversarial Perturbation</span>
                    <span class="tag-warn">&times;15 Amplified</span>
                </div>
                <div class="card-img-wrap">
                    <img src="data:image/png;base64,{pert_b64}" alt="Perturbation Pattern">
                </div>
                <div class="card-body">
                    <div class="metric-title">Attack Parameters</div>
                    <div class="metric-val" style="color: var(--accent-amber);">&epsilon; = {max_linf*255:.1f} / 255</div>
                    <p style="color: var(--text-secondary);">L2 Distance: <strong>{l2_norm:.2f}</strong></p>
                    <p style="color: #64748b; font-size: 0.85rem; margin-top: 0.5rem;">Engineered noise maximizes model cross-entropy loss along gradient direction.</p>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span>3. Adversarial Result</span>
                    <span class="tag-danger">Attack Success</span>
                </div>
                <div class="card-img-wrap">
                    <img src="data:image/png;base64,{adv_b64}" alt="Adversarial Image">
                </div>
                <div class="card-body">
                    <div class="metric-title">Hijacked Label</div>
                    <div class="metric-val" style="color: var(--accent-red);">{categories[adv_idxs[0]].title()}</div>
                    <p style="color: var(--text-secondary);">Model Confidence: <strong>{adv_probs[0]*100:.1f}%</strong></p>
                    <p style="color: #ef4444; font-size: 0.85rem; margin-top: 0.5rem;">Vehicle action: Fails to detect stop sign, ignores junction rules!</p>
                </div>
            </div>
        </div>

        <div class="interactive-compare">
            <h2>Interactive Pixel Comparison Slider</h2>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem;">Drag the slider below to reveal that the clean image and adversarial image are visually indistinguishable to human eyes:</p>
            <div class="compare-slider-container">
                <img class="compare-img" src="data:image/png;base64,{adv_b64}" alt="Adversarial">
                <div class="compare-overlay" id="overlay">
                    <img src="data:image/png;base64,{clean_b64}" alt="Clean">
                </div>
            </div>
            <input type="range" class="slider-range" id="slider" min="0" max="100" value="50">
            <p style="font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">← Clean View | Adversarial View →</p>
        </div>

        <div class="takeaway-section">
            <h3>Key Takeaways & Real-World Solutions:</h3>
            <ul>
                <li><strong>Human Perception vs Deep Learning:</strong> While biological eyes easily recognize the octagonal geometry and contrast, standard convolutional networks rely on high-frequency textural features that can be exploited.</li>
                <li><strong>Physical-World Threats:</strong> Researchers have demonstrated that these perturbations can be printed as physical stickers or graffiti on actual traffic signs, blinding self-driving cars.</li>
                <li><strong>Defenses Available:</strong> Using this repository's <code>robustbench</code> suite, models can be fortified using Adversarial Training (TRADES, PGD-AT) to resist these attacks.</li>
            </ul>
        </div>
    </div>

    <script>
        const slider = document.getElementById('slider');
        const overlay = document.getElementById('overlay');
        slider.addEventListener('input', (e) => {{
            overlay.style.width = e.target.value + '%';
        }});
    </script>
</body>
</html>
"""

    report_path = os.path.join(output_dir, "interactive_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"-> Saved interactive HTML report: {report_path}")
    return chart_path, report_path

def main():
    print("=" * 70)
    print(" AUTONOMOUS DRIVING SAFETY DEMO: STOP SIGN ADVERSARIAL ATTACK")
    print("=" * 70)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_img_path = os.path.join(current_dir, "sample_images", "stop_sign_closeup.jpg")
    
    model, categories = load_system()
    pil_img, tensor_img = load_and_preprocess(sample_img_path)
    
    clean_probs, clean_idxs, adv_img, adv_probs, adv_idxs = run_adversarial_attack(
        model, tensor_img, attack_type="PGD", eps=8/255, steps=20
    )
    
    chart_path, report_path = generate_visual_artifacts(
        tensor_img, adv_img, categories, clean_probs, clean_idxs, adv_probs, adv_idxs, current_dir
    )
    
    print("\n" + "=" * 70)
    print("ATTACK SUMMARY RESULTS:")
    print(f"  * Original Prediction : {categories[clean_idxs[0]].upper()} ({clean_probs[0]*100:.2f}%)")
    print(f"  * Adversarial Hijack  : {categories[adv_idxs[0]].upper()} ({adv_probs[0]*100:.2f}%)")
    print(f"  * Perturbation Metric : L-infinity = 8/255 (Imperceptible to human eyes)")
    print(f"  * Comparison Chart    : {chart_path}")
    print(f"  * Interactive Report  : {report_path}")
    print("=" * 70)

    # Attempt to open report in default browser
    try:
        webbrowser.open("file://" + os.path.abspath(report_path))
    except Exception:
        pass

if __name__ == "__main__":
    main()
