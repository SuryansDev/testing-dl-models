# 🚗 Autonomous Driving Safety Threat: Stop Sign Adversarial Attack Demo

This demo demonstrates how adversarial attacks can compromise deep learning perception models in autonomous driving systems.

Using state-of-the-art attacks from the `torchattacks` library, an imperceptible mathematical perturbation is applied to an image of a **Stop Sign**. While to human eyes the sign looks completely unchanged, the deep learning vision model (ResNet-18) flips from predicting **"Street Sign" (94.5%)** to **"Mailbox" (100.0%)**, blinding the vehicle's navigation system.

---

## ⚡ Quick Start (Two Ways to Run)

### Method 1: Instant Visual Report & Image Generation
Simply double-click `run_demo.bat` or run:
```bash
python run_demo.py
```
**What happens:**
1. Loads the vision model and target stop sign image.
2. Crafts a PGD (Projected Gradient Descent) adversarial perturbation ($\epsilon = 8/255$).
3. Generates a high-resolution visual comparison image: `attack_comparison.png`.
4. Automatically opens `interactive_report.html` in your default browser, complete with an interactive pixel-scrubbing slider and safety hazard metrics.

---

### Method 2: Live Interactive Web Simulator
Double-click `run_web_ui.bat` or run:
```bash
python web_app.py
```
Then visit **`http://127.0.0.1:8501`** in your browser.

**Interactive Features:**
- **Select Image**: Test between close-up stop sign and full autonomous vehicle dashcam perspective.
- **Select Attack Algorithm**: PGD, FGSM, BIM, CW (Carlini & Wagner).
- **Perturbation Slider**: Adjust $\epsilon$ from $1/255$ up to $24/255$.
- **Live Pixel Reveal Lens**: Drag the slider left and right to inspect the seamless pixel match between clean and adversarial images.
- **Real-Time Probability Bars**: Observe the confidence distribution dynamically shift from safety to failure.

---

## 📁 Directory Contents

| File | Description |
| :--- | :--- |
| `run_demo.py` | Standalone script to execute the attack, output metrics, and generate reports. |
| `web_app.py` | Zero-dependency local web dashboard powered by Python's built-in `http.server`. |
| `run_demo.bat` | 1-Click Windows launcher for `run_demo.py`. |
| `run_web_ui.bat` | 1-Click Windows launcher for `web_app.py`. |
| `interactive_report.html` | Self-contained HTML report with embedded visuals and comparison slider. |
| `attack_comparison.png` | 5-panel infographic showing clean input, magnified noise, adversarial input, and before/after logits. |
| `sample_images/` | Pre-bundled high-resolution dashcam and close-up stop sign photographs. |

---

## 🛡️ Autonomous Vehicle Implications

In safety-critical applications like automated driving:
- **Catastrophic Failure**: If an attacker applies an adversarial sticker or perturbation to a physical road sign, the car fails to stop at the intersection.
- **Mitigation**: Using the `robustbench` suite included in this repository, vision models can be hardened through adversarial training (e.g. TRADES, PGD-AT) to remain resilient against perturbations.
