<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=GAN%20Experiments&fontSize=72&fontColor=ffffff&fontAlignY=35&desc=Generative%20Adversarial%20Networks%20%E2%80%94%20From%20Baseline%20to%20State-of-the-Art&descAlignY=58&descSize=18&animation=fadeIn" alt="header"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-T4%20×2%20GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Deployed-FFD21E?style=for-the-badge)](https://huggingface.co/Faiezeee)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>


![](https://i.imgur.com/waxVImv.png)

> **Three production-ready GAN experiments** — covering unconditional generation, paired image translation, and unpaired domain adaptation — each trained on Kaggle T4×2 GPUs and deployed as live interactive Gradio apps on HuggingFace Spaces.

<br/>

```
 ┌─────────────────────────────────────────────────────────────────┐
 │   Q1 · DCGAN / WGAN-GP    →    Anime Face Generation           │
 │   Q2 · Pix2Pix            →    Sketch to Realistic Photo       │
 │   Q3 · CycleGAN           →    Unpaired Sketch ↔ Photo         │
 └─────────────────────────────────────────────────────────────────┘
```

</div>

---

## 🗺 Repository Map

```
GAN_Experiments/
│
├── 📓 Q1_DCGAN_WGANGP/
│   ├── GAN_DCGAN_WGANGP.ipynb        ← Full training notebook
│   ├── checkpoints/                   ← Saved .pth weight files
│   └── samples/                       ← Generated image grids per epoch
│
├── 📓 Q2_Pix2Pix/
│   ├── pix2pix_improved.ipynb        ← Full training notebook
│   ├── checkpoints/                   ← Face & Anime model weights
│   └── outputs/                       ← Result grids + loss plots
│
├── 📓 Q3_CycleGAN/
│   ├── cyclegan_notebook.ipynb       ← Full training notebook
│   ├── checkpoints/                   ← G_AB & G_BA weights
│   └── outputs/                       ← Cycle visualisations
│
└── README.md
```

---

## 🧪 Question 1 — Mode Collapse in GANs: DCGAN vs WGAN-GP

<div align="center">

### The Problem: Mode Collapse

> When a GAN's generator discovers a small set of outputs that consistently fool the discriminator, it collapses to producing only those outputs. The result: **repetitive, homogeneous, low-diversity images**.

</div>

### What We Built

| Component | DCGAN | WGAN-GP |
|---|---|---|
| **Loss Function** | Binary Cross-Entropy (`BCEWithLogitsLoss`) | Wasserstein Distance + Gradient Penalty |
| **D/C Output** | Raw logit (no Sigmoid) | Raw critic score (no activation) |
| **Normalisation** | BatchNorm | InstanceNorm |
| **Critic Steps per G** | 1 | 5 |
| **Gradient Penalty λ** | — | 10 |
| **Mode Collapse** | ⚠️ Prone | ✅ Eliminated |

### Architecture

```
DCGAN Generator                          WGAN-GP Critic
────────────────                         ──────────────────
z (100-dim noise)                        x (64×64 RGB)
     │                                        │
ConvTranspose2d → BN → ReLU            Conv2d → InstanceNorm → LeakyReLU
     × 4 layers (4→64px)               × 4 layers (64→4px)
     │                                        │
ConvTranspose2d → Tanh                  Conv2d (no activation)
     │                                        │
64×64 RGB Image                         scalar score ∈ ℝ
```

### Datasets

| Dataset | Source | Images | Purpose |
|---|---|---|---|
| **Anime Faces** | [soumikrakshit/anime-faces](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) | ~21,000 | Primary training domain |
| **Pokemon Sprites** | [jackemartin/pokemon-sprites](https://www.kaggle.com/datasets/jackemartin/pokemon-sprites) | ~1,000 | Optional secondary domain |

### Key Technical Fixes (vs vanilla GAN)

- **AMP-safe loss** — `BCELoss` + `autocast` crashes; `BCEWithLogitsLoss` fuses Sigmoid numerically
- **GP in float32** — Gradient penalty `autograd.grad` must bypass AMP; only Generator uses mixed precision  
- **`unwrap()` before save** — `DataParallel` wrapping breaks `load_state_dict` on single-GPU inference
- **`persistent_workers=True`** — Prevents DataLoader worker death between training phases

### Results & Visualisations

<div align="center">

| Metric | DCGAN | WGAN-GP |
|---|---|---|
| Pixel Std-Dev (diversity ↑) | lower | **higher** |
| Cosine Similarity (diversity ↓) | higher | **lower** |
| Mode Collapse | ⚠️ Present | ✅ None |
| FID Score | higher | **lower** |

*Generated images shown at Epochs 5 / 10 / 15 / 17 for both models*

</div>

### 🚀 Live Demo

<div align="center">

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Try%20It%20Live-Anime%20Face%20Generator-FFD21E?style=for-the-badge&logoColor=black)](https://huggingface.co/spaces/Faiezeee/Anime_Face_Generator)

**Generate synthetic anime faces using DCGAN or WGAN-GP — compare outputs side-by-side in real time**

</div>

---

## 🎨 Question 2 — Doodle-to-Real: Pix2Pix Image Translation

<div align="center">

### The Problem: Weak / Washed-Out Pixels

> Standard Pix2Pix with `LAMBDA_L1 = 100` forces the generator to minimise pixel-level error. The mathematically safest solution is to predict the **average colour** — producing grey, blurry, washed-out outputs.

</div>

### What We Built

**Pix2Pix** learns a mapping between **paired** input/output images using a conditional GAN. Input: sketch or edge image. Output: realistic photo or colourised image.

#### Generator: U-Net with Skip Connections

```
Input (256×256 Sketch)
    ↓  e1: 256→128  (no BN)
    ↓  e2: 128→64
    ↓  e3:  64→32
    ↓  e4:  32→16        ← encoder captures structure
    ↓  e5:  16→8
    ↓  e6:   8→4
    ↓  e7:   4→2
    ↓  e8:   2→1  (bottleneck)
    ↑  d1: + e7     (Dropout 0.3)
    ↑  d2: + e6     (Dropout 0.3)   ← skip connections preserve detail
    ↑  d3: + e5     (Dropout 0.3)
    ↑  d4: + e4
    ↑  d5: + e3
    ↑  d6: + e2
    ↑  d7: + e1
    ↑  d8: ConvTranspose → Tanh
Output (256×256 Realistic/Colourised Image)
```

#### Discriminator: PatchGAN (with Spectral Norm)

Classifies 70×70 overlapping image patches as real or fake. **Conditions on the input sketch** — so it judges whether the `(sketch, output)` pair is coherent, not just whether the output looks realistic in isolation.

### Improved Loss Function (4 Components)

| Loss | Formula | Weight | Purpose |
|---|---|---|---|
| **Adversarial** | `BCEWithLogitsLoss` on patch map | 1 | Realism |
| **L1 Pixel** | `|fake - real|` mean | **10** *(was 100)* | Fidelity |
| **VGG Perceptual** | `L1(VGG(fake), VGG(real))` at `relu2_2` + `relu3_3` | 10 | Sharp textures |
| **Sobel Edge** | `L1(∇fake, ∇real)` | 5 | Sharp edges |

> **Why perceptual loss fixes the blur**: L1 in pixel space penalises every pixel equally, so G learns to predict the expectation (blurry average). VGG features capture texture and structure — comparing in feature space forces G to reproduce sharp details.

### Datasets

| Dataset | Source | Task |
|---|---|---|
| **CUHK Face Sketch (CUFS)** | [arbazkhan971/cuhk-face-sketch-database-cufs](https://www.kaggle.com/datasets/arbazkhan971/cuhk-face-sketch-database-cufs) | Sketch → Real Face Photo |
| **Anime Sketch Colorization** | [ktaebum/anime-sketch-colorization-pair](https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair) | Sketch → Colourised Anime |

> ⚠️ **Direction note for Anime dataset**: Each image is side-by-side — `LEFT = colour (target)`, `RIGHT = sketch (input)`. This is opposite of the naive assumption.

### Key Technical Improvements

| Change | Before | After | Why |
|---|---|---|---|
| L1 Lambda | 100 | **10** | Reduce grey-average bias |
| Generator width | ngf=64 | **ngf=96** | More feature capacity |
| Dropout | 0.5 | **0.3** | Less colour noise |
| D normalisation | BatchNorm | **+ Spectral Norm** | Stable training |
| LR schedule | Linear decay | **ReduceLROnPlateau** | Gentler, metric-aware |
| Augmentation | Flip only | **+ ColorJitter + RandomCrop** | Richer colour training |

### Results

<div align="center">

| Task | SSIM | PSNR |
|---|---|---|
| Face Sketch → Real Photo | measured at runtime | measured at runtime |
| Anime Sketch → Colourised | measured at runtime | measured at runtime |

</div>

### 🚀 Live Demo

<div align="center">

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Try%20It%20Live-Sketch%20To%20Image%20Generator-FFD21E?style=for-the-badge&logoColor=black)](https://huggingface.co/spaces/Faiezeee/Sketch_To_Image_Generator)

**Upload any sketch — choose Face or Anime model — get a realistic output instantly**

</div>

---

## 🔄 Question 3 — Unpaired Domain Adaptation: CycleGAN

<div align="center">

### The Problem: No Paired Data

> In the real world, matched sketch–photo pairs are expensive and rare. CycleGAN learns bidirectional mappings between two domains **without any paired examples** — using cycle consistency as the only supervision signal.

</div>

### What We Built

CycleGAN trains **four networks simultaneously**:

```
Domain A (Sketch)                         Domain B (Photo)
──────────────────                        ─────────────────

real_A ──→[ G_AB ]──→ fake_B ──→[ G_BA ]──→ rec_A  ≈ real_A  (cycle A)
            │                                                      ↑
            └──→[ D_B ] judges whether fake_B looks like a real photo

real_B ──→[ G_BA ]──→ fake_A ──→[ G_AB ]──→ rec_B  ≈ real_B  (cycle B)
            │                                                      ↑
            └──→[ D_A ] judges whether fake_A looks like a real sketch
```

#### Generator: ResNet-based (6 ResNet blocks)

```
Input (128×128)
  ↓  c7s1-64:   ReflectionPad + Conv7×7 + InstanceNorm + ReLU
  ↓  d128:      Conv3×3 stride-2 + InstanceNorm + ReLU
  ↓  d256:      Conv3×3 stride-2 + InstanceNorm + ReLU
  ↓  R256 × 6:  ResNet Blocks (ReflectionPad + Conv + IN + ReLU + Dropout)
  ↑  u128:      ConvTranspose stride-2 + InstanceNorm + ReLU
  ↑  u64:       ConvTranspose stride-2 + InstanceNorm + ReLU
  ↑  c7s1-3:    ReflectionPad + Conv7×7 + Tanh
Output (128×128)
```

> **Why InstanceNorm** (not BatchNorm): CycleGAN processes single images — BatchNorm statistics from a batch of sketches should not bleed into the photo generator's normalisation. InstanceNorm normalises each image independently.

> **Why ReflectionPad** (not zero-pad): Zero-padding creates visible border artifacts after multiple convolutions. ReflectionPad mirrors the image at borders, producing natural-looking edges.

### Loss Functions

| Loss | Weight | Formula | Purpose |
|---|---|---|---|
| **Adversarial (LSGAN)** | 1 | `MSE(D(fake), 1)` | Domain realism |
| **Cycle Consistency** | λ=10 | `L1(G_BA(G_AB(A)), A)` | Structure preservation |
| **Identity** | λ=5 | `L1(G_BA(A), A)` | Colour/style preservation |

> **Why LSGAN** (MSE) instead of BCE: BCE saturates when D becomes very confident — gradients vanish, G stops learning. MSE penalises samples that are far from the decision boundary, providing non-zero gradients throughout training.

> **Why Identity Loss**: Without it, G_AB might freely change the colour of a photo even when it doesn't need to translate (e.g., a sketch that already looks like a photo). Identity loss anchors the mapping.

### Datasets

| Dataset | Source | Domain | Images |
|---|---|---|---|
| **Sketchy** | [sharanyasundar/sketchy-dataset](https://www.kaggle.com/datasets/sharanyasundar/sketchy-dataset) | Sketch + Photo | 75k sketches, 12.5k photos, 125 categories |
| **TU-Berlin** | [sdiaeyu6n/tu-berlin (HuggingFace)](https://huggingface.co/datasets/sdiaeyu6n/tu-berlin) | Sketch | 20k sketches, 250 categories |
| **QuickDraw** | [quickdraw-doodle-recognition](https://www.kaggle.com/c/quickdraw-doodle-recognition/data) | Sketch (strokes→PNG) | 50M drawings, 345 categories |

### Replay Buffer

A 50-image history buffer stores previously generated fakes. During D training, D sees a random mix of current and past fakes instead of only current ones. This prevents the training oscillation where G and D chase each other in cycles.

### Quantitative Evaluation

CycleGAN is unpaired — there is no ground truth translation to compare against. Metrics are computed on **cycle reconstruction** (how well the original image is recovered after two translations):

| Metric | Cycle A: Sketch→Photo→Sketch | Cycle B: Photo→Sketch→Photo |
|---|---|---|
| **SSIM** | measured at runtime | measured at runtime |
| **PSNR** | measured at runtime | measured at runtime |

> Higher cycle SSIM/PSNR = better structural preservation across the round trip.

### 🚀 Live Demo

<div align="center">

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Try%20It%20Live-CycleGAN%20Sketch%20↔%20Photo-FFD21E?style=for-the-badge&logoColor=black)](https://huggingface.co/spaces/Faiezeee/CYCLEGAN_Image-Sketch-Image_Converter)

**Translate sketches → realistic photos OR photos → sketches — bidirectional, no paired data needed**

</div>

---

## ⚡ Training Infrastructure

All three experiments run on **Kaggle Notebooks with GPU T4 × 2** using the following stack:

```python
# Mixed Precision (AMP)
from torch.amp import GradScaler, autocast
scaler = GradScaler('cuda')

with autocast(device_type='cuda'):
    output = model(input)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Multi-GPU (DataParallel)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Always unwrap before saving
torch.save(model.module.state_dict(), 'weights.pt')   # NOT model.state_dict()
```

| Setting | Q1 DCGAN/WGAN | Q2 Pix2Pix | Q3 CycleGAN |
|---|---|---|---|
| **Image Size** | 64×64 | 256×256 | 128×128 |
| **Batch Size** | 64 | 16 | 4 |
| **Epochs** | 17 | 25 | 50 |
| **Optimizer** | Adam | Adam | Adam |
| **Learning Rate** | 2e-4 | 2e-4 | 2e-4 |
| **Adam Betas** | (0.5, 0.999) | (0.5, 0.999) | (0.5, 0.999) |
| **Mixed Precision** | ✅ | ✅ | ✅ |
| **DataParallel** | ✅ | ✅ | ✅ |

---

## 🏗 GAN Architecture Comparison

```
┌──────────────────┬────────────────┬──────────────────┬──────────────────┐
│                  │   DCGAN/WGAN   │    Pix2Pix       │   CycleGAN       │
├──────────────────┼────────────────┼──────────────────┼──────────────────┤
│ Supervision      │ Unpaired       │ Paired           │ Unpaired         │
│ Generator        │ DCGAN ConvNet  │ U-Net            │ ResNet (6 blocks)│
│ Discriminator    │ Standard D     │ PatchGAN (cond.) │ PatchGAN (uncond)│
│ # Generators     │ 1              │ 1                │ 2 (G_AB, G_BA)   │
│ # Discriminators │ 1              │ 1                │ 2 (D_A, D_B)     │
│ Adv. Loss        │ BCE / MSE      │ BCE              │ LSGAN (MSE)      │
│ Extra Loss       │ W-distance+GP  │ L1+Perceptual    │ Cycle+Identity   │
│ Normalisation    │ BatchNorm / IN │ BatchNorm        │ InstanceNorm     │
│ Replay Buffer    │ —              │ —                │ ✅ (50 images)   │
│ Conditioning     │ Noise only     │ Input sketch     │ Domain only      │
└──────────────────┴────────────────┴──────────────────┴──────────────────┘
```

---

## 🚀 HuggingFace Deployed Apps

<div align="center">

| # | Space | Description | Status |
|---|---|---|---|
| **Q1** | [🤗 Anime Face Generator](https://huggingface.co/spaces/Faiezeee/Anime_Face_Generator) | Generate anime faces with DCGAN or WGAN-GP | ![Running](https://img.shields.io/badge/Status-Running-22c55e?style=flat-square) |
| **Q2** | [🤗 Sketch To Image Generator](https://huggingface.co/spaces/Faiezeee/Sketch_To_Image_Generator) | Convert sketches to realistic face/anime photos | ![Running](https://img.shields.io/badge/Status-Running-22c55e?style=flat-square) |
| **Q3** | [🤗 CycleGAN Image↔Sketch Converter](https://huggingface.co/spaces/Faiezeee/CYCLEGAN_Image-Sketch-Image_Converter) | Bidirectional unpaired domain translation | ![Running](https://img.shields.io/badge/Status-Running-22c55e?style=flat-square) |

**[→ View all spaces on HuggingFace](https://huggingface.co/methebilalashiq)**

</div>

---

## 📚 References & Further Reading

| Paper | Year | Relevance |
|---|---|---|
| [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Goodfellow et al. | 2014 | Original GAN formulation |
| [Unsupervised Representation Learning with Deep Convolutional GANs](https://arxiv.org/abs/1511.06434) — Radford et al. | 2016 | DCGAN architecture |
| [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — Arjovsky et al. | 2017 | Wasserstein distance for GAN training |
| [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) — Gulrajani et al. | 2017 | Gradient Penalty (WGAN-GP) |
| [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Isola et al. | 2017 | Pix2Pix / PatchGAN |
| [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — Zhu et al. | 2017 | CycleGAN |
| [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155) — Johnson et al. | 2016 | VGG Perceptual Loss |

---

## 🛠 Quick Start

```bash
# Clone the repository
git clone https://github.com/faiez123tariq/GAN_Experiments.git
cd GAN_Experiments

# Install core dependencies
pip install torch torchvision kagglehub datasets scikit-image gradio Pillow tqdm

# Run any notebook on Kaggle (recommended for GPU)
# Or locally if you have an NVIDIA GPU:
jupyter notebook Q1_DCGAN_WGANGP/GAN_DCGAN_WGANGP.ipynb
jupyter notebook Q2_Pix2Pix/pix2pix_improved.ipynb
jupyter notebook Q3_CycleGAN/cyclegan_notebook.ipynb
```

![](https://i.imgur.com/waxVImv.png)

## 👤 Author

<div align="center">

<img src="https://avatars.githubusercontent.com/thecallmeBilalAshiq" width="100" style="border-radius: 50%;" alt="Muhammad Bilal Ashiq"/>

### **Muhammad Bilal Ashiq**

[![GitHub](https://img.shields.io/badge/GitHub-BilalAshiq-181717?style=for-the-badge&logo=github)](https://github.com/thecallmeBilalAshiq)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-BilalAshiq-0A66C2?style=for-the-badge&logo=linkedin)](https://linkedin.com/in/bilal-ashiq)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-BilalAshiq-FFD21E?style=for-the-badge)](https://huggingface.co/methebilalashiq)

*Open to Collaboration*

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" alt="footer"/>

**If this repository helped you, please consider giving it a ⭐**

</div>
