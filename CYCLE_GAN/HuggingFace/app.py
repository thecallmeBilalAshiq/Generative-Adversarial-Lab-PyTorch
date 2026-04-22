
# ═══════════════════════════════════════════════════════════
#  HuggingFace Gradio Space — CycleGAN Sketch <-> Photo
#  Files needed:
#    app.py  requirements.txt
#    G_AB_final.pt   (Sketch -> Photo)
#    G_BA_final.pt   (Photo  -> Sketch)
#    cyclegan_config.json
# ═══════════════════════════════════════════════════════════
import json, torch, gradio as gr, numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T

with open("cyclegan_config.json") as f:
    cfg = json.load(f)
IMG_SIZE    = cfg["IMG_SIZE"]
N_RESBLOCKS = cfg["N_RESBLOCKS"]
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch,ch,3,bias=False),
            nn.InstanceNorm2d(ch), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.ReflectionPad2d(1), nn.Conv2d(ch,ch,3,bias=False),
            nn.InstanceNorm2d(ch))
    def forward(self, x): return x + self.block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3), nn.Conv2d(in_ch,ngf,7,bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(inplace=True),
            nn.Conv2d(ngf,ngf*2,3,stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(inplace=True),
            nn.Conv2d(ngf*2,ngf*4,3,stride=2,padding=1,bias=False),
            nn.InstanceNorm2d(ngf*4), nn.ReLU(inplace=True),
        ] + [ResBlock(ngf*4) for _ in range(n_blocks)] + [
            nn.ConvTranspose2d(ngf*4,ngf*2,3,stride=2,padding=1,output_padding=1,bias=False),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2,ngf,3,stride=2,padding=1,output_padding=1,bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3), nn.Conv2d(ngf,out_ch,7), nn.Tanh()
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

def load_gen(path):
    g = ResNetGenerator(n_blocks=N_RESBLOCKS).to(device)
    g.load_state_dict(torch.load(path, map_location=device))
    return g.eval()

G_AB = load_gen("G_AB_final.pt")   # Sketch -> Photo
G_BA = load_gen("G_BA_final.pt")   # Photo  -> Sketch

tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), T.InterpolationMode.BICUBIC),
    T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)
])

def proc(pil): return tf(pil.convert("RGB")).unsqueeze(0).to(device)
def deproc(t):
    a = (t.squeeze(0).cpu().float().numpy()*0.5+0.5).clip(0,1)
    return Image.fromarray((a.transpose(1,2,0)*255).astype("uint8"))

@torch.no_grad()
def translate(img_pil, direction):
    x = proc(img_pil)
    G = G_AB if direction == "Sketch -> Photo" else G_BA
    return deproc(G(x))

@torch.no_grad()
def full_cycle(img_pil, direction):
    x = proc(img_pil)
    if direction == "Sketch -> Photo":
        translated = G_AB(x)
        reconstructed = G_BA(translated)
    else:
        translated = G_BA(x)
        reconstructed = G_AB(translated)
    return deproc(translated), deproc(reconstructed)

with gr.Blocks(title="CycleGAN Sketch <-> Photo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # CycleGAN: Sketch ↔ Photo Domain Translation
    Unpaired image translation — no matched pairs required during training.

    | Direction | Cycle SSIM | Cycle PSNR |
    |-----------|-----------|----------|
    | Sketch→Photo→Sketch | {cfg["ssim_cycle_A"]} | {cfg["psnr_cycle_A"]}dB |
    | Photo→Sketch→Photo | {cfg["ssim_cycle_B"]} | {cfg["psnr_cycle_B"]}dB |
    """)

    with gr.Tab("Translate"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Input Image")
                dir_radio = gr.Radio(
                    ["Sketch -> Photo", "Photo -> Sketch"],
                    value="Sketch -> Photo", label="Translation Direction")
                btn = gr.Button("Translate", variant="primary")
            out = gr.Image(type="pil", label="Translated Output")
        btn.click(translate, [inp, dir_radio], out)

    with gr.Tab("Full Cycle"):
        gr.Markdown("Shows: Input -> Translated -> Reconstructed (cycle)")
        with gr.Row():
            inp2 = gr.Image(type="pil", label="Input")
            dir2 = gr.Radio(
                ["Sketch -> Photo", "Photo -> Sketch"],
                value="Sketch -> Photo", label="Direction")
        btn2 = gr.Button("Run Full Cycle", variant="primary")
        with gr.Row():
            t_out = gr.Image(type="pil", label="Translated")
            r_out = gr.Image(type="pil", label="Reconstructed")
        btn2.click(full_cycle, [inp2, dir2], [t_out, r_out])

    with gr.Tab("Architecture"):
        gr.Markdown(f"""
        ## CycleGAN Architecture
        | Component | Details |
        |-----------|--------|
        | G_AB | ResNet Generator ({N_RESBLOCKS} blocks), Sketch→Photo |
        | G_BA | ResNet Generator ({N_RESBLOCKS} blocks), Photo→Sketch |
        | D_A | PatchGAN, classifies sketch domain |
        | D_B | PatchGAN, classifies photo domain |
        | Image Size | {IMG_SIZE}×{IMG_SIZE} |

        ## Loss Functions
        | Loss | Weight | Purpose |
        |------|--------|---------|
        | Adversarial (LSGAN) | 1 | Domain realism |
        | Cycle Consistency | {cfg["LAMBDA_CYCLE"]} | Structural preservation |
        | Identity | {cfg["LAMBDA_ID"]} | Style preservation |
        """)

demo.launch()
