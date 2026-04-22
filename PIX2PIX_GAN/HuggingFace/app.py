import json, torch, gradio as gr
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np

# ── Load architecture config ──────────────────────────────────
with open("architecture.json") as f:
    cfg = json.load(f)

IMG_SIZE = cfg["training"]["img_size"]
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model Definition (must match training exactly) ────────────
class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_bn)]
        if use_bn: layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                  nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        if dropout: layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        self.e1 = DownBlock(in_ch,  ngf,    use_bn=False)
        self.e2 = DownBlock(ngf,    ngf*2)
        self.e3 = DownBlock(ngf*2,  ngf*4)
        self.e4 = DownBlock(ngf*4,  ngf*8)
        self.e5 = DownBlock(ngf*8,  ngf*8)
        self.e6 = DownBlock(ngf*8,  ngf*8)
        self.e7 = DownBlock(ngf*8,  ngf*8)
        self.e8 = DownBlock(ngf*8,  ngf*8, use_bn=False)
        self.d1 = UpBlock(ngf*8,    ngf*8, dropout=True)
        self.d2 = UpBlock(ngf*8*2,  ngf*8, dropout=True)
        self.d3 = UpBlock(ngf*8*2,  ngf*8, dropout=True)
        self.d4 = UpBlock(ngf*8*2,  ngf*8)
        self.d5 = UpBlock(ngf*8*2,  ngf*4)
        self.d6 = UpBlock(ngf*4*2,  ngf*2)
        self.d7 = UpBlock(ngf*2*2,  ngf)
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, out_ch, 4, 2, 1), nn.Tanh())
    def forward(self, x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4); e6=self.e6(e5); e7=self.e7(e6); b=self.e8(e7)
        d1=self.d1(b)
        d2=self.d2(torch.cat([d1,e7],1)); d3=self.d3(torch.cat([d2,e6],1))
        d4=self.d4(torch.cat([d3,e5],1)); d5=self.d5(torch.cat([d4,e4],1))
        d6=self.d6(torch.cat([d5,e3],1)); d7=self.d7(torch.cat([d6,e2],1))
        return self.d8(torch.cat([d7,e1],1))

# ── Load weights ──────────────────────────────────────────────
def load_model(weight_file):
    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(weight_file, map_location=device))
    G.eval()
    return G

G_face  = load_model("generator_face.pth")
G_anime = load_model("generator_anime.pth")
print("✅ Both models loaded.")

# ── Inference ─────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

@torch.no_grad()
def generate(sketch_pil, model_choice):
    G = G_face if "Face" in model_choice else G_anime
    tensor = transform(sketch_pil.convert("RGB")).unsqueeze(0).to(device)
    out = G(tensor).squeeze(0).cpu().float().numpy()
    out = (out * 0.5 + 0.5).clip(0, 1).transpose(1, 2, 0)
    return Image.fromarray((out * 255).astype(np.uint8))

# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Pix2Pix Sketch → Real") as demo:
    gr.Markdown("# 🎨 Pix2Pix: Sketch → Realistic / Colorized Image")
    gr.Markdown("Upload a sketch and choose a model to generate a realistic or colorized output.")
    with gr.Row():
        with gr.Column():
            inp    = gr.Image(type="pil", label="Upload Sketch")
            choice = gr.Radio(
                choices=["Face Sketch -> Real Photo", "Anime Sketch -> Colorized"],
                value="Face Sketch -> Real Photo", label="Model")
            btn = gr.Button("Generate ✨", variant="primary")
        with gr.Column():
            out_img = gr.Image(type="pil", label="Generated Output")
    btn.click(fn=generate, inputs=[inp, choice], outputs=out_img)

demo.launch()