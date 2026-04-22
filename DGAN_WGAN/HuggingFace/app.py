import json, torch, gradio as gr, numpy as np
from PIL import Image
from torchvision.utils import make_grid
import torch.nn as nn

with open("model_config.json") as f:
    cfg = json.load(f)
NZ, NGF, NC = cfg["NZ"], cfg["NGF"], cfg["NC"]
DATASET = cfg["DATASET"]
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz,    ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf,   nc,    4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z): return self.main(z)

g_dc = Generator(NZ, NGF, NC).to(device)
g_dc.load_state_dict(torch.load("dcgan_generator_final.pth", map_location=device))
g_dc.eval()

g_wg = Generator(NZ, NGF, NC).to(device)
g_wg.load_state_dict(torch.load("wgan_generator_final.pth", map_location=device))
g_wg.eval()

def gen(model_name, n, seed):
    torch.manual_seed(int(seed))
    noise = torch.randn(int(n), NZ, 1, 1, device=device)
    model = g_dc if model_name == "DCGAN" else g_wg
    with torch.no_grad():
        imgs = (model(noise) * 0.5 + 0.5).clamp(0, 1).cpu()
    grid = make_grid(imgs, nrow=min(int(n), 5), padding=2)
    return Image.fromarray((grid.permute(1,2,0).numpy()*255).astype("uint8"))

def compare(n, seed):
    return gen("DCGAN", n, seed), gen("WGAN-GP", n, seed)

with gr.Blocks(title=f"GAN - {DATASET.upper()}", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# DCGAN vs WGAN-GP | {DATASET.upper()} | 64x64")
    with gr.Tab("Generate"):
        with gr.Row():
            md = gr.Dropdown(["DCGAN","WGAN-GP"], value="DCGAN", label="Model")
            ns = gr.Slider(1, 25, value=10, step=1, label="Images")
            si = gr.Number(value=42, label="Seed")
        out = gr.Image(type="pil", label="Output")
        gr.Button("Generate", variant="primary").click(gen, [md,ns,si], out)
    with gr.Tab("Compare Side-by-Side"):
        with gr.Row():
            cn = gr.Slider(1,10,value=5,step=1,label="Images per model")
            cs = gr.Number(value=42, label="Seed")
        od = gr.Image(label="DCGAN", type="pil")
        ow = gr.Image(label="WGAN-GP", type="pil")
        gr.Button("Compare", variant="primary").click(compare, [cn,cs], [od,ow])
    with gr.Tab("Model Info"):
        gr.Markdown(f"""| | DCGAN | WGAN-GP |
|---|---|---|
| G Loss | {cfg["dcgan_final_g_loss"]} | {cfg["wgan_final_g_loss"]} |
| D/C Loss | {cfg["dcgan_final_d_loss"]} | {cfg["wgan_final_c_loss"]} |
| Epochs | {cfg["DCGAN_EPOCHS"]} | {cfg["WGAN_EPOCHS"]} |""")
demo.launch()
