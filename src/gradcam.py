"""Minimal Grad-CAM for the timm EfficientNet-B0 (last conv block). Optional; used by the notebook."""
import numpy as np, torch
from PIL import Image


def gradcam(model, x, device):
    """x: [1,3,S,S] normalised tensor -> (heatmap SxS in [0,1], logit)."""
    model.eval(); feats, grads = [], []
    layer = model.blocks[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: feats.append(o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        x = x.to(device).requires_grad_(True)
        z = model(x).squeeze()
        model.zero_grad(); z.backward()
        w = grads[0].mean((2, 3), keepdim=True)
        cam = torch.relu((w * feats[0]).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min(); cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), float(z.detach())
    finally:
        h1.remove(); h2.remove()


def overlay(pil_img, cam):
    import matplotlib.cm as cm
    base = np.asarray(pil_img.convert("RGB").resize(cam.shape[::-1]), dtype=np.float32) / 255
    heat = cm.jet(cam)[..., :3]
    return Image.fromarray((np.clip(0.55 * base + 0.45 * heat, 0, 1) * 255).astype(np.uint8))
