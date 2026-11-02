from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from main import MVTecDataset, ResNet18_MS3, test


root = Path(__file__).resolve().parent
image_path = root.parent / "data/carpet/test/color/000.png"
checkpoint_path = root / "snapshots/carpet/best.pth.tar"
output_path = root / "results/carpet_color_000_heatmap_formal.png"

transform = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
loader = DataLoader(MVTecDataset([str(image_path)], transform=transform), batch_size=1)

teacher = ResNet18_MS3(pretrained=True).cuda()
student = ResNet18_MS3(pretrained=False).cuda()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
student.load_state_dict(checkpoint["state_dict"])

score = cv2.resize(test(teacher, student, loader)[0], (256, 256))
display_score = (255 * (score - score.min()) / (np.ptp(score) + 1e-12)).astype(np.uint8)
heatmap = cv2.cvtColor(cv2.applyColorMap(display_score, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

image = Image.open(image_path).convert("RGB").resize((256, 256))
heatmap_image = Image.fromarray(heatmap)
overlay = Image.blend(image, heatmap_image, 0.45)
result = Image.new("RGB", (768, 256))
for index, panel in enumerate((image, heatmap_image, overlay)):
    result.paste(panel, (index * 256, 0))

output_path.parent.mkdir(parents=True, exist_ok=True)
result.save(output_path)
print(f"Saved: {output_path}")
print(f"Raw anomaly score range: {score.min():.6f} to {score.max():.6f}")
