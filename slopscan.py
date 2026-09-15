"""Minimal single-image inference for the Slopscan AI/NotAI classifier.

    from slopscan import Slopscan

    clf = Slopscan("best.pt")
    prob = clf.classify("image.png")  # P(AI generated), 0..1
"""
import timm
import torch
from PIL import Image
from torchvision import transforms

BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
IMG_SIZE = 1024
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Slopscan:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict") or ckpt.get("ema_state") or ckpt
        self.model = timm.create_model(BACKBONE, pretrained=False, num_classes=2)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE + 32),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def classify(self, image):
        img = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image)
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(x), dim=1)[0, 1].item()
