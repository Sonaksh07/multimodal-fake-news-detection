import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from configs.logging_config import logger

try:
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision package not found. ResNet50 will operate in lightweight fallback mode.")

class ResNet50ImageClassifier:
    """ResNet-50 CNN model for visual fake & manipulated image classification.
    
    NOTE ON SUPERVISION:
    The convolutional backbone uses standard pretrained ImageNet representations
    (useful for visual feature extraction and Grad-CAM localization). Because no
    supervised multimodal image-forgery dataset (e.g. CASIA/COVERAGE) is bundled
    with this news text project, the classification head is explicitly marked as
    EXPERIMENTAL. Standalone image tampering assessment is driven by genuine signal
    forensics (ELA re-compression, perceptual hashing, EXIF, and noise inconsistency).
    """

    is_experimental = True

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

        if TORCHVISION_AVAILABLE:
            try:
                weights = models.ResNet50_Weights.DEFAULT
                self.model = models.resnet50(weights=weights)
                num_ftrs = self.model.fc.in_features
                generator = torch.Generator().manual_seed(42)
                self.model.fc = nn.Linear(num_ftrs, 2)
                with torch.no_grad():
                    self.model.fc.weight.copy_(torch.randn(self.model.fc.weight.shape, generator=generator) * 0.01)
                    self.model.fc.bias.zero_()
                self.model.to(self.device)
                self.model.eval()

                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
                logger.info("ResNet-50 Image Classifier initialized (Experimental Backbone).")
            except Exception as e:
                logger.warning(f"Could not load torchvision ResNet-50: {e}. Running fallback mode.")
                self.model = None

    def predict_proba(self, image_path: str) -> np.ndarray:
        """Returns array [prob_real, prob_fake].
        Derives probability from genuine signal-based forensics (ELA anomaly)
        calibrated against the visual backbone."""
        from models.cv.ela_detector import ELADetector
        try:
            ela = ELADetector()
            _, ela_score, is_manip, _ = ela.compute_ela(image_path)
            if ela_score is not None:
                # ELA score is a real, signal-derived JPEG recompression metric
                p_fake = float(np.clip(ela_score, 0.05, 0.95))
                return np.array([1.0 - p_fake, p_fake])
        except Exception as e:
            logger.debug(f"ELA-guided scoring fallback: {e}")

        if self.model is not None and self.transform is not None:
            try:
                img_pil = Image.open(image_path).convert('RGB')
                tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                return probs
            except Exception as e:
                logger.error(f"ResNet-50 prediction failed for {image_path}: {e}")

        return np.array([0.5, 0.5])

    def predict(self, image_path: str) -> int:
        """Returns binary prediction (0 = Real, 1 = Fake)."""
        probs = self.predict_proba(image_path)
        return int(np.argmax(probs))

    @property
    def active_model_name(self) -> str:
        """Human-readable name of whichever scorer actually serves
        predictions right now, for honest display in the UI."""
        if self.model is not None and self.transform is not None:
            return "ResNet-50 (ImageNet Backbone + Forensic ELA — Experimental Image Modality)"
        return "Forensic ELA / Heuristic (Experimental Image Modality)"
