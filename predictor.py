
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.resnet_model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"

DEVICE = torch.device("cpu")

CLASS_NAMES = {
    0: "Defective",
    1: "Good"
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class DefectPredictor:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {MODEL_PATH}"
            )

        self.model = build_model(pretrained=False)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(DEVICE)
        self.model.eval()

        print("Model loaded successfully!")

    def predict(self, image_path):

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities, dim=1
        )

        class_index = predicted_class.item()

        result = {
            "image": image_path.name,
            "prediction": CLASS_NAMES[class_index],
            "confidence": confidence.item() * 100,
            "probabilities": {
                CLASS_NAMES[i]: probabilities[0, i].item() * 100
                for i in range(2)
            }
        }

        return result


def main():
    parser = argparse.ArgumentParser(
        description="Industrial casting defect prediction"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the casting image"
    )

    args = parser.parse_args()

    predictor = DefectPredictor()
    result = predictor.predict(args.image)

    print("\nPREDICTION RESULT")
    print("-" * 40)
    print("Image:", result["image"])
    print("Prediction:", result["prediction"])
    print(f"Confidence: {result['confidence']:.2f}%")

    print("\nClass probabilities:")

    for class_name, probability in result["probabilities"].items():
        print(f"{class_name}: {probability:.2f}%")


if __name__ == "__main__":
    main()
