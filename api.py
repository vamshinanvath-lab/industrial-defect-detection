
from pathlib import Path
from io import BytesIO

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile, HTTPException

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


app = FastAPI(
    title="Industrial Defect Detection API",
    description="ResNet18-based industrial casting defect classifier",
    version="1.0.0"
)


model = None


@app.on_event("startup")
def load_model():
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    model = build_model(pretrained=False)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()


@app.get("/")
def home():
    return {
        "message": "Industrial Defect Detection API is running",
        "model": "ResNet18",
        "classes": ["Defective", "Good"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG and PNG images are supported."
        )

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]

        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item() * 100

        return {
            "filename": file.filename,
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 2),
            "probabilities": {
                CLASS_NAMES[i]: round(
                    probabilities[i].item() * 100, 2
                )
                for i in range(2)
            }
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Image processing failed: {str(error)}"
        )
