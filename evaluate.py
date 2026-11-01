
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from src.data.dataset import create_dataloaders
from src.models.resnet_model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

DEVICE = torch.device("cpu")


def evaluate_model():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("INDUSTRIAL DEFECT DETECTION - TEST EVALUATION")
    print("=" * 55)

    # Load the test DataLoader
    _, _, test_loader, class_to_idx = create_dataloaders()

    # Build the model without downloading pretrained weights
    model = build_model(pretrained=False)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if checkpoint["class_to_idx"] != class_to_idx:
        raise ValueError("Checkpoint and dataset class mappings differ.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=[class_to_idx["def_front"], class_to_idx["ok_front"]],
        target_names=["Defective", "Good"],
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[
            class_to_idx["def_front"],
            class_to_idx["ok_front"]
        ]
    )

    # Print results
    print("\nFINAL TEST METRICS")
    print("-" * 40)
    print(f"Test images: {len(all_labels)}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-score:  {f1 * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print(report)

    print("CONFUSION MATRIX")
    print(matrix)

    # Save confusion matrix image
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Defective", "Good"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Industrial Defect Detection - Test Confusion Matrix")
    plt.tight_layout()

    matrix_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(matrix_path, dpi=300)
    plt.close()

    # Save metrics and classification report
    metrics_path = RESULTS_DIR / "test_metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("INDUSTRIAL DEFECT DETECTION - TEST RESULTS\n")
        file.write("=" * 50 + "\n")
        file.write(f"Test images: {len(all_labels)}\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"Precision: {precision * 100:.2f}%\n")
        file.write(f"Recall: {recall * 100:.2f}%\n")
        file.write(f"F1-score: {f1 * 100:.2f}%\n\n")
        file.write("CLASSIFICATION REPORT\n")
        file.write(report)
        file.write("\nCONFUSION MATRIX\n")
        file.write(str(matrix))

    print("\nEvaluation completed.")
    print("Metrics saved:", metrics_path)
    print("Confusion matrix saved:", matrix_path)


if __name__ == "__main__":
    evaluate_model()
