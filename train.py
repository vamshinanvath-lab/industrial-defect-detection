
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import create_dataloaders
from src.models.resnet_model import build_model


# ---------------- CONFIGURATION ----------------

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.pth"

EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- TRAINING FUNCTION ----------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    # Keep the pretrained feature extractor frozen.
    # Only the final classification layer is trainable.
    model.eval()
    model.fc.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ---------------- VALIDATION FUNCTION ----------------

def validate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ---------------- MAIN TRAINING PIPELINE ----------------

def main():
    print("INDUSTRIAL DEFECT DETECTION - TRAINING")
    print("=" * 50)

    print("Device:", DEVICE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    # Load datasets
    train_loader, val_loader, test_loader, class_to_idx = (
        create_dataloaders()
    )

    # Build model
    model = build_model(pretrained=True)
    model.to(DEVICE)

    # Freeze the pretrained feature extractor
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Train only the final classification layer
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    # Calculate class weights from the actual training subset
    train_subset = train_loader.dataset
    underlying_dataset = train_subset.dataset

    train_labels = [
        underlying_dataset.targets[index]
        for index in train_subset.indices
    ]

    class_counts = torch.bincount(
        torch.tensor(train_labels),
        minlength=len(class_to_idx)
    ).float()

    class_weights = len(train_labels) / (
        len(class_counts) * class_counts
    )

    class_weights = class_weights.to(DEVICE)

    print("\nClass counts:", class_counts.tolist())
    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")

    print("\nSTARTING TRAINING")
    print("=" * 50)

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Save the best checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "class_to_idx": class_to_idx,
                "class_weights": class_weights.tolist(),
            }, MODEL_PATH)

            print("Best model checkpoint saved!")

    print("\nTRAINING COMPLETED")
    print("Best validation loss:", round(best_val_loss, 4))
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
