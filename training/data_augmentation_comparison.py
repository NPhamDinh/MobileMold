import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, cohen_kappa_score, matthews_corrcoef
from tqdm import tqdm
import os
from PIL import Image
import json
import copy
from torch import distributed as dist


def init_process_group():
    """
    Join the process group and return whether this is the rank 0 process,
    the CUDA device to use, and the total number of GPUs used for training.
    """
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    num_gpus = int(os.getenv("WORLD_SIZE", 1))
    dist.init_process_group("nccl")
    return rank == 0, torch.device(f"cuda:{local_rank}"), num_gpus


is_rank0, device, num_gpus = init_process_group()

# Configuration
DATA_ROOT = "cropped_resized/"
BATCH_SIZE = 64
LR = 1e-5
NUM_EPOCHS = 60  # Finetuning all layers needs more epochs
NUM_CLASSES = 2
INPUT_SIZE = 224

from torchvision.models import swin_b, mobilenet_v3_large


def get_models():
    return {
        "swin": swin_b(weights="IMAGENET1K_V1"),
        "mobilenet": mobilenet_v3_large(weights="IMAGENET1K_V2"),
    }


# --- Augmentation Strategies to Test ---
# The 'Baseline' uses the same transforms as the original model_comparison_unfreeze.py
augmentation_strategies = {
    "Baseline": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "TrivialAugment": transforms.Compose(
        [transforms.TrivialAugmentWide(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    ),
    "RandAugment": transforms.Compose(
        [transforms.RandAugment(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    ),
    "AutoAugment": transforms.Compose(
        [
            transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "AugMix": transforms.Compose(
        [transforms.AugMix(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
}

# The transform for batch-level augmentations (they are applied after ToTensor)
batch_level_base_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class MoldDataset(Dataset):
    def __init__(self, metadata_file, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(DATA_ROOT, row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["mold"])

        if self.transform:
            image = self.transform(image)

        return image, label


# Validation/Test transform is always the same
val_test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# Model Preparation Functions -- ensure all layers are unfrozen
def prepare_cnn(model, model_name):
    for param in model.parameters():
        param.requires_grad = True  # Finetune all layers

    if "mobilenet" in model_name:
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs, NUM_CLASSES))
    return model


def prepare_transformer(model, model_name):
    for param in model.parameters():
        param.requires_grad = True  # Finetune all layers

    if "swin" in model_name:
        num_ftrs = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs, NUM_CLASSES))
    return model


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs),
        "pr_auc": average_precision_score(all_labels, all_probs),
        "kappa": cohen_kappa_score(all_labels, all_preds),
        "mcc": matthews_corrcoef(all_labels, all_preds),
    }
    return metrics


def train_model(model, model_name, aug_name, train_loader, val_loader, device):
    writer = SummaryWriter(f"runs_augmentation/{model_name}/{aug_name}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_mcc = -1.0
    best_model_wts = None

    for epoch in range(NUM_EPOCHS):
        model.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        val_metrics = evaluate(model, val_loader, device)

        # Log metrics
        writer.add_scalar("MCC/val", val_metrics["mcc"], epoch)
        writer.add_scalar("F1/val", val_metrics["f1"], epoch)

        # Save best model based on validation MCC
        if val_metrics["mcc"] > best_mcc:
            best_mcc = val_metrics["mcc"]
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"  -> New best model saved with Val MCC: {best_mcc:.4f} at epoch {epoch+1}")

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Val MCC: {val_metrics['mcc']:.4f} | F1: {val_metrics['f1']:.4f}")

    if best_model_wts:
        torch.save(best_model_wts, f"aug_{model_name}_{aug_name}_best.pth")

    writer.close()
    return best_model_wts


def main():
    results = {}

    # Create static validation and test datasets
    val_dataset = MoldDataset("val_metadata.csv", val_test_transform)
    test_dataset = MoldDataset("test_metadata.csv", val_test_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Outer loop: Iterate through models
    for model_name, model_instance in get_models().items():
        results[model_name] = {}

        # Inner loop: Iterate through augmentation strategies
        for aug_name, transform_obj in augmentation_strategies.items():
            print(f"\n{'='*20}\nTraining {model_name} with {aug_name}...\n{'='*20}")

            # 1. Select the correct transform for training
            if transform_obj == "batch_level":
                current_train_transform = batch_level_base_transform
            else:
                current_train_transform = transform_obj

            # 2. Create training dataset and dataloader for the current experiment
            train_dataset = MoldDataset("train_metadata.csv", current_train_transform)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

            # 3. Prepare a fresh instance of the model
            model = get_models()[model_name]
            if "swin" in model_name:
                model = prepare_transformer(model, model_name)
            else:
                model = prepare_cnn(model, model_name)

            # 4. Train the model
            best_weights = train_model(model, model_name, aug_name, train_loader, val_loader, device)

            # 5. Evaluate the best model
            if best_weights:
                model.load_state_dict(best_weights)

                val_metrics = evaluate(model, val_loader, device)
                test_metrics = evaluate(model, test_loader, device)

                results[model_name][aug_name] = {"val": val_metrics, "test": test_metrics}

                print(f"\n--- Results for {model_name} with {aug_name} ---")
                print("Validation:", val_metrics)
                print("Test:", test_metrics)
            else:
                print(f"!!! Training failed for {model_name} with {aug_name}, no best model found. !!!")

            # 6. Cleanup to free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save all results to a single JSON file
    with open("augmentation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n\nExperiment finished. All results saved to 'augmentation_results.json'")


if __name__ == "__main__":
    main()
