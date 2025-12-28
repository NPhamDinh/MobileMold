import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score, cohen_kappa_score,
    matthews_corrcoef
)
from tqdm import tqdm
import os
from PIL import Image
from torch import distributed as dist # Wichtig für DDP!
from torchvision.models import swin_b

# --- DDP Setup Funktion ---
def init_process_group():
    # Diese Umgebungsvariablen werden von torchrun/torch.distributed.launch gesetzt
    rank = int(os.getenv('RANK', '0'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1')) # Gesamtzahl der Prozesse
    
    # Initiiere die Prozessgruppe
    # 'nccl' ist die empfohlene Backend für GPU-Training
    # 'env://' Rendezvous-Backend bedeutet, dass die Env-Variablen verwendet werden
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Setze das Device basierend auf local_rank
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device) # Wichtig, um die Standard-CUDA-Gerät für diesen Prozess zu setzen

    # rank == 0 ist der Hauptprozess für Logging, Speichern etc.
    is_rank0 = (rank == 0)
    
    return is_rank0, device, world_size # num_gpus ist jetzt world_size

# --- Configuration ---
DATA_ROOT = "cropped_resized/" # Sicherstellen, dass dies auf dem Cluster zugänglich ist
BATCH_SIZE = 64
LR = 1e-5
NUM_EPOCHS = 100
NUM_MOLD_CLASSES = 2
INPUT_SIZE = 224

# --- Bestimme Lebensmittelklassen und Mapping ---
# Diese Logik sollte idealerweise vor init_process_group() liegen,
# oder die resultierenden Variablen sollten über alle Prozesse synchronisiert werden.
# Für diesen Fall gehen wir davon aus, dass die Metadaten auf allen Nodes gleich sind
# und die Klassen konsistent bestimmt werden können.
all_metadata_paths = ["train_metadata.csv", "val_metadata.csv", "test_metadata.csv"]
all_food_types = set()

for path in all_metadata_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'food' in df.columns:
            all_food_types.update(df['food'].dropna().unique())
        else:
            print(f"Warning: 'food' column not found in {path}. Skipping food type detection for this file.")
    #else: # Entfernt, da es im Cluster nicht immer alle Pfade lokal geben muss
    #    print(f"Warning: Metadata file not found: {path}. Skipping.")

sorted_food_types = sorted(list(all_food_types))
food_to_idx = {food: idx for idx, food in enumerate(sorted_food_types)}
idx_to_food = {idx: food for food, idx in food_to_idx.items()}
NUM_FOOD_CLASSES = len(sorted_food_types)


# Initialize DDP process group at the very beginning
is_rank0, device, world_size = init_process_group() # 'num_gpus' ist jetzt 'world_size'


if is_rank0:
    print(f"Detected {NUM_FOOD_CLASSES} unique food types: {sorted_food_types}")
    print(f"Food to index mapping: {food_to_idx}")
    print(f"Using device: {device}")
    print(f"Total distributed processes: {world_size}")


# Dataset Class for Multi-Task Learning
class MoldFoodDataset(Dataset):
    def __init__(self, metadata_file, food_to_idx_map, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.food_to_idx_map = food_to_idx_map

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(DATA_ROOT, row['filename'])
        image = Image.open(img_path).convert('RGB')
        mold_label = int(row['mold'])
        
        food_label_str = row['food']
        if pd.isna(food_label_str):
            raise ValueError(f"Missing food label for image {row['filename']}. Please check your metadata.")
        
        food_label = self.food_to_idx_map[food_label_str]
        
        if self.transform:
            image = self.transform(image)
        return image, mold_label, food_label

# Transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = MoldFoodDataset("train_metadata.csv", food_to_idx, train_transform)
val_dataset = MoldFoodDataset("val_metadata.csv", food_to_idx, val_test_transform)
test_dataset = MoldFoodDataset("test_metadata.csv", food_to_idx, val_test_transform)

# --- Wichtig: DistributedSampler verwenden! ---
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=dist.get_rank())
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=4)

# Multi-Task Swin Model Definition
class MultiTaskSwin(nn.Module):
    def __init__(self, original_swin_model, num_mold_classes, num_food_classes):
        super().__init__()
        self.base_model = original_swin_model
        
        shared_bottom_output_features = self.base_model.head.in_features
        
        # Ersetze den ursprünglichen Head des Swin-Modells durch eine Identity-Layer
        self.base_model.head = nn.Identity()

        self.mold_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(shared_bottom_output_features, num_mold_classes)
        )
        self.food_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(shared_bottom_output_features, num_food_classes)
        )
        for m in [self.mold_head, self.food_head]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.base_model(x)
        mold_output = self.mold_head(features)
        food_output = self.food_head(features)
        return mold_output, food_output

# Evaluation function (updated for multi-task)
def evaluate(model, dataloader, device):
    model.eval()
    all_mold_labels = []
    all_mold_preds = []
    all_mold_probs = []
    all_food_labels = []
    all_food_preds = []
    all_food_probs = []

    with torch.no_grad():
        for inputs, mold_labels, food_labels in dataloader:
            inputs = inputs.to(device)
            mold_labels = mold_labels.to(device)
            food_labels = food_labels.to(device)
            mold_outputs, food_outputs = model(inputs)

            mold_probs = torch.softmax(mold_outputs, dim=1)
            _, mold_preds = torch.max(mold_outputs, 1)
            all_mold_labels.extend(mold_labels.cpu().numpy())
            all_mold_preds.extend(mold_preds.cpu().numpy())
            all_mold_probs.extend(mold_probs[:, 1].cpu().numpy())

            food_probs = torch.softmax(food_outputs, dim=1)
            _, food_preds = torch.max(food_outputs, 1)
            all_food_labels.extend(food_labels.cpu().numpy())
            all_food_preds.extend(food_preds.cpu().numpy())
            all_food_probs.extend(food_probs.cpu().numpy())

    all_mold_labels = np.array(all_mold_labels)
    all_mold_preds = np.array(all_mold_preds)
    all_mold_probs = np.array(all_mold_probs)
    all_food_labels = np.array(all_food_labels)
    all_food_preds = np.array(all_food_preds)
    all_food_probs = np.array(all_food_probs)

    metrics = {}
    metrics['mold_accuracy'] = accuracy_score(all_mold_labels, all_mold_preds)
    metrics['mold_f1'] = f1_score(all_mold_labels, all_mold_preds)
    metrics['mold_auc'] = roc_auc_score(all_mold_labels, all_mold_probs)
    metrics['mold_pr_auc'] = average_precision_score(all_mold_labels, all_mold_probs)
    metrics['mold_kappa'] = cohen_kappa_score(all_mold_labels, all_mold_preds)
    metrics['mold_mcc'] = matthews_corrcoef(all_mold_labels, all_mold_preds)

    metrics['food_accuracy'] = accuracy_score(all_food_labels, all_food_preds)
    metrics['food_f1_macro'] = f1_score(all_food_labels, all_food_preds, average='macro')
    metrics['food_f1_weighted'] = f1_score(all_food_labels, all_food_preds, average='weighted')
    
    if NUM_FOOD_CLASSES > 0:
        food_labels_one_hot = np.eye(NUM_FOOD_CLASSES)[all_food_labels]
        metrics['food_auc_ovo'] = roc_auc_score(food_labels_one_hot, all_food_probs, multi_class='ovo', average='macro')
        metrics['food_auc_ovr'] = roc_auc_score(food_labels_one_hot, all_food_probs, multi_class='ovr', average='macro')
    else:
        metrics['food_auc_ovo'] = np.nan
        metrics['food_auc_ovr'] = np.nan

    metrics['food_kappa'] = cohen_kappa_score(all_food_labels, all_food_preds)
    metrics['food_mcc'] = matthews_corrcoef(all_food_labels, all_food_preds)
    return metrics

# Training function for Multi-Task Model
def train_model(model, model_name, train_loader, val_loader, device):
    # SummaryWriter nur im Hauptprozess initialisieren
    writer = SummaryWriter(f'runs_multitask/{model_name}') if is_rank0 else None
    
    model = model.to(device)
    # --- DDP Wrapper anwenden ---
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    
    criterion_mold = nn.CrossEntropyLoss()
    criterion_food = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_mold_mcc = -1.0
    best_model_wts = None

    for epoch in range(NUM_EPOCHS):
        # --- Sampler-Epoch für DDP setzen ---
        train_loader.sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        mold_loss_total = 0.0
        food_loss_total = 0.0
        
        # tqdm nur im Hauptprozess anzeigen
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', disable=not is_rank0)
        for inputs, mold_labels, food_labels in pbar:
            inputs = inputs.to(device)
            mold_labels = mold_labels.to(device)
            food_labels = food_labels.to(device)
            optimizer.zero_grad()
            mold_outputs, food_outputs = model(inputs)
            loss_mold = criterion_mold(mold_outputs, mold_labels)
            loss_food = criterion_food(food_outputs, food_labels)
            loss = loss_mold + loss_food
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            mold_loss_total += loss_mold.item()
            food_loss_total += loss_food.item()
            
            if is_rank0:
                pbar.set_postfix({
                    'Loss': f'{total_loss / (pbar.n + 1):.4f}',
                    'Mold Loss': f'{mold_loss_total / (pbar.n + 1):.4f}',
                    'Food Loss': f'{food_loss_total / (pbar.n + 1):.4f}'
                })
        
        # --- Losses über alle Prozesse mitteln (synchronisieren) ---
        avg_total_loss = torch.tensor(total_loss / len(train_loader), device=device)
        avg_mold_loss = torch.tensor(mold_loss_total / len(train_loader), device=device)
        avg_food_loss = torch.tensor(food_loss_total / len(train_loader), device=device)
        
        dist.all_reduce(avg_total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_mold_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(avg_food_loss, op=dist.ReduceOp.SUM)
        
        avg_total_loss /= world_size # Durch Gesamtzahl der Prozesse teilen
        avg_mold_loss /= world_size
        avg_food_loss /= world_size

        # --- Validierung und Logging nur im Hauptprozess ---
        if is_rank0:
            val_metrics = evaluate(model.module, val_loader, device) # model.module für das Basismodell
            writer.add_scalar('Loss/train_total', avg_total_loss.item(), epoch)
            writer.add_scalar('Loss/train_mold', avg_mold_loss.item(), epoch)
            writer.add_scalar('Loss/train_food', avg_food_loss.item(), epoch)
            writer.add_scalar('Mold_Metrics/val_accuracy', val_metrics['mold_accuracy'], epoch)
            writer.add_scalar('Mold_Metrics/val_f1', val_metrics['mold_f1'], epoch)
            writer.add_scalar('Mold_Metrics/val_auc', val_metrics['mold_auc'], epoch)
            writer.add_scalar('Mold_Metrics/val_pr_auc', val_metrics['mold_pr_auc'], epoch)
            writer.add_scalar('Mold_Metrics/val_kappa', val_metrics['mold_kappa'], epoch)
            writer.add_scalar('Mold_Metrics/val_mcc', val_metrics['mold_mcc'], epoch)
            writer.add_scalar('Food_Metrics/val_accuracy', val_metrics['food_accuracy'], epoch)
            writer.add_scalar('Food_Metrics/val_f1_macro', val_metrics['food_f1_macro'], epoch)
            writer.add_scalar('Food_Metrics/val_f1_weighted', val_metrics['food_f1_weighted'], epoch)
            writer.add_scalar('Food_Metrics/val_auc_ovo', val_metrics['food_auc_ovo'], epoch)
            writer.add_scalar('Food_Metrics/val_auc_ovr', val_metrics['food_auc_ovr'], epoch)
            writer.add_scalar('Food_Metrics/val_kappa', val_metrics['food_kappa'], epoch)
            writer.add_scalar('Food_Metrics/val_mcc', val_metrics['food_mcc'], epoch)
            
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} (Rank {dist.get_rank()})")
            print(f"Train Loss: {avg_total_loss.item():.4f} (Mold: {avg_mold_loss.item():.4f}, Food: {avg_food_loss.item():.4f})")
            print(f"Val Mold MCC: {val_metrics['mold_mcc']:.4f} | F1: {val_metrics['mold_f1']:.4f}")
            print(f"Val Food Acc: {val_metrics['food_accuracy']:.4f} | F1 (Macro): {val_metrics['food_f1_macro']:.4f}")
            
            if val_metrics['mold_mcc'] > best_mold_mcc:
                best_mold_mcc = val_metrics['mold_mcc']
                best_model_wts = model.module.state_dict() # model.module für das Basismodell
                torch.save(best_model_wts, f'multitask_best_{model_name}.pth')
                print(f"Saved best model with Mold MCC: {best_mold_mcc:.4f}")
        
        # --- Barriere, damit alle Prozesse synchron sind vor dem nächsten Epoch ---
        dist.barrier()

    if is_rank0:
        if best_model_wts is None: # Falls nie ein besseres Modell gefunden wurde
            best_model_wts = model.module.state_dict()
            torch.save(best_model_wts, f'multitask_best_{model_name}.pth')
        writer.close()

    # --- Besten Modellzustand vom Rank 0 an alle anderen Prozesse senden ---
    # Dies stellt sicher, dass alle Prozesse denselben finalen Modellzustand haben
    # für die finale Evaluierung oder für das Speichern.
    if is_rank0:
        state_dict_list = [best_model_wts]
    else:
        state_dict_list = [None]
    dist.broadcast_object_list(state_dict_list, src=0)
    final_best_wts = state_dict_list[0]
    
    return final_best_wts

# Main training and evaluation loop (focused on Swin)
results = {}
model_name = "swin"

if is_rank0:
    print(f"\nStarting Multi-Task Training for {model_name}...")

original_swin_model = swin_b(weights="IMAGENET1K_V1")
model = MultiTaskSwin(original_swin_model, NUM_MOLD_CLASSES, NUM_FOOD_CLASSES)

best_weights = train_model(model, model_name, train_loader, val_loader, device)

# Erstelle eine neue Instanz des Modells für die finale Evaluierung
final_model_for_eval = MultiTaskSwin(swin_b(weights="IMAGENET1K_V1"), NUM_MOLD_CLASSES, NUM_FOOD_CLASSES)
final_model_for_eval.load_state_dict(best_weights)
final_model_for_eval = final_model_for_eval.to(device)


if is_rank0:
    print(f"\nEvaluating final Multi-Task {model_name} model...")
    # Wichtig: Val und Test Loader müssen auch DDP Sampler nutzen
    # Für die Evaluation kann es effizienter sein, wenn nur Rank 0 evaluiert
    # oder die Ergebnisse über alle Ränge gesammelt werden.
    # Hier wird angenommen, dass evaluate() für den Sub-Sampler funktioniert.
    val_metrics = evaluate(final_model_for_eval, val_loader, device)
    test_metrics = evaluate(final_model_for_eval, test_loader, device)

    results[model_name] = {
        'val': val_metrics,
        'test': test_metrics
    }

    print(f"\nMulti-Task {model_name} Final Results:")
    print("Validation:", val_metrics)
    print("Test:", test_metrics)

    import json
    with open('multitask_swin_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    del final_model_for_eval
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- Prozessgruppe am Ende zerstören ---
dist.destroy_process_group()