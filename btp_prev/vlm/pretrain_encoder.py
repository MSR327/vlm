"""
pretrain_encoder.py
=============================================================================
Domain Adaptation & Supervised Pre-Training of MultimodalEdgeEncoder on Town01.
Implements the LMDrive multi-task representation learning recipe:
1. Steering Angle Prediction (MSE Loss)
2. Ego-Speed Prediction (MSE Loss)

Output:
    Saves driving-specialized encoder checkpoint to:
    models/pretrained_vlm_encoder.pth

Usage:
    python pretrain_encoder.py --data data_collected_town01 --epochs 40 --batch_size 32
=============================================================================
"""

import os
import glob
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from multimodal_encoder import MultimodalEdgeEncoder, COMMAND_VOCAB, NUM_COMMANDS
from parameters import VLM_LATENT_DIM, NUM_CAMERAS, PRETRAINED_MODEL_PATH


class CarlaDrivingDataset(Dataset):
    """PyTorch Dataset loading synchronized 4-Camera (360° Surround) + LiDAR + Telemetry frames."""
    def __init__(self, data_dir, is_train=True, train_ratio=0.85):
        self.data_dir = data_dir
        self.labels_file = os.path.join(data_dir, "labels.csv")
        self.frames_dir = os.path.join(data_dir, "frames")

        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Labels CSV not found at: {self.labels_file}")

        df = pd.read_csv(self.labels_file)
        
        # Calculate prev_speed before shuffling
        df["prev_speed_kmh"] = df["speed_kmh"].shift(1)
        
        # Shuffle dataset
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # Train / Validation Split
        n_total = len(df)
        n_train = int(n_total * train_ratio)
        if is_train:
            self.df = df.iloc[:n_train].reset_index(drop=True)
        else:
            self.df = df.iloc[n_train:].reset_index(drop=True)

        print(f" Loaded {'Train' if is_train else 'Validation'} Split: {len(self.df)} frames from {data_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frame_id = str(row["frame_id"]).zfill(6)
        npz_path = os.path.join(self.frames_dir, f"{frame_id}.npz")

        data = np.load(npz_path)
        front = data["front"] # (80, 160, 3) uint8
        left  = data["left"]
        right = data["right"]
        rear  = data["rear"] if "rear" in data else np.zeros_like(front)
        lidar = data["lidar"] # (80, 160, 1) float32

        # Convert images to (3, 80, 160) float32 in [0, 1]
        front_t = torch.from_numpy(np.transpose(front, (2, 0, 1))).float() / 255.0
        left_t  = torch.from_numpy(np.transpose(left, (2, 0, 1))).float() / 255.0
        right_t = torch.from_numpy(np.transpose(right, (2, 0, 1))).float() / 255.0
        rear_t  = torch.from_numpy(np.transpose(rear, (2, 0, 1))).float() / 255.0
        lidar_t = torch.from_numpy(np.transpose(lidar, (2, 0, 1))).float()

        # Telemetry labels
        steer = float(row["steer"])
        speed = float(row["speed_kmh"])

        # Infer driving command based on steering/speed dynamics for self-supervised conditioning
        prev_speed = float(row["prev_speed_kmh"]) if pd.notna(row["prev_speed_kmh"]) else None

        if speed < 1.0:
            cmd = 'emergency_stop'
        elif prev_speed is not None and speed > prev_speed + 3.0:
            cmd = 'speed_up'
        elif prev_speed is not None and speed < prev_speed - 3.0:
            cmd = 'slow_down'
        elif steer < -0.3:
            cmd = 'turn_left'
        elif steer > 0.3:
            cmd = 'turn_right'
        elif steer < -0.1:
            cmd = 'shift_left_lane'
        elif steer > 0.1:
            cmd = 'shift_right_lane'
        else:
            cmd = 'keep_lane'

        return {
            "front": front_t,
            "left": left_t,
            "right": right_t,
            "rear": rear_t,
            "lidar": lidar_t,
            "command": cmd,
            "steer": torch.tensor([steer], dtype=torch.float32),
            "speed": torch.tensor([speed], dtype=torch.float32)
        }


class PretrainModelWrapper(nn.Module):
    """Wraps MultimodalEdgeEncoder with auxiliary prediction heads for pre-training (4 Cameras + LiDAR)."""
    def __init__(self, latent_dim=VLM_LATENT_DIM):
        super().__init__()
        self.encoder = MultimodalEdgeEncoder(latent_dim=latent_dim, num_cameras=NUM_CAMERAS, use_lidar=True)
        
        # Auxiliary Task Heads
        self.steer_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self.speed_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, front, left, right, rear, lidar, command):
        # Extract visual-linguistic latent representation with full autograd gradient flow
        obs = self.encoder(front, left, right, rear, lidar, command, telemetry=None)
        vlm_latent = obs[:, :self.encoder.latent_dim]
        
        pred_steer = self.steer_head(vlm_latent)
        pred_speed = self.speed_head(vlm_latent)
        return pred_steer, pred_speed


def main():
    parser = argparse.ArgumentParser(description="Pre-Train Multimodal Encoder on Town01")
    parser.add_argument("--data", type=str, default="data_collected_town01", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--out", type=str, default=PRETRAINED_MODEL_PATH, help="Output checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Pre-training on Device: {device} (4 Cameras + LiDAR = 1000 tokens)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1. Datasets & Loaders
    train_dataset = CarlaDrivingDataset(args.data, is_train=True)
    val_dataset   = CarlaDrivingDataset(args.data, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Model & Optimizer
    model = PretrainModelWrapper().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion_steer = nn.MSELoss()
    criterion_speed = nn.MSELoss()

    best_val_loss = float("inf")

    print(f"\n Starting Auxiliary Pre-Training for {args.epochs} Epochs...")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_steer_loss = 0.0
        train_speed_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}]"):
            front = batch["front"].to(device)
            left  = batch["left"].to(device)
            right = batch["right"].to(device)
            rear  = batch["rear"].to(device)
            lidar = batch["lidar"].to(device)
            cmd   = batch["command"]
            gt_steer = batch["steer"].to(device)
            gt_speed = batch["speed"].to(device)

            optimizer.zero_grad()

            # Forward pass directly through encoder (4 Cameras + LiDAR)
            unified_obs = model.encoder(front, left, right, rear, lidar, cmd, telemetry=None)
            vlm_latent = unified_obs[:, :model.encoder.latent_dim]

            pred_steer = model.steer_head(vlm_latent)
            pred_speed = model.speed_head(vlm_latent)

            loss_steer = criterion_steer(pred_steer, gt_steer)
            loss_speed = 0.01 * criterion_speed(pred_speed, gt_speed) # Scaled speed loss
            loss = loss_steer + loss_speed

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * front.size(0)
            train_steer_loss += loss_steer.item() * front.size(0)

        scheduler.step()

        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_steer_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                front = batch["front"].to(device)
                left  = batch["left"].to(device)
                right = batch["right"].to(device)
                rear  = batch["rear"].to(device)
                lidar = batch["lidar"].to(device)
                cmd   = batch["command"]
                gt_steer = batch["steer"].to(device)
                gt_speed = batch["speed"].to(device)

                unified_obs = model.encoder(front, left, right, rear, lidar, cmd, telemetry=None)
                vlm_latent = unified_obs[:, :model.encoder.latent_dim]

                pred_steer = model.steer_head(vlm_latent)
                pred_speed = model.speed_head(vlm_latent)

                loss_steer = criterion_steer(pred_steer, gt_steer)
                loss_speed = 0.01 * criterion_speed(pred_speed, gt_speed)
                loss = loss_steer + loss_speed

                val_loss += loss.item() * front.size(0)
                val_steer_loss += loss_steer.item() * front.size(0)

        epoch_train_loss = train_loss / len(train_dataset)
        epoch_val_loss   = val_loss / len(val_dataset)
        epoch_val_steer  = val_steer_loss / len(val_dataset)

        print(f" Epoch [{epoch+1}/{args.epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} (Steer MSE: {epoch_val_steer:.4f})")

        # Save Best Checkpoint (encoder only)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.encoder.state_dict(), args.out)
            print(f"  --> Saved BEST encoder weights to: {args.out}")

    print(f"\n Pre-Training Complete! Best model saved to: {args.out}")


if __name__ == "__main__":
    main()
