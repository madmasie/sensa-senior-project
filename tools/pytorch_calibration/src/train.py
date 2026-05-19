"""
train.py — Training and validation loops for the PM2.5 calibration model.

WHAT "TRAINING" MEANS IN NEURAL NETWORKS (for EEs):
    Think of the network weights as the coefficients of a transfer function.
    Training is the process of finding coefficients that minimise the error
    between the network's output and the true reference measurement.

    Each iteration (called a "step") does:
      1. Feed a batch of sensor readings through the network → get predictions.
      2. Compute the loss: how wrong are the predictions? (We use MSE.)
      3. Backpropagate: use calculus (chain rule) to find the gradient of the
         loss with respect to every weight — i.e., "which direction should
         each weight move to reduce the error?"
      4. Update the weights by a small step in that direction.

    One full pass over the entire training set is called an "epoch".
    We run many epochs until the validation loss stops improving.

LOSS FUNCTION — Mean Squared Error (MSE):
    MSE = average of (prediction - truth)² across the batch.
    Squaring the error penalises large misses more than small ones.
    For PM2.5 calibration, a 20 µg/m³ error is far worse than a 2 µg/m³
    error, so MSE is appropriate.

OPTIMIZER — Adam:
    Adam is an adaptive gradient-descent algorithm. It tracks the running
    mean and variance of past gradients and adjusts the effective learning
    rate per weight accordingly. This makes it robust to poor initial
    learning-rate choices. It is the standard default for regression tasks.

LEARNING RATE SCHEDULE:
    If validation loss plateaus, we halve the learning rate automatically.
    This is analogous to reducing the step size of a search algorithm when
    you get close to the minimum.

EARLY STOPPING:
    If validation loss does not improve for `patience` consecutive epochs,
    training stops. The best checkpoint is then loaded before returning.
    This prevents the model from "overfitting" — memorising the training data
    rather than learning the underlying sensor response curve.
"""

import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one full pass over the training data, updating weights after each batch.

    Args:
        model:     The SensaCalibNet being trained.
        loader:    DataLoader that yields (X_batch, y_batch) tuples.
        optimizer: Adam optimiser that updates model weights.
        criterion: Loss function (MSELoss).
        device:    'cpu' or 'cuda' — where tensors are stored.

    Returns:
        Average training loss across all batches in this epoch.
    """
    # Switch the model to training mode.
    # This enables BatchNorm's running-stat updates and would enable Dropout
    # if we were using it. It must be called at the start of every training epoch
    # because validate() switches the model back to eval mode.
    model.train()

    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        # Move tensors to the compute device.
        # On CPU (the default), this is a no-op. On GPU, it transfers memory.
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Clear gradients from the previous batch.
        # In PyTorch, gradients accumulate by default — if you forget this,
        # the weight updates will be wrong from the second batch onwards.
        optimizer.zero_grad()

        # Forward pass: feed inputs through the network to get predictions.
        predictions = model(X_batch)          # shape: (batch, 1)

        # Compute the loss (mean squared error between predictions and targets).
        loss = criterion(predictions, y_batch)

        # Backward pass: compute d(loss)/d(weight) for every trainable weight.
        # PyTorch builds a computation graph during the forward pass and
        # traverses it in reverse here (hence "backpropagation").
        loss.backward()

        # Apply the weight updates computed by the backward pass.
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on the validation set without updating weights.

    Validation loss tells us whether the model is generalising to unseen data
    or just memorising the training set (overfitting). We use it to decide
    when to stop training and which checkpoint to keep.

    Args:
        model:     Trained (or partially trained) SensaCalibNet.
        loader:    Validation DataLoader.
        criterion: Loss function (same as training).
        device:    Compute device.

    Returns:
        Average validation loss across all batches.
    """
    # Switch to evaluation mode: disables BatchNorm parameter updates.
    # This must be paired with torch.no_grad() below.
    model.eval()

    total_loss = 0.0
    n_batches = 0

    # torch.no_grad() tells PyTorch not to build the computation graph.
    # Since we are not calling .backward(), we do not need gradients.
    # This saves memory and runs faster.
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> Tuple[nn.Module, list]:
    """
    Full training loop: runs epochs until early stopping, returns best model.

    Args:
        model:        The SensaCalibNet to train (weights initialised randomly).
        train_loader: DataLoader for training data.
        val_loader:   DataLoader for validation data.
        config:       Dict loaded from config.yaml (training sub-section).
        device:       Compute device.

    Returns:
        (trained_model, history) where history is a list of dicts with
        keys 'epoch', 'train_loss', 'val_loss', 'lr' — useful for plotting.
    """
    save_dir = Path(config['export']['model_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / config['export']['pytorch_checkpoint']

    train_cfg = config['training']
    lr            = train_cfg['learning_rate']
    weight_decay  = train_cfg.get('weight_decay', 1e-4)
    max_epochs    = train_cfg['epochs']
    patience      = train_cfg.get('early_stopping_patience', 20)

    # Adam: adaptive moment estimation optimiser.
    # weight_decay adds L2 regularisation — a small penalty proportional to
    # the magnitude of each weight that discourages overfitting.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Reduce the learning rate by half if validation loss has not improved
    # for 7 consecutive epochs. patience=7 gives the model time to escape a
    # flat region before we slow down.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7 #, verbose=False
    )

    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []

    model.to(device)
    print(f"\nTraining on: {device}")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}  {'Time':>7}")
    print("─" * 58)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed    = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
            f"{current_lr:>10.2e}  {elapsed:>5.1f}s"
        )

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'lr': current_lr
        })

        # Save the model whenever we achieve a new best validation loss.
        # We save only the weights (state_dict), not the full model object —
        # this is the recommended practice in PyTorch.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"         ✓ Checkpoint saved (val_loss={val_loss:.6f})")
        else:
            epochs_no_improve += 1

        # If the model has not improved for `patience` epochs, stop early.
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break

    # Restore the best weights before returning.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")

    return model, history
