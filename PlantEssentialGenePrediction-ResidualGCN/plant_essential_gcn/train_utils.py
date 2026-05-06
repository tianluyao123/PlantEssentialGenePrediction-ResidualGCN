import os

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.data import WeightedRandomSampler

from .features import FeatureReducer
from .losses import FocalLoss, CostSensitiveLoss
from .metrics import evaluate_youden_on_loader


def train_fold_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                     config, fold: int) -> tuple:

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

    train_labels = []
    for batch in train_loader.dataset:
        try:
            y_val = batch.y.item() if batch.y.numel() == 1 else batch.y[0].item()
            train_labels.append(y_val)
        except Exception:
            train_labels.append(0)

    class_counts = np.bincount(train_labels)
    print(f"  - Training set distribution: negative class={class_counts[0]}, positive class={class_counts[1] if len(class_counts) > 1 else 0}")

    if len(class_counts) == 2 and class_counts[1] > 0:
        raw_weight = (class_counts[0] / class_counts[1]) * config.POS_WEIGHT_MULTIPLIER
        pos_weight = min(raw_weight, config.POS_WEIGHT_CAP)
    else:
        pos_weight = 1.0

    class_weights = torch.tensor([1.0, pos_weight], device=config.DEVICE, dtype=torch.float32)
    print(f"  - Class weights: negative class=1.0, positive class={pos_weight:.2f}")

    if config.USE_FOCAL_LOSS:
        base_criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA,
                                   weight=class_weights, reduction='none')
    else:
        base_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING,
                                             reduction='none')
    criterion = CostSensitiveLoss(base_criterion, fn_penalty=config.FN_PENALTY, device=config.DEVICE)

    warmup_scheduler = LinearLR(optimizer, start_factor=config.WARMUP_START_FACTOR,
                                total_iters=config.WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.MAX_EPOCHS - config.WARMUP_EPOCHS),
                                         eta_min=config.LR * 0.01)

    best_model_path = os.path.join(config.SAVE_BASE_PATH, f"fold_{fold}_best_train_loss_model.pth")
    best_train_loss = float('inf')
    best_train_loss_when_youden1 = float('inf')
    best_youden_j = -float('inf')
    best_epoch = 0
    no_improve_epochs = 0
    train_history = []

    print(f"\n Training fold {fold + 1}/{config.N_FOLDS}: best model is selected by the minimum train_loss")

    for epoch in range(config.MAX_EPOCHS):
        model.train()
        total_loss = 0.0
        seen = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(config.DEVICE, non_blocking=True)
            try:
                if hasattr(batch, 'x_p') and batch.x_p.dtype != torch.float32:
                    batch.x_p = batch.x_p.float()
                if hasattr(batch, 'x_f') and batch.x_f.dtype != torch.float32:
                    batch.x_f = batch.x_f.float()

                logits = model(batch)
                targets = batch.y.view(-1)

                loss = criterion(logits, targets) / config.GRADIENT_ACCUMULATION_STEPS
                loss.backward()

                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS * targets.numel()
                seen += targets.numel()
            except Exception as e:
                print(f"Warning: Training error: {e}")
                continue

        if len(train_loader) % config.GRADIENT_ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(1, seen)

        if epoch < config.WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']


        current_youden_j, current_sn, current_sp = evaluate_youden_on_loader(model, val_loader, config)

        improved = False
        save_reason = ""

        if current_youden_j == 1.0:

            if avg_loss < best_train_loss_when_youden1:
                best_train_loss_when_youden1 = avg_loss
                best_train_loss = avg_loss
                best_youden_j = current_youden_j
                improved = True
                save_reason = "Youden_J=1.0 and train_loss decreased"
        else:

            if current_youden_j > best_youden_j:
                best_youden_j = current_youden_j
                best_train_loss = avg_loss
                improved = True
                save_reason = "Youden_J improved"

        if improved:
            best_epoch = epoch + 1
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            print(
                f"  - Epoch {epoch + 1:03d}: {save_reason}, saved model "
                f"(Youden_J={current_youden_j:.6f}, SN={current_sn:.4f}, SP={current_sp:.4f}, "
                f"train_loss={avg_loss:.6f}, lr={current_lr:.2e})"
            )
        else:
            no_improve_epochs += 1
            if (epoch + 1) % 10 == 0 or no_improve_epochs == 1:
                if best_youden_j == 1.0:
                    print(
                        f"  - Epoch {epoch + 1:03d}: Youden_J={current_youden_j:.6f}, "
                        f"train_loss={avg_loss:.6f}, did not improve the best loss after Youden_J reached 1.0 "
                        f"{best_train_loss_when_youden1:.6f}, consecutive non-improving epochs {no_improve_epochs}/{config.EARLY_STOP_PATIENCE}"
                    )
                else:
                    print(
                        f"  - Epoch {epoch + 1:03d}: Youden_J={current_youden_j:.6f}, "
                        f"did not exceed the best Youden_J {best_youden_j:.6f}, consecutive non-improving epochs "
                        f"{no_improve_epochs}/{config.EARLY_STOP_PATIENCE}"
                    )

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": round(float(avg_loss), 6),
            "youden_j": float(current_youden_j),
            "sn": float(current_sn),
            "sp": float(current_sp),
            "best_youden_j": float(best_youden_j),
            "best_train_loss": round(float(best_train_loss), 6),
            "best_train_loss_when_youden1": round(float(best_train_loss_when_youden1), 6) if best_train_loss_when_youden1 < float('inf') else None,
            "best_epoch": best_epoch,
            "no_improve_epochs": no_improve_epochs,
            "lr": float(current_lr),
        })

        if no_improve_epochs >= config.EARLY_STOP_PATIENCE:
            print(
                f"  - consecutive {config.EARLY_STOP_PATIENCE} epochs without a valid improvement, stopping early; "
                f"using epoch {best_epoch} as the best model"
            )
            break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        print(
            f"  - Loaded best model: epoch={best_epoch}, best_Youden_J={best_youden_j:.6f}, "
            f"best_train_loss={best_train_loss:.6f}"
        )

    return model, {"best_youden_j": best_youden_j, "best_train_loss": best_train_loss, "epoch": best_epoch, "history": train_history}, pos_weight


def get_fold_dataloaders(train_idx, val_idx, cv_graphs, cv_labels, cv_feats, cv_bio_feats, config):
    train_graphs = [cv_graphs[i] for i in train_idx]
    val_graphs = [cv_graphs[i] for i in val_idx]
    train_labels = cv_labels[train_idx]
    val_labels = cv_labels[val_idx]
    train_feats = cv_feats[train_idx]
    val_feats = cv_feats[val_idx]
    train_bio_feats = cv_bio_feats[train_idx] if cv_bio_feats is not None else None
    val_bio_feats = cv_bio_feats[val_idx] if cv_bio_feats is not None else None

    class_counts = np.bincount(train_labels)
    if len(class_counts) == 2 and class_counts[1] > 0:
        raw_weight = (class_counts[0] / class_counts[1]) * config.POS_WEIGHT_MULTIPLIER
        pos_weight = min(raw_weight, config.POS_WEIGHT_CAP)
    else:
        pos_weight = 1.0

    sample_weights = np.where(train_labels == 1, pos_weight, 1.0)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_graphs), replacement=True)

    main_reducer = FeatureReducer(config.REDUCED_MAIN_DIM)
    bio_reducer = FeatureReducer(config.BIO_FEATURE_DIM) if config.USE_BIO_FEATURES else None

    if config.USE_FEATURE_REDUCTION and len(train_feats) > 0:
        _ = main_reducer.fit_transform(train_feats)
        if config.USE_BIO_FEATURES and train_bio_feats is not None and len(train_bio_feats) > 0:
            _ = bio_reducer.fit_transform(train_bio_feats)
    else:
        main_reducer.is_fitted = True
        if bio_reducer:
            bio_reducer.is_fitted = True


    for i, (graph, feat) in enumerate(zip(train_graphs, train_feats)):
        try:
            reduced_feat = main_reducer.transform(feat.reshape(1, -1)).flatten()
            if config.USE_BIO_FEATURES and train_bio_feats is not None and i < len(train_bio_feats):
                bio_feat = train_bio_feats[i]
                if bio_reducer is not None and len(bio_feat) > 0:
                    reduced_bio_feat = bio_reducer.transform(bio_feat.reshape(1, -1)).flatten()
                    combined_feat = np.concatenate([reduced_feat, reduced_bio_feat])
                else:
                    combined_feat = np.concatenate([reduced_feat, bio_feat])
            else:
                combined_feat = reduced_feat

            graph.x_p = torch.tensor(combined_feat.astype(np.float32), dtype=torch.float32).unsqueeze(0)
            graph.y = torch.tensor([train_labels[i]], dtype=torch.long)
        except Exception as e:
            print(f"Warning: Failed to process training features: {e}")

    for i, (graph, feat) in enumerate(zip(val_graphs, val_feats)):
        try:
            reduced_feat = main_reducer.transform(feat.reshape(1, -1)).flatten()
            if config.USE_BIO_FEATURES and val_bio_feats is not None and i < len(val_bio_feats):
                bio_feat = val_bio_feats[i]
                if bio_reducer is not None and len(bio_feat) > 0:
                    reduced_bio_feat = bio_reducer.transform(bio_feat.reshape(1, -1)).flatten()
                    combined_feat = np.concatenate([reduced_feat, reduced_bio_feat])
                else:
                    combined_feat = np.concatenate([reduced_feat, bio_feat])
            else:
                combined_feat = reduced_feat

            graph.x_p = torch.tensor(combined_feat.astype(np.float32), dtype=torch.float32).unsqueeze(0)
            graph.y = torch.tensor([val_labels[i]], dtype=torch.long)
        except Exception as e:
            print(f"Warning: Failed to process validation features: {e}")

    train_loader = DataLoader(train_graphs, batch_size=config.BATCH_SIZE, sampler=sampler,
                              pin_memory=True, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_graphs, batch_size=config.BATCH_SIZE, shuffle=False, pin_memory=True)
    return train_loader, val_loader, main_reducer, bio_reducer
