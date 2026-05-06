import json
import os
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from plant_essential_gcn.config import OptimizedConfig
from plant_essential_gcn.data import load_kfold_dataset
from plant_essential_gcn.deployment import save_deployment_artifacts
from plant_essential_gcn.features import FeatureReducer
from plant_essential_gcn.losses import FocalLoss, CostSensitiveLoss
from plant_essential_gcn.metrics import evaluate_model, evaluate_youden_on_loader
from plant_essential_gcn.model import SimpleModel


def parameter_tuning():
    print("=" * 80)
    print(" SN-oriented training with deployment export for all repeats")
    print("=" * 80)

    torch.manual_seed(OptimizedConfig.RANDOM_STATE)
    np.random.seed(OptimizedConfig.RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = OptimizedConfig()


    if os.path.exists(config.DEPLOYMENT_PATH):
        import shutil
        shutil.rmtree(config.DEPLOYMENT_PATH)
    os.makedirs(config.DEPLOYMENT_PATH, exist_ok=True)

    print(f"\n Key configuration:")
    print(f"  - Sequence length: {config.SEQ_MAX_LEN}")
    print(f"  - Feature dimension: {config.REDUCED_MAIN_DIM}")
    print(f"  - FN penalty: {config.FN_PENALTY}x (balanced setting)")
    print(f"  - Deployment directory: {config.DEPLOYMENT_PATH}")

    K, d = config.K, config.D


    global_config = {
        "K": K, "D": d,
        "N_REPEATS": config.N_REPEATS,
        "base_path": config.DEPLOYMENT_PATH,
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(config.DEPLOYMENT_PATH, "global_config.json"), 'w') as f:
        json.dump(global_config, f, indent=2)

    try:
        (first_cv_graphs, first_cv_labels, first_cv_feats, first_cv_bio_feats,
         first_test_graphs, first_test_labels, first_test_feats, first_test_bio_feats,
         first_fold_splits, test_indices) = load_kfold_dataset(
            config, K, d, random_state=config.RANDOM_STATE, test_indices=None, use_predefined_split=True)

        all_repetition_results = []

        for rep in range(config.N_REPEATS):
            current_seed = config.RANDOM_STATE + rep + 1
            print(f"\n{'=' * 80}")
            print(f" Repeat {rep + 1}/{config.N_REPEATS} (seed={current_seed})")
            print(f"{'=' * 80}")

            torch.manual_seed(current_seed)
            np.random.seed(current_seed)

            (cv_graphs, cv_labels, cv_feats, cv_bio_feats,
             test_graphs, test_labels, test_feats, test_bio_feats,
             fold_splits, _) = load_kfold_dataset(
                config, K, d, random_state=current_seed, test_indices=test_indices, use_predefined_split=True)

            main_feat_dim = config.REDUCED_MAIN_DIM if config.USE_FEATURE_REDUCTION else cv_feats.shape[1]
            bio_feat_dim = config.BIO_FEATURE_DIM if config.USE_BIO_FEATURES else 0


            print(f"\n Training the final model on all cross-validation data...")


            all_train_graphs = []
            all_train_feats = cv_feats
            all_train_bio_feats = cv_bio_feats


            main_reducer = FeatureReducer(config.REDUCED_MAIN_DIM)
            bio_reducer = FeatureReducer(config.BIO_FEATURE_DIM) if config.USE_BIO_FEATURES else None

            if config.USE_FEATURE_REDUCTION:
                _ = main_reducer.fit_transform(all_train_feats)
                if config.USE_BIO_FEATURES and cv_bio_feats is not None:
                    _ = bio_reducer.fit_transform(cv_bio_feats)


            for i, (graph, feat) in enumerate(zip(cv_graphs, cv_feats)):
                try:
                    reduced_feat = main_reducer.transform(feat.reshape(1, -1)).flatten()
                    if config.USE_BIO_FEATURES and cv_bio_feats is not None and i < len(cv_bio_feats):
                        bio_feat = cv_bio_feats[i]
                        if bio_reducer is not None and len(bio_feat) > 0:
                            reduced_bio_feat = bio_reducer.transform(bio_feat.reshape(1, -1)).flatten()
                            combined_feat = np.concatenate([reduced_feat, reduced_bio_feat])
                        else:
                            combined_feat = np.concatenate([reduced_feat, bio_feat])
                    else:
                        combined_feat = reduced_feat

                    graph.x_p = torch.tensor(combined_feat.astype(np.float32), dtype=torch.float32).unsqueeze(0)
                    graph.y = torch.tensor([cv_labels[i]], dtype=torch.long)
                    all_train_graphs.append(graph)
                except Exception as e:
                    print(f"Warning: Feature processing failed: {e}")


            full_train_loader = DataLoader(all_train_graphs, batch_size=config.BATCH_SIZE,
                                           shuffle=True, pin_memory=True, drop_last=True)


            model = SimpleModel(config, main_feat_dim=main_feat_dim, bio_feat_dim=bio_feat_dim).to(config.DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)

            class_counts = np.bincount(cv_labels)
            if len(class_counts) == 2 and class_counts[1] > 0:
                pos_weight = min((class_counts[0] / class_counts[1]) * config.POS_WEIGHT_MULTIPLIER,
                                 config.POS_WEIGHT_CAP)
            else:
                pos_weight = 1.0

            class_weights = torch.tensor([1.0, pos_weight], device=config.DEVICE, dtype=torch.float32)

            if config.USE_FOCAL_LOSS:
                base_criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA,
                                           weight=class_weights, reduction='none')
            else:
                base_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
            criterion = CostSensitiveLoss(base_criterion, fn_penalty=config.FN_PENALTY, device=config.DEVICE)


            warmup_scheduler = LinearLR(optimizer, start_factor=config.WARMUP_START_FACTOR,
                                        total_iters=config.WARMUP_EPOCHS)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.MAX_EPOCHS - config.WARMUP_EPOCHS),
                                                 eta_min=config.LR * 0.01)

            best_youden_j = -float('inf')
            best_train_loss = float('inf')
            best_train_loss_when_youden1 = float('inf')
            best_epoch = 0
            no_improve_epochs = 0
            best_model_state = None
            train_history = []
            best_youden_loss_path = os.path.join(config.SAVE_BASE_PATH, f"rep_{rep + 1}_best_youden_loss_model.pth")
            full_train_eval_loader = DataLoader(all_train_graphs, batch_size=config.BATCH_SIZE,
                                                shuffle=False, pin_memory=True, drop_last=False)

            for epoch in range(config.MAX_EPOCHS):
                model.train()
                total_loss = 0.0
                seen = 0
                optimizer.zero_grad()

                for batch_idx, batch in enumerate(full_train_loader):
                    batch = batch.to(config.DEVICE)
                    if batch.x_p.dtype != torch.float32:
                        batch.x_p = batch.x_p.float()

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

                if len(full_train_loader) % config.GRADIENT_ACCUMULATION_STEPS != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                avg_train_loss = total_loss / max(1, seen)

                if epoch < config.WARMUP_EPOCHS:
                    warmup_scheduler.step()
                else:
                    cosine_scheduler.step()

                current_lr = optimizer.param_groups[0]['lr']


                current_youden_j, current_sn, current_sp = evaluate_youden_on_loader(model, full_train_eval_loader, config)

                improved = False
                save_reason = ""

                if current_youden_j == 1.0:
                    if avg_train_loss < best_train_loss_when_youden1:
                        best_train_loss_when_youden1 = avg_train_loss
                        best_train_loss = avg_train_loss
                        best_youden_j = current_youden_j
                        improved = True
                        save_reason = "Youden_J=1.0 and train_loss decreased"
                else:
                    if current_youden_j > best_youden_j:
                        best_youden_j = current_youden_j
                        best_train_loss = avg_train_loss
                        improved = True
                        save_reason = "Youden_J improved"

                if improved:
                    best_epoch = epoch + 1
                    no_improve_epochs = 0
                    best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    torch.save(best_model_state, best_youden_loss_path)
                    print(
                        f"  - Epoch {epoch + 1:03d}: {save_reason}, saved current best model "
                        f"(Youden_J={current_youden_j:.6f}, SN={current_sn:.4f}, SP={current_sp:.4f}, "
                        f"train_loss={avg_train_loss:.6f}, lr={current_lr:.2e})"
                    )
                else:
                    no_improve_epochs += 1
                    if (epoch + 1) % 10 == 0 or no_improve_epochs == 1:
                        if best_youden_j == 1.0:
                            print(
                                f"  - Epoch {epoch + 1:03d}: Youden_J={current_youden_j:.6f}, "
                                f"train_loss={avg_train_loss:.6f}, did not improve the best loss after Youden_J reached 1.0 "
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
                    "train_loss": round(float(avg_train_loss), 6),
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


            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(
                    f"  - Loaded best model: epoch={best_epoch}, best_Youden_J={best_youden_j:.6f}, "
                    f"best_train_loss={best_train_loss:.6f}"
                )

            history_path = os.path.join(config.SAVE_BASE_PATH, f"rep_{rep + 1}_youden_loss_history.csv")
            pd.DataFrame(train_history).to_csv(history_path, index=False, encoding='utf-8-sig')
            print(f"  - Youden_J + train_loss training history saved: {history_path}")


            test_graphs_processed = []
            for i, (graph, feat) in enumerate(zip(test_graphs, test_feats)):
                try:
                    reduced_feat = main_reducer.transform(feat.reshape(1, -1)).flatten()
                    if config.USE_BIO_FEATURES and test_bio_feats is not None and i < len(test_bio_feats):
                        bio_feat = test_bio_feats[i]
                        if bio_reducer is not None and len(bio_feat) > 0:
                            reduced_bio_feat = bio_reducer.transform(bio_feat.reshape(1, -1)).flatten()
                            combined_feat = np.concatenate([reduced_feat, reduced_bio_feat])
                        else:
                            combined_feat = np.concatenate([reduced_feat, bio_feat])
                    else:
                        combined_feat = reduced_feat

                    graph.x_p = torch.tensor(combined_feat.astype(np.float32), dtype=torch.float32).unsqueeze(0)
                    graph.y = torch.tensor([test_labels[i]], dtype=torch.long)
                    test_graphs_processed.append(graph)
                except Exception as e:
                    print(f"Warning: Test feature processing failed: {e}")

            test_loader = DataLoader(test_graphs_processed, batch_size=config.BATCH_SIZE,
                                     shuffle=False, pin_memory=True)
            test_metrics = evaluate_model(model, test_loader, config, "test set", save_report=False)


            save_deployment_artifacts(
                model=model,
                main_reducer=main_reducer,
                bio_reducer=bio_reducer,
                config=config,
                metrics=test_metrics,
                pos_weight=pos_weight,
                save_dir=config.DEPLOYMENT_PATH,
                rep_id=rep + 1
            )

            all_repetition_results.append({
                'rep': rep + 1,
                'test_metrics': test_metrics,
                'pos_weight': pos_weight,
                'seed': current_seed
            })

            print(f"\n Repeat {rep + 1} completed and saved")
            print(f"  - test set: SN={test_metrics['SN']:.4f}, SP={test_metrics['SP']:.4f}, "
                  f"Youden_J={test_metrics['Youden_J']:.4f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        summary_df = pd.DataFrame([
            {
                'rep': r['rep'],
                'SN': r['test_metrics']['SN'],
                'SP': r['test_metrics']['SP'],
                'ACC': r['test_metrics']['ACC'],
                'AUC': r['test_metrics']['AUC'],
                'Youden_J': r['test_metrics']['Youden_J'],
                'G_mean': r['test_metrics']['G_mean'],
                'MCC': r['test_metrics']['MCC']
            } for r in all_repetition_results
        ])

        summary_path = os.path.join(config.DEPLOYMENT_PATH, "repetitions_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

        print(f"\n{'=' * 80}")
        print(" Five-repeat summary with all models saved")
        print(f"{'=' * 80}")
        print(summary_df.to_string(index=False))
        print(f"\nAll deployment models were saved to: {config.DEPLOYMENT_PATH}")


        print("\n" + "=" * 80)
        print(" Deployment example for ensemble prediction:")
        print("=" * 80)
        print("""
from deployment_predictor import EnsemblePredictor

Load five models for ensemble prediction
predictor = EnsemblePredictor(r"D:\\outputs\\plant_gcn\\deployment_models")
results = predictor.predict_batch([
    {"cds_seq": "ATG...", "aa_seq": "MKT..."},
    {"cds_seq": "ATG...", "aa_seq": "MKT..."}
])

The output includes probabilities and ensemble voting results
        """)

    except Exception as e:
        print(f"Error: Execution failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parameter_tuning()
