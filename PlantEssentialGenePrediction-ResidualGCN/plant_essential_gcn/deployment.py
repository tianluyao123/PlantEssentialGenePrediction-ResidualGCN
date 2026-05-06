import json
import os
from typing import Optional

import torch
from torch import nn

from .config import DeploymentConfig, OptimizedConfig
from .features import FeatureReducer


def save_deployment_artifacts(model: nn.Module,
                              main_reducer: FeatureReducer,
                              bio_reducer: Optional[FeatureReducer],
                              config: OptimizedConfig,
                              metrics: dict,
                              pos_weight: float,
                              save_dir: str,
                              rep_id: int):

    rep_dir = os.path.join(save_dir, f"rep_{rep_id}")
    os.makedirs(rep_dir, exist_ok=True)

    print(f"\n Saving deployment artifacts for repeat {rep_id} to: {rep_dir}")


    model_path = os.path.join(rep_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  OK Model parameters: {model_path}")


    main_reducer_path = os.path.join(rep_dir, "reducer_main.pkl")
    main_reducer.save(main_reducer_path)

    if bio_reducer is not None:
        bio_reducer_path = os.path.join(rep_dir, "reducer_bio.pkl")
        bio_reducer.save(bio_reducer_path)
        print(f"  OK Biological feature reducer: {bio_reducer_path}")
    else:
        bio_reducer_path = None


    deploy_config = DeploymentConfig(
        K=config.K,
        D=config.D,
        SEQ_MAX_LEN=config.SEQ_MAX_LEN,
        REDUCED_MAIN_DIM=config.REDUCED_MAIN_DIM,
        USE_FEATURE_REDUCTION=config.USE_FEATURE_REDUCTION,
        USE_BIO_FEATURES=config.USE_BIO_FEATURES,
        CDS_FEATURE_DIM=config.CDS_FEATURE_DIM,
        AA_FEATURE_DIM=config.AA_FEATURE_DIM,
        BIO_FEATURE_DIM=config.BIO_FEATURE_DIM,
        GCN_HIDDEN=config.GCN_HIDDEN,
        GCN_LAYERS=config.GCN_LAYERS,
        DROPOUT=config.DROPOUT,
        pos_weight=pos_weight,
        optimal_threshold=0.5
    )


    config_path = os.path.join(rep_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(deploy_config.to_dict(), f, indent=2)
    print(f"  OK Configuration file: {config_path}")


    metadata = {
        "rep_id": rep_id,
        "metrics": metrics,
        "pos_weight": float(pos_weight),
        "main_feat_dim": config.REDUCED_MAIN_DIM,
        "bio_feat_dim": config.BIO_FEATURE_DIM if config.USE_BIO_FEATURES else 0,
        "total_params": sum(p.numel() for p in model.parameters()),
        "device": str(config.DEVICE)
    }
    metadata_path = os.path.join(rep_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  OK Metadata file: {metadata_path}")

    return rep_dir
