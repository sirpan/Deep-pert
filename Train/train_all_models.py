"""
Unified training entrypoint to run PCA preprocessing, AE, DAE, VAE, and baselines (ICA/RP) in one shot.

Usage example (Windows PowerShell):
  python code/train_all_models.py ^
    --models PCA,AE,DAE,ICA,RP ^
    --cancer-types A549,HEPG2 ^
    --latent-dims 5,10,25,50 ^
    --runs 5 ^
    --input-root D:\\data\\PCA_inputs ^
    --perturbation-root D:\\data\\perturbation_profiles ^
    --output-root D:\\outputs

Notes:
  - PCA creates: {input-root}/{cancer}/{cancer}_DATA_TOP2_JOINED_PCA_XXXL.tsv (+ COMPONENTS)
  - AE/DAE expect per-cancer PCA TSVs named like: {cancer}_DATA_TOP2_JOINED_PCA_*L.tsv inside input-root/{cancer}/
  - ICA/RP expect perturbation matrices at: perturbation-root/{cancer}/perturbation_matrix.csv
  - VAE will be skipped if the VAE implementation module is not available in this workspace.
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple


# -------------------------------
# Optional imports for deep models
# -------------------------------

UNIFIED_DEEP = None
try:
    import unified_deep_models as UNIFIED_DEEP  # type: ignore
except Exception:
    UNIFIED_DEEP = None


# -------------------------------
# Baseline model implementations (ICA/RP) and PCA precompute
# -------------------------------

def _determine_pca_components(num_samples: int) -> int:
    # Mirror the heuristic in Create_PCs_for_DeepLearning_Models_GEN.py
    # Current thresholds all map to 100; keep structure for future flexibility
    if num_samples >= 1000:
        return 100
    elif (num_samples >= 500) and (num_samples < 1000):
        return 100
    elif (num_samples >= 250) and (num_samples < 500):
        return 100
    elif (num_samples >= 100) and (num_samples < 250):
        return 100
    elif (num_samples >= 50) and (num_samples < 100):
        return 100
    elif num_samples <= 25:
        return 100
    return 100


def run_pca_precompute(perturbation_root: str, input_root: str, cancer_type: str, pca_components: int = 0) -> int:
    """Create PCA inputs for deep models and write into input_root/{cancer}.

    Returns the number of components written.
    """
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    input_folder = os.path.join(perturbation_root, cancer_type)
    output_folder = os.path.join(input_root, cancer_type)
    os.makedirs(output_folder, exist_ok=True)

    data_path = os.path.join(input_folder, "perturbation_matrix.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing perturbation matrix: {data_path}")

    data_df = pd.read_csv(data_path, index_col=0)
    training_data = np.nan_to_num(data_df.values)
    n_components = pca_components if pca_components and pca_components > 0 else _determine_pca_components(training_data.shape[0])

    pca = PCA(n_components=n_components)
    pca.fit(training_data)
    components = pca.components_

    component_df = pd.DataFrame(components.T, index=data_df.columns)
    component_df.to_csv(
        os.path.join(output_folder, f"{cancer_type}_DATA_TOP2_JOINED_PCA_{n_components}L_COMPONENTS.tsv"),
        sep="\t",
    )

    encoded_data = pca.transform(training_data)
    encoded_df = pd.DataFrame(encoded_data, index=data_df.index)
    encoded_df.to_csv(
        os.path.join(output_folder, f"{cancer_type}_DATA_TOP2_JOINED_PCA_{n_components}L.tsv"),
        sep="\t",
    )

    return n_components

def run_ica_baseline(perturbation_root: str, output_root: str, cancer_type: str, n_components: int = 150) -> None:
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import FastICA

    input_folder = os.path.join(perturbation_root, cancer_type)
    output_folder = os.path.join(output_root, f"{cancer_type}_ICA")
    os.makedirs(output_folder, exist_ok=True)

    data_path = os.path.join(input_folder, "perturbation_matrix.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing perturbation matrix: {data_path}")

    data_df = pd.read_csv(data_path, index_col=0)
    training_data = np.nan_to_num(data_df.values)

    ica = FastICA(n_components=n_components, random_state=12345, max_iter=100000)
    ica.fit(training_data)

    components = ica.components_
    component_df = pd.DataFrame(components.T, index=data_df.columns)
    component_df.to_csv(
        os.path.join(output_folder, f"{cancer_type}_DATA_TOP2_JOINED_ICA_{n_components}L_COMPONENTS.tsv"),
        sep="\t",
    )

    encoded_data = ica.transform(training_data)
    encoded_df = pd.DataFrame(encoded_data, index=data_df.index)
    encoded_df.to_csv(
        os.path.join(output_folder, f"{cancer_type}_DATA_TOP2_JOINED_ICA_{n_components}L.tsv"),
        sep="\t",
    )


def run_rp_baseline(perturbation_root: str, output_root: str, cancer_type: str, n_components: int = 150) -> None:
    import numpy as np
    import pandas as pd
    from sklearn.random_projection import GaussianRandomProjection

    input_folder = os.path.join(perturbation_root, cancer_type)
    output_folder = os.path.join(output_root, f"{cancer_type}_RP")
    os.makedirs(output_folder, exist_ok=True)

    data_path = os.path.join(input_folder, "perturbation_matrix.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing perturbation matrix: {data_path}")

    data_df = pd.read_csv(data_path, index_col=0)
    training_data = np.nan_to_num(data_df.values)

    transformer = GaussianRandomProjection(n_components=n_components, random_state=12345)
    transformer.fit(training_data)

    components = transformer.components_
    component_df = pd.DataFrame(components.T, index=data_df.columns)
    component_df.to_csv(
        os.path.join(output_folder, f"{cancer_type}_DATA_TOP2_JOINED_RP_{n_components}L_COMPONENTS.tsv"),
        sep="\t",
    )

    encoded_data = transformer.transform(training_data)
    encoded_df = pd.DataFrame(encoded_data, index=data_df.index)
    encoded_df.to_csv(
        os.path.join(output_folder, f"{cancer_type}_DATA_TOP2_JOINED_RP_{n_components}L.tsv"),
        sep="\t",
    )


# -------------------------------
# Deep model orchestrators
# -------------------------------

DIM_MAPPING: Dict[int, Tuple[int, int]] = {
    5: (100, 25),
    10: (250, 50),
    25: (250, 100),
    50: (250, 100),
    95: (250, 100),
    100: (250, 100),
}


def run_ae(input_root: str, output_root: str, cancer_types: List[str], latent_dims: List[int], runs: int) -> Dict[str, list]:
    if UNIFIED_DEEP is None:
        print("[AE] unified_deep_models not found. Skipping AE training.")
        return {}

    results = defaultdict(list)
    for latent_dim in latent_dims:
        dim1, dim2 = DIM_MAPPING.get(latent_dim, (250, 100))
        for cancer_type in cancer_types:
            input_base = os.path.join(input_root, cancer_type)
            output_base = os.path.join(output_root, "AE_embedding", cancer_type)
            os.makedirs(output_base, exist_ok=True)

            for run in range(runs):
                print(f"\n[AE] {cancer_type} | run={run} | latent={latent_dim}")
                try:
                    UNIFIED_DEEP.run_model(
                        model_type="AE",
                        cancer_type=cancer_type,
                        dim1=dim1,
                        dim2=dim2,
                        latent_dim=latent_dim,
                        run=run,
                        input_base=input_base + os.sep,
                        output_base=output_base,
                        result_collector=results,
                        pca_method="PCA",
                    )
                except Exception as e:
                    print(f"[AE] {cancer_type} run {run} failed: {e}")
                    continue

    return results


def run_dae(input_root: str, output_root: str, cancer_types: List[str], latent_dims: List[int], runs: int) -> Dict[str, list]:
    if UNIFIED_DEEP is None:
        print("[DAE] unified_deep_models not found. Skipping DAE training.")
        return {}

    results = defaultdict(list)
    for latent_dim in latent_dims:
        dim1, dim2 = DIM_MAPPING.get(latent_dim, (250, 100))
        for cancer_type in cancer_types:
            input_base = os.path.join(input_root, cancer_type)
            output_base = os.path.join(output_root, "DAE_embedding", cancer_type)
            os.makedirs(output_base, exist_ok=True)

            for run in range(runs):
                print(f"\n[DAE] {cancer_type} | run={run} | latent={latent_dim}")
                try:
                    UNIFIED_DEEP.run_model(
                        model_type="DAE",
                        cancer_type=cancer_type,
                        dim1=dim1,
                        dim2=dim2,
                        latent_dim=latent_dim,
                        run=run,
                        input_base=input_base + os.sep,
                        output_base=output_base,
                        result_collector=results,
                        pca_method="PCA",
                    )
                except Exception as e:
                    print(f"[DAE] {cancer_type} run {run} failed: {e}")
                    continue

    return results


def run_vae(input_root: str, output_root: str, cancer_types: List[str], latent_dims: List[int], runs: int) -> Dict[str, list]:
    if UNIFIED_DEEP is None:
        print("[VAE] unified_deep_models not found. Skipping VAE training.")
        return {}

    results = defaultdict(list)
    for latent_dim in latent_dims:
        dim1, dim2 = DIM_MAPPING.get(latent_dim, (250, 100))
        for cancer_type in cancer_types:
            input_base = os.path.join(input_root, cancer_type)
            output_base = os.path.join(output_root, "VAE_embedding", cancer_type)
            os.makedirs(output_base, exist_ok=True)

            for run in range(runs):
                print(f"\n[VAE] {cancer_type} | run={run} | latent={latent_dim}")
                try:
                    UNIFIED_DEEP.run_model(
                        model_type="VAE",
                        cancer_type=cancer_type,
                        dim1=dim1,
                        dim2=dim2,
                        latent_dim=latent_dim,
                        run=run,
                        input_base=input_base + os.sep,
                        output_base=output_base,
                        result_collector=results,
                        pca_method="PCA",
                        gpu_number="1",
                    )
                except Exception as e:
                    print(f"[VAE] {cancer_type} run {run} failed: {e}")
                    continue

    return results


# -------------------------------
# Utilities
# -------------------------------

def calculate_statistics(results: Dict[str, list]) -> Dict[str, dict]:
    import pandas as pd

    stats = {}
    for cancer_type, runs in results.items():
        if len(runs) == 0:
            continue
        df = pd.DataFrame(runs)
        def safe_mean(col: str):
            return float(df[col].mean()) if col in df.columns else float("nan")
        def safe_std(col: str):
            return float(df[col].std()) if col in df.columns else float("nan")
        stats[cancer_type] = {
            "mean_mse": safe_mean("eval_mse") if "eval_mse" in df.columns else safe_mean("test_mse"),
            "std_mse": safe_std("eval_mse") if "eval_mse" in df.columns else safe_std("test_mse"),
            "mean_r2": safe_mean("eval_r2") if "eval_r2" in df.columns else safe_mean("test_r2"),
            "std_r2": safe_std("eval_r2") if "eval_r2" in df.columns else safe_std("test_r2"),
            "mean_train_loss": safe_mean("train_loss"),
            "std_train_loss": safe_std("train_loss"),
            "mean_val_loss": safe_mean("val_loss"),
            "std_val_loss": safe_std("val_loss"),
        }
    return stats


def save_results(results_dir: str, model_name: str, latent_dims: List[int], results: Dict[str, list]) -> None:
    os.makedirs(results_dir, exist_ok=True)
    stats = calculate_statistics(results)
    payload = {"raw_results": results, "statistics": stats}
    out_path = os.path.join(results_dir, f"training_results_{model_name}_{'-'.join(map(str, latent_dims))}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {model_name} results to: {out_path}")


# -------------------------------
# Main
# -------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AE/DAE/VAE and ICA/RP baselines end-to-end.")
    parser.add_argument("--models", type=str, default="VAE,PCA", help="Comma-separated list: PCA,AE,DAE,VAE,ICA,RP")
    parser.add_argument("--cancer-types", type=str, default="A549", help="Comma-separated cancer types")
    parser.add_argument("--latent-dims", type=str, default="5,10,25,50", help="Comma-separated latent dims for AE/DAE/VAE")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per setting for deep models")
    parser.add_argument("--input-root", type=str,  help="Root folder containing per-cancer PCA inputs",default="/home/fpan/GO_ON/deep-profile/Code/code")
    parser.add_argument("--perturbation-root", type=str, required=False, default="/home/fpan/GO_ON/deep-profile/Code/Compound_alert_data/perturbation_profiles", help="Root folder containing per-cancer perturbation_matrix.csv for ICA/RP")
    parser.add_argument("--output-root", type=str,  help="Root folder to write outputs",default='/home/fpan/GO_ON/deep-profile/Code/code/results')
    parser.add_argument("--ica-components", type=int, default=150, help="Components for ICA baseline")
    parser.add_argument("--rp-components", type=int, default=150, help="Components for RP baseline")
    parser.add_argument("--pca-components", type=int, default=150, help="Components for PCA preprocessing (0 = heuristic)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_models = {m.strip().upper() for m in args.models.split(",") if m.strip()}
    cancer_types = [c.strip() for c in args.cancer_types.split(",") if c.strip()]
    latent_dims = [int(x) for x in args.latent_dims.split(",") if x.strip()]

    # Ensure input directories exist
    for c in cancer_types:
        p = os.path.join(args.input_root, c)
        if not os.path.isdir(p):
            print(f"[WARN] Input folder missing for {c}: {p}")

    # Optional: PCA preprocessing (before deep models)
    if "PCA" in selected_models:
        if not args.perturbation_root:
            print("[WARN] --perturbation-root not provided; skipping PCA preprocessing.")
        else:
            for c in cancer_types:
                print(f"\n[PCA] {c}")
                try:
                    nc = run_pca_precompute(args.perturbation_root, args.input_root, c, pca_components=args.pca_components)
                    print(f"[PCA] {c} wrote PCA {nc}L to {os.path.join(args.input_root, c)}")
                except Exception as e:
                    print(f"[PCA] {c} failed: {e}")

    # Run deep models
    if "AE" in selected_models:
        ae_results = run_ae(args.input_root, args.output_root, cancer_types, latent_dims, args.runs)
        if ae_results:
            save_results(os.path.join(args.output_root, "AE_embedding", "results"), "AE", latent_dims, ae_results)

    if "DAE" in selected_models:
        dae_results = run_dae(args.input_root, args.output_root, cancer_types, latent_dims, args.runs)
        if dae_results:
            save_results(os.path.join(args.output_root, "DAE_embedding", "results"), "DAE", latent_dims, dae_results)

    if "VAE" in selected_models:
        vae_results = run_vae(args.input_root, args.output_root, cancer_types, latent_dims, args.runs)
        if vae_results:
            save_results(os.path.join(args.output_root, "VAE_embedding", "results"), "VAE", latent_dims, vae_results)

    # Run baselines
    if ("ICA" in selected_models) or ("RP" in selected_models):
        if not args.perturbation_root:
            print("[WARN] --perturbation-root not provided; skipping ICA/RP baselines.")
        else:
            for c in cancer_types:
                if "ICA" in selected_models:
                    print(f"\n[ICA] {c} | components={args.ica_components}")
                    try:
                        run_ica_baseline(args.perturbation_root, args.output_root, c, n_components=args.ica_components)
                    except Exception as e:
                        print(f"[ICA] {c} failed: {e}")

                if "RP" in selected_models:
                    print(f"\n[RP] {c} | components={args.rp_components}")
                    try:
                        run_rp_baseline(args.perturbation_root, args.output_root, c, n_components=args.rp_components)
                    except Exception as e:
                        print(f"[RP] {c} failed: {e}")

    print("\nAll selected tasks completed.")


if __name__ == "__main__":
    main()


