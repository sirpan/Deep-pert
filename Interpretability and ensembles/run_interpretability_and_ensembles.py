"""
Standalone pipeline to compute interpretability (Integrated Gradients),
select best cluster count using GMeans (from notebook logic),
and generate ensemble labels, ensemble gene weights, and training embeddings.

This does NOT trigger training; it only consumes existing trained encoders
and latent embeddings stored by the training pipeline.

Example (Windows PowerShell):
  python code/run_interpretability_and_ensembles.py ^
    --models VAE,AE,DAE ^
    --cancer-types A549 ^
    --latent-dims 5,10,25,50 ^
    --runs 3 ^
    --input-root D:\\data\\PCA_inputs ^
    --output-root D:\\outputs ^
    --pca-method PCA
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5" 
import argparse
import json
import math
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.stats import anderson


# Keras compatibility imports (keras or tensorflow.keras)
try:
    from keras.models import model_from_json  # type: ignore
    from keras import optimizers, metrics  # type: ignore
except Exception:
    from tensorflow.keras.models import model_from_json  # type: ignore
    from tensorflow.keras import optimizers, metrics  # type: ignore

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale

from sklearn import datasets

from scipy.stats import anderson

from pdb import set_trace


class GMeans(object):
	
	"""strictness = how strict should the anderson-darling test for normality be
			0: not at all strict
			4: very strict
	"""

	def __init__(self, min_obs=1, max_depth=10, random_state=None, strictness=4):

		super(GMeans, self).__init__()

		self.max_depth = max_depth
		
		self.min_obs = min_obs

		self.random_state = random_state

		if strictness not in range(5):
			raise ValueError("strictness parameter must be integer from 0 to 4")
		self.strictness = strictness

		self.stopping_criteria = []
		
	def _gaussianCheck(self, vector):
		"""
		check whether a given input vector follows a gaussian distribution
		H0: vector is distributed gaussian
		H1: vector is not distributed gaussian
		"""
		output = anderson(vector)

		if output[0] <= output[1][self.strictness]:
			return True
		else:
			return False
		
	
	def _recursiveClustering(self, data, depth, index):
		"""
		recursively run kmeans with k=2 on your data until a max_depth is reached or we have
			gaussian clusters
		"""
		depth += 1
		if depth == self.max_depth:
			self.data_index[index[:, 0]] = index
			self.stopping_criteria.append('max_depth')
			return
			
		km = MiniBatchKMeans(n_clusters=2, random_state=self.random_state)
		km.fit(data)
		
		centers = km.cluster_centers_
		v = centers[0] - centers[1]
		x_prime = scale(data.dot(v) / (v.dot(v)))
		gaussian = self._gaussianCheck(x_prime)
		
		# print gaussian

		if gaussian == True:
			self.data_index[index[:, 0]] = index
			self.stopping_criteria.append('gaussian')
			return

		labels = set(km.labels_)
		for k in labels:
			current_data = data[km.labels_ == k]

			if current_data.shape[0] <= self.min_obs:
				self.data_index[index[:, 0]] = index
				self.stopping_criteria.append('min_obs')
				return
			

			current_index = index[km.labels_==k]
			current_index[:, 1] = np.random.randint(0,100000000000)
			self._recursiveClustering(data=current_data, depth=depth, index=current_index)

		# set_trace()
	

	def fit(self, data):
		"""
		fit the recursive clustering model to the data
		"""
		self.data = data
		
		data_index = np.array([(i, False) for i in range(data.shape[0])])
		self.data_index = data_index

		self._recursiveClustering(data=data, depth=0, index=data_index)

		self.labels_ = self.data_index[:, 1]
		
		

def _load_encoder(model_base: str, latent_dim: int, run: int):
    encoder_json_path = os.path.join(model_base, f'encoder_{latent_dim}L_run{run}.json')
    encoder_weights_path = os.path.join(model_base, f'encoder_{latent_dim}L_run{run}.h5')
    if not (os.path.exists(encoder_json_path) and os.path.exists(encoder_weights_path)):
        raise FileNotFoundError(f"Missing encoder json/h5 at {model_base}")
    with open(encoder_json_path, 'r', encoding='utf-8') as jf:
        loaded_model_json = jf.read()
    encoder = model_from_json(loaded_model_json)
    encoder.load_weights(encoder_weights_path)
    adam = optimizers.Adam(lr=5e-4)
    def reconstruction_loss(x_input, x_decoded):
        return metrics.mse(x_input, x_decoded)
    encoder.compile(optimizer=adam, loss=reconstruction_loss)
    return encoder


def compute_integrated_gradients_for_model(model_name: str,
                                           input_root: str,
                                           output_root: str,
                                           cancer_types: List[str],
                                           latent_dims: List[int],
                                           runs: int,
                                           pca_method: str = "PCA") -> None:
    from IntegratedGradients import integrated_gradients
    import glob as _glob

    assert model_name in ("VAE", "AE", "DAE")

    for cancer_type in cancer_types:
        # Resolve PCA/ICA/RP inputs used by encoders
        input_folder = os.path.join(input_root, cancer_type)
        pattern_data = os.path.join(input_folder, f"{cancer_type}_DATA_TOP2_JOINED_{pca_method}_*L.tsv")
        pattern_comp = os.path.join(input_folder, f"{cancer_type}_DATA_TOP2_JOINED_{pca_method}_*L_COMPONENTS.tsv")
        data_files = _glob.glob(pattern_data)
        comp_files = _glob.glob(pattern_comp)
        if not data_files or not comp_files:
            print(f"[IG-{model_name}][{cancer_type}] Input files not found; skip.")
            continue
        data_path = data_files[0]
        comp_path = comp_files[0]
        input_df = pd.read_csv(data_path, sep='\t', index_col=0)
        pca_df = pd.read_csv(comp_path, sep='\t', index_col=0)
        pca_components = pca_df.values

        base_dir = os.path.join(output_root, f"{model_name}_embedding", cancer_type)
        model_base = os.path.join(base_dir, cancer_type)
        out_weights_dir = os.path.join(base_dir, "VAE_WEIGHTS")
        os.makedirs(out_weights_dir, exist_ok=True)

        for latent_dim in latent_dims:
            for run in range(runs):
                try:
                    encoder = _load_encoder(model_base, latent_dim, run)
                except Exception as e:
                    print(f"[IG-{model_name}][{cancer_type}] load encoder failed dim={latent_dim} run={run}: {e}")
                    continue

                ig = integrated_gradients(encoder)
                num_samples = input_df.shape[0]
                num_features = input_df.shape[1]
                overall_weights = np.zeros((pca_components.shape[0], latent_dim))

                for latent_node in range(latent_dim):
                    node_gradients_per_sample = np.zeros((num_samples, num_features))
                    for i in range(num_samples):
                        grads = ig.explain(input_df.values[i, :], outc=latent_node)
                        node_gradients_per_sample[i, :] = grads
                    gene_space_gradients = np.matmul(node_gradients_per_sample, pca_components.T)
                    overall_weights[:, latent_node] = np.mean(np.abs(gene_space_gradients), axis=0)

                if model_name == "VAE":
                    fname = f"{cancer_type}_DATA_VAE_Cluster_Weights_TRAINING_{latent_dim}L_fold{run}.tsv"
                else:
                    fname = f"{cancer_type}_DATA_{model_name}_Weights_TRAINING_{latent_dim}L_fold{run}.tsv"
                ig_df = pd.DataFrame(overall_weights, index=pca_df.index, columns=[f'node_{i}' for i in range(latent_dim)])
                out_path = os.path.join(out_weights_dir, fname)
                ig_df.to_csv(out_path, sep='\t')
                print(f"[IG-{model_name}][{cancer_type}] saved: {out_path}")


def select_best_k_with_gmeans(output_root: str,
                              cancer_types: List[str],
                              latent_dims: List[int],
                              runs: int) -> Dict[str, int]:
    # Follow Select_Latent_Dimension_with_Gmeans.ipynb logic

    best_k_by_cancer: Dict[str, int] = {}

    for cancer_type in cancer_types:
        data_list = []
        for dim in latent_dims:
            for i in range(runs):
                path = os.path.join(output_root, "VAE_embedding", cancer_type, cancer_type, f"latent_{dim}L_run{i}.tsv")
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path, sep='\t', index_col=0)
                data_list.append(df.values)
        if not data_list:
            print(f"[GMEANS][{cancer_type}] No latent embeddings found; skip.")
            continue
        joined_df = np.concatenate(data_list, axis=1)
        X = joined_df.T  # samples = concatenated latent variables

        
        gmeans = GMeans(strictness=3, random_state = 423)
        gmeans.fit(X)
        labels=gmeans.labels_
        #! 映射为 0..K-1 的连续整数
        labels = pd.factorize(labels)[0].astype(np.int64)
        
        selected_L = int(len(np.unique(labels)))
        best_k_by_cancer[cancer_type] = selected_L
        print(f"[GMEANS][{cancer_type}] Selected dimension {selected_L}")

        # Save labels
        labels_dir = os.path.join(output_root, "VAE_embedding", cancer_type, "VAE_kmeans_labels")
        os.makedirs(labels_dir, exist_ok=True)
        labels_path = os.path.join(labels_dir, f"{cancer_type}_TRAINING_DATA_kmeans_ENSEMBLE_LABELS_{selected_L}L.txt")
        np.savetxt(labels_path, labels, fmt="%d", delimiter='\t')
        print(f"[GMEANS][{cancer_type}] saved labels: {labels_path}")
        print("[DEBUG] labels dtype:", labels.dtype)
        print("[DEBUG] labels[:20] =", labels[:20])
        print("[DEBUG] unique labels:", np.unique(labels))

    return best_k_by_cancer
# def select_best_k_with_gmeans_debug(output_root: str,
#                               cancer_types: List[str],
#                               latent_dims: List[int],
#                               runs: int) -> Dict[str, int]:
    

#     best_k_by_cancer: Dict[str, int] = {}

#     for cancer_type in cancer_types:
#         print(f"\n[DEBUG][{cancer_type}] Start GMEANS selection")
#         data_list = []
#         for dim in latent_dims:
#             for i in range(runs):
#                 path = os.path.join(output_root, "VAE_embedding", cancer_type, cancer_type,
#                                      f"latent_{dim}L_run{i}.tsv")
#                 if not os.path.exists(path):
#                     print(f"[DEBUG][{cancer_type}] Missing latent file: {path}")
#                     continue
#                 df = pd.read_csv(path, sep='\t', index_col=0)
#                 print(f"[DEBUG][{cancer_type}] Loaded latent {path}, shape={df.shape}")
#                 data_list.append(df.values)

#         if not data_list:
#             print(f"[GMEANS][{cancer_type}] No latent embeddings found; skip.")
#             continue

#         # 拼接潜在向量
#         joined_df = np.concatenate(data_list, axis=1)
#         print(f"[DEBUG][{cancer_type}] joined_df shape = {joined_df.shape}")

#         # 转置之后用于 GMeans
#         X = joined_df.T
#         print(f"[DEBUG][{cancer_type}] X shape (samples x features) = {X.shape}")

#         # 拟合 GMeans
#         gmeans = GMeans(strictness=3, random_state=423)
#         gmeans.fit(X)
#         labels = gmeans.labels_

#         print(f"[DEBUG][{cancer_type}] Raw labels dtype: {labels.dtype}")
#         print(f"[DEBUG][{cancer_type}] Raw labels (first 20): {labels[:20]}")
#         print(f"[DEBUG][{cancer_type}] Unique labels: {np.unique(labels)}")

#         selected_L = int(len(np.unique(labels)))
#         best_k_by_cancer[cancer_type] = selected_L
#         print(f"[GMEANS][{cancer_type}] Selected dimension {selected_L}")

#         # 保存标签（强制转为 0~L-1 整数）
#         labels_int = pd.factorize(labels)[0]
#         print(f"[DEBUG][{cancer_type}] Factorized labels unique: {np.unique(labels_int)}")

#         labels_dir = os.path.join(output_root, "VAE_embedding", cancer_type, "VAE_kmeans_labels")
#         os.makedirs(labels_dir, exist_ok=True)
#         labels_path = os.path.join(labels_dir,
#                                    f"{cancer_type}_TRAINING_DATA_kmeans_ENSEMBLE_LABELS_{selected_L}L.txt")
#         np.savetxt(labels_path, labels_int, fmt="%d", delimiter='\t')
#         print(f"[GMEANS][{cancer_type}] saved labels: {labels_path}")

#     return best_k_by_cancer


# def create_ensemble_weights_and_training_embeddings(output_root: str,
#                                                     cancer_types: List[str],
#                                                     latent_dims: List[int],
#                                                     runs: int,
#                                                     best_k_by_cancer: Dict[str, int]) -> None:
#     for cancer_type in cancer_types:
#         if cancer_type not in best_k_by_cancer:
#             print(f"[ENSEMBLE][{cancer_type}] No best_k found; skip.")
#             continue
#         L = int(best_k_by_cancer[cancer_type])
#         base_dir = os.path.join(output_root, "VAE_embedding", cancer_type)
#         weights_dir = os.path.join(base_dir, "VAE_WEIGHTS")
#         labels_path = os.path.join(base_dir, "VAE_kmeans_labels", f"{cancer_type}_TRAINING_DATA_kmeans_ENSEMBLE_LABELS_{L}L.txt")
#         if not os.path.exists(labels_path):
#             print(f"[ENSEMBLE][{cancer_type}] labels missing: {labels_path}")
#             continue
#         labels = np.loadtxt(labels_path, dtype=int, delimiter='\t')

#         # Build joined weights matrix (stack runs and dims)
#         weight_blocks = []
#         gene_index = None
#         for dim in latent_dims:
#             for i in range(runs):
#                 path = os.path.join(weights_dir, f"{cancer_type}_DATA_VAE_Cluster_Weights_TRAINING_{dim}L_fold{i}.tsv")
#                 if not os.path.exists(path):
#                     continue
#                 df = pd.read_csv(path, sep='\t', index_col=0)
#                 if gene_index is None:
#                     gene_index = df.index
#                 weight_blocks.append(df.values)
#         if not weight_blocks or gene_index is None:
#             print(f"[ENSEMBLE][{cancer_type}] No IG weights found; skip.")
#             continue
        
#         # Debug: print shapes and dimensions
#         print(f"[ENSEMBLE][{cancer_type}] Debug: gene_index length: {len(gene_index)}")
#         print(f"[ENSEMBLE][{cancer_type}] Debug: weight_blocks shapes: {[w.shape for w in weight_blocks]}")
        
#         # 修复权重组合逻辑：每个权重矩阵应该是 (genes x latent_dim)
#         # 我们需要将latent维度连接起来，而不是直接concatenate
#         max_latent_dim = max(w.shape[1] for w in weight_blocks)
#         total_latent_dim = sum(w.shape[1] for w in weight_blocks)
        
#         # 创建组合权重矩阵：(total_latent_dim, genes)
#         joined_weights = np.zeros((total_latent_dim, len(gene_index)))
#         current_pos = 0
#         for weight_block in weight_blocks:
#             # weight_block shape: (genes, latent_dim)
#             # 转置为 (latent_dim, genes) 然后添加到joined_weights
#             joined_weights[current_pos:current_pos + weight_block.shape[1], :] = weight_block.T
#             current_pos += weight_block.shape[1]
        
#         print(f"[ENSEMBLE][{cancer_type}] Debug: joined_weights shape: {joined_weights.shape}")
#         print(f"[ENSEMBLE][{cancer_type}] Debug: labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
        
#         # 现在joined_weights的形状应该是 (total_latent_dim, genes)
#         # 转置为 (genes, total_latent_dim) 以便后续处理
#         joined_weights = joined_weights.T  # 现在是 (genes, total_latent_dim)

#         # Ensemble weights (L x genes)
#         ensemble_weights = np.zeros((L, len(gene_index)))
#         for label in range(L):
#             indices = np.where(labels == label)[0]
#             if len(indices) == 0:
#                 continue
#             # 对每个基因，计算该cluster中所有latent维度的平均权重
#             ensemble_weights[label, :] = np.mean(joined_weights[:, indices], axis=1)

#         out_dir = os.path.join(base_dir, 'Ensemble_Gene_Importance_Weights')
#         os.makedirs(out_dir, exist_ok=True)
#         ensemble_df = pd.DataFrame(ensemble_weights, index=np.arange(L), columns=gene_index)
#         ensemble_path = os.path.join(out_dir, f"{cancer_type}_DeepProfile_Ensemble_Gene_Importance_Weights_{L}L.tsv")
#         ensemble_df.to_csv(ensemble_path, sep='\t')
#         print(f"[ENSEMBLE][{cancer_type}] saved ensemble weights: {ensemble_path}")

#         # Create training embeddings (samples x L)
#         # Join latent embeddings again
#         data_list = []
#         sample_index = None
#         for dim in latent_dims:
#             for i in range(runs):
#                 path = os.path.join(base_dir, cancer_type, f"latent_{dim}L_run{i}.tsv")
#                 if not os.path.exists(path):
#                     continue
#                 df = pd.read_csv(path, sep='\t', index_col=0)
#                 if sample_index is None:
#                     sample_index = df.index
#                 data_list.append(df.values)
#         if not data_list or sample_index is None:
#             print(f"[ENSEMBLE][{cancer_type}] No latent embeddings for training embedding; skip.")
#             continue
#         joined_data = np.concatenate(data_list, axis=1)
#         ensemble_embeddings = np.zeros((joined_data.shape[0], L))
#         for label in range(L):
#             indices = np.where(labels == label)[0]
#             if len(indices) == 0:
#                 continue
#             ensemble_embeddings[:, label] = np.mean(joined_data[:, indices], axis=1)

#         emb_df = pd.DataFrame(ensemble_embeddings, index=sample_index, columns=np.arange(L))
#         emb_path = os.path.join(base_dir, f"{cancer_type}_DeepProfile_Training_Embedding_{L}L.tsv")
#         emb_df.to_csv(emb_path, sep='\t')
#         print(f"[ENSEMBLE][{cancer_type}] saved training embedding: {cancer_type}_DeepProfile_Training_Embedding_{L}L.tsv")
def create_ensemble_weights_and_training_embeddings(
    output_root: str,
    cancer_types: List[str],
    latent_dims: List[int],
    runs: int,
    best_k_by_cancer: Dict[str, int]
) -> None:
    """
    调试版：
    1) 对每一步做严格的维度/一致性/内容检查并打印统计
    2) 明确对齐关系：labels 的长度 = 拼接后的 total_latent_dim
    3) 检测全零列/NaN，并在均值前做防御性处理（跳过空索引/全NaN）
    4) 不重算 IG，仅读取已有 VAE_WEIGHTS 与 latent_*.tsv
    """
    def _summary(name, arr):
        arr = np.asarray(arr)
        return {
            "shape": tuple(arr.shape),
            "dtype": str(arr.dtype),
            "nan": int(np.isnan(arr).sum()),
            "inf": int(np.isinf(arr).sum()),
            "min": float(np.nanmin(arr)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
            "mean": float(np.nanmean(arr)) if arr.size else None,
            "nnz": int(np.count_nonzero(arr)),
        }

    for cancer_type in cancer_types:
        if cancer_type not in best_k_by_cancer:
            print(f"[ENSEMBLE][{cancer_type}] No best_k found; skip.")
            continue
        L = int(best_k_by_cancer[cancer_type])
        base_dir = os.path.join(output_root, "VAE_embedding", cancer_type)
        weights_dir = os.path.join(base_dir, "VAE_WEIGHTS")
        labels_path = os.path.join(
            base_dir, "VAE_kmeans_labels",
            f"{cancer_type}_TRAINING_DATA_kmeans_ENSEMBLE_LABELS_{L}L.txt"
        )

        print(f"\n[ENSEMBLE][{cancer_type}] ===== DEBUG START =====")
        print(f"[ENSEMBLE][{cancer_type}] L (clusters) = {L}")
        print(f"[ENSEMBLE][{cancer_type}] weights_dir = {weights_dir}")
        print(f"[ENSEMBLE][{cancer_type}] labels_path = {labels_path}")

        if not os.path.exists(labels_path):
            print(f"[ENSEMBLE][{cancer_type}] labels missing: {labels_path}")
            print(f"[ENSEMBLE][{cancer_type}] ===== DEBUG END (labels missing) =====")
            continue

        # --- 读取聚类标签 ---
        try:
            labels = np.loadtxt(labels_path, dtype=int, delimiter='\t')
        except Exception as e:
            print(f"[ENSEMBLE][{cancer_type}] ERROR reading labels: {e}")
            print(f"[ENSEMBLE][{cancer_type}] ===== DEBUG END (labels load error) =====")
            continue
        if labels.ndim != 1:
            labels = labels.ravel()
        print(f"[ENSEMBLE][{cancer_type}] labels summary: {_summary('labels', labels)}")
        print(f"[ENSEMBLE][{cancer_type}] labels unique: {np.unique(labels)}")

        # --- 读取并拼接 IG 权重 ---
        weight_blocks = []
        gene_index = None
        missing_weight_files = []
        for dim in latent_dims:
            for i in range(runs):
                path = os.path.join(
                    weights_dir,
                    f"{cancer_type}_DATA_VAE_Cluster_Weights_TRAINING_{dim}L_fold{i}.tsv"
                )
                if not os.path.exists(path):
                    missing_weight_files.append(path)
                    continue
                df = pd.read_csv(path, sep='\t', index_col=0)
                if gene_index is None:
                    gene_index = df.index
                else:
                    # 基因索引需要完全一致
                    if not df.index.equals(gene_index):
                        print(f"[ENSEMBLE][{cancer_type}] ERROR: gene index mismatch in {path}")
                        print(f"[ENSEMBLE][{cancer_type}] ===== DEBUG END (gene index mismatch) =====")
                        return
                weight_blocks.append(df.values)
                print(f"[ENSEMBLE][{cancer_type}] loaded weights {path} -> shape {df.values.shape}")

        if missing_weight_files:
            print(f"[ENSEMBLE][{cancer_type}] WARNING missing IG files ({len(missing_weight_files)}):")
            for p in missing_weight_files[:5]:
                print(f"  - {p}")
            if len(missing_weight_files) > 5:
                print("  ...")

        if not weight_blocks or gene_index is None:
            print(f"[ENSEMBLE][{cancer_type}] No IG weights found; skip.")
            print(f"[ENSEMBLE][{cancer_type}] ===== DEBUG END (no IG weights) =====")
            continue

        # 打印每个块的统计
        for idx, w in enumerate(weight_blocks):
            print(f"[ENSEMBLE][{cancer_type}] block[{idx}] stats: {_summary(f'wb{idx}', w)}")

        # 计算总 latent 列数
        total_latent_dim = sum(w.shape[1] for w in weight_blocks)
        genes_n = len(gene_index)
        print(f"[ENSEMBLE][{cancer_type}] gene_index length = {genes_n}")
        print(f"[ENSEMBLE][{cancer_type}] total_latent_dim (from IG files) = {total_latent_dim}")

        # 将 (genes x latent_dim) 的块转成 (latent_dim x genes) 后按行堆叠
        joined_weights_LT_G = np.zeros((total_latent_dim, genes_n), dtype=float)
        cur = 0
        for w in weight_blocks:
            ld = w.shape[1]
            joined_weights_LT_G[cur:cur+ld, :] = w.T  # (ld x genes)
            cur += ld

        print(f"[ENSEMBLE][{cancer_type}] joined_weights (latent x genes) stats: "
              f"{_summary('joined_weights_LT_G', joined_weights_LT_G)}")

        # 关键一致性检查：labels 数量必须等于 total_latent_dim
        if labels.size != total_latent_dim:
            print(f"[ENSEMBLE][{cancer_type}] ERROR: labels length ({labels.size}) "
                  f"!= total_latent_dim ({total_latent_dim}).")
            print("[ENSEMBLE] 这会导致按簇聚合时维度错位，从而平均为 0。")
            print(f"[ENSEMBLE][{cancer_type}] ===== DEBUG END (labels/latent mismatch) =====")
            continue

        # 检查是否存在全零的 latent 列（可能来自全 0 IG 或读取错误）
        zero_latent_cols = np.where(~np.any(joined_weights_LT_G, axis=1))[0]
        if zero_latent_cols.size:
            print(f"[ENSEMBLE][{cancer_type}] WARNING: {zero_latent_cols.size} latent columns are all-zero in IG.")
            print(f"  例: {zero_latent_cols[:10]} (最多展示 10 个)")

        # 转成 (genes x latent) 方便按列索引
        joined_weights_G_L = joined_weights_LT_G.T  # (genes x total_latent_dim)

        # --- 计算 Ensemble gene importance (L x genes) ---
        ensemble_weights = np.zeros((L, genes_n), dtype=float)
        for label in range(L):
            idxs = np.where(labels == label)[0]
            print(f"[ENSEMBLE][{cancer_type}] cluster {label}: size={idxs.size}")
            if idxs.size == 0:
                continue
            # 取该簇所有 latent 维度，对每个基因求平均权重
            sub = joined_weights_G_L[:, idxs]  # (genes x |idxs|)
            # 统计
            print(f"[ENSEMBLE][{cancer_type}]  cluster {label} sub stats: {_summary('sub', sub)}")
            # 若出现全 NaN（极少见），跳过
            col_mean = np.nanmean(sub, axis=1)  # (genes,)
            col_mean[np.isnan(col_mean)] = 0.0
            ensemble_weights[label, :] = col_mean

        out_dir = os.path.join(base_dir, 'Ensemble_Gene_Importance_Weights')
        os.makedirs(out_dir, exist_ok=True)
        ensemble_df = pd.DataFrame(ensemble_weights, index=np.arange(L), columns=gene_index)
        ensemble_path = os.path.join(out_dir, f"{cancer_type}_DeepProfile_Ensemble_Gene_Importance_Weights_{L}L.tsv")
        ensemble_df.to_csv(ensemble_path, sep='\t')
        print(f"[ENSEMBLE][{cancer_type}] saved ensemble weights: {ensemble_path}")
        print(f"[ENSEMBLE][{cancer_type}] ensemble_weights stats: {_summary('ensemble_weights', ensemble_weights)}")

        # --- 训练样本的 Ensemble Embeddings (samples x L) ---
        data_list = []
        sample_index = None
        missing_latent_files = []
        for dim in latent_dims:
            for i in range(runs):
                path = os.path.join(base_dir, cancer_type, f"latent_{dim}L_run{i}.tsv")
                if not os.path.exists(path):
                    missing_latent_files.append(path)
                    continue
                df = pd.read_csv(path, sep='\t', index_col=0)
                if sample_index is None:
                    sample_index = df.index
                else:
                    if not df.index.equals(sample_index):
                        print(f"[ENSEMBLE][{cancer_type}] ERROR: sample index mismatch in {path}")
                        print(f"[ENSEMBLE][{cancer_type}] ===== DEBUG END (sample index mismatch) =====")
                        return
                data_list.append(df.values)
                print(f"[ENSEMBLE][{cancer_type}] loaded latent {path} -> shape {df.values.shape}")

        if missing_latent_files:
            print(f"[ENSEMBLE][{cancer_type}] WARNING missing latent files ({len(missing_latent_files)}):")
            for p in missing_latent_files[:5]:
                print(f"  - {p}")
            if len(missing_latent_files) > 5:
                print("  ...")

        if not data_list or sample_index is None:
            print(f"[ENSEMBLE][{cancer_type}] No latent embeddings for training embedding; skip.")
            print(f"[ENSEMBLE][{cancer_type}] ===== DEBUG END (no latent matrices) =====")
            continue

        joined_data = np.concatenate(data_list, axis=1)  # (samples x total_latent_dim)
        print(f"[ENSEMBLE][{cancer_type}] joined_data stats: {_summary('joined_data', joined_data)}")

        # 再次一致性检查
        if joined_data.shape[1] != labels.size:
            print(f"[ENSEMBLE][{cancer_type}] ERROR: joined_data latent cols ({joined_data.shape[1]}) "
                  f"!= labels length ({labels.size}).")
            print("[ENSEMBLE] 拼接顺序与 GMeans 时使用的顺序不一致会导致这个问题。")
            print(f"[ENSEMBLE][{cancer_type}] ===== DEBUG END (data/labels mismatch) =====")
            continue

        ensemble_embeddings = np.zeros((joined_data.shape[0], L), dtype=float)
        for label in range(L):
            idxs = np.where(labels == label)[0]
            if idxs.size == 0:
                continue
            sub = joined_data[:, idxs]  # (samples x |idxs|)
            print(f"[ENSEMBLE][{cancer_type}]  train-embed cluster {label} sub stats: {_summary('sub_train', sub)}")
            col_mean = np.nanmean(sub, axis=1)  # (samples,)
            col_mean[np.isnan(col_mean)] = 0.0
            ensemble_embeddings[:, label] = col_mean

        emb_df = pd.DataFrame(ensemble_embeddings, index=sample_index, columns=np.arange(L))
        emb_path = os.path.join(base_dir, f"{cancer_type}_DeepProfile_Training_Embedding_{L}L.tsv")
        emb_df.to_csv(emb_path, sep='\t')
        print(f"[ENSEMBLE][{cancer_type}] saved training embedding: {emb_path}")
        print(f"[ENSEMBLE][{cancer_type}] ensemble_embeddings stats: {_summary('ensemble_embeddings', ensemble_embeddings)}")
        print(f"[ENSEMBLE][{cancer_type}] ===== DEBUG END =====\n")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-training pipeline: IG -> GMeans -> ensemble labels/weights/embeddings")
    parser.add_argument("--models", type=str, default="VAE", help="Which models to compute IG for: VAE,AE,DAE")
    parser.add_argument("--cancer-types", type=str, default="A549", help="Comma-separated cancer types")
    parser.add_argument("--latent-dims", type=str, default="5", help="Comma-separated latent dims")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per latent dim")
    parser.add_argument("--input-root", type=str,  help="Root folder with per-cancer PCA inputs",default='/home/fpan/GO_ON/deep-profile/Code/code')
    parser.add_argument("--output-root", type=str, help="Root folder with trained model outputs",default='/home/fpan/GO_ON/deep-profile/Code/code/results')
    parser.add_argument("--pca-method", type=str, default="PCA", help="Feature method used during training: PCA|ICA|RP")
    parser.add_argument("--skip-ig", action="store_true", help="Skip IG computation and only run GMeans + ensemble steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_models = {m.strip().upper() for m in args.models.split(",") if m.strip()}
    cancer_types = [c.strip() for c in args.cancer_types.split(",") if c.strip()]
    latent_dims = [int(x) for x in args.latent_dims.split(",") if x.strip()]

    if not args.skip_ig:
        if "VAE" in selected_models:
            compute_integrated_gradients_for_model("VAE", args.input_root, args.output_root, cancer_types, latent_dims, args.runs, pca_method=args.pca_method)
        if "AE" in selected_models:
            compute_integrated_gradients_for_model("AE", args.input_root, args.output_root, cancer_types, latent_dims, args.runs, pca_method=args.pca_method)
        if "DAE" in selected_models:
            compute_integrated_gradients_for_model("DAE", args.input_root, args.output_root, cancer_types, latent_dims, args.runs, pca_method=args.pca_method)

    best_k_by_cancer = select_best_k_with_gmeans(args.output_root, cancer_types, latent_dims, args.runs)
    create_ensemble_weights_and_training_embeddings(args.output_root, cancer_types, latent_dims, args.runs, best_k_by_cancer)
    print("\nPost-training pipeline completed.")


if __name__ == "__main__":
    main()


