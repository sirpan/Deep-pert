"""
Target pair prediction analysis script
Predict potential drug target combinations based on gene importance weights from deep learning models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.interpolate import interp1d
import os


# New: parse user-provided gene list
def read_user_gene_list(file_path):
    """Read a user-provided gene list file and return a DataFrame with a 'SYMBOL' column."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Gene list file not found: {file_path}")
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == '.csv':
        df_gene = pd.read_csv(file_path)
    else:
        # Default to tab-delimited for .txt/.tsv
        df_gene = pd.read_table(file_path)
    if 'SYMBOL' in df_gene.columns:
        symbols = df_gene['SYMBOL'].dropna().astype(str).str.strip()
    else:
        # If no 'SYMBOL' column, use the first column
        first_col = df_gene.columns[0]
        symbols = df_gene[first_col].dropna().astype(str).str.strip()
    symbols = symbols[symbols != '']
    return pd.DataFrame({'SYMBOL': symbols})


def load_and_preprocess_data(weights_path, target_pair_path, target_info_path, drug_comb_path):
    """Load and preprocess all required data."""
    print("Loading data...")
    
    # Load gene importance weights
    df = pd.read_csv(weights_path, sep='\t', index_col=0)
    df = df.drop(columns='dose') if 'dose' in df.columns else df
    df = df.T  # Ensure format is (genes × clusters)
    
    # Load target pair information
    target_pair = pd.read_table(target_pair_path, sep='\t')
    target_pair_info = pd.read_table(target_info_path, sep='\t')
    combine_pair = pd.read_table(drug_comb_path, sep='\t')
    
    # Merge data
    com_tar_info = pd.merge(target_pair, combine_pair, on="dcid")
    target_combination = pd.merge(target_pair_info, com_tar_info, on='tpid')
    target_combination = target_combination.drop_duplicates(subset=['genename1', 'genename2', 'fd_name'])
    
    print(f"Data loaded: {df.shape[0]} genes, {df.shape[1]} clusters")
    return df, target_combination


def compute_gene_importance(df):
    """Compute gene importance stats."""
    clusters = df.columns.tolist()
    df['Average_Importance'] = df[clusters].mean(axis=1)
    df['Total_Importance'] = df[clusters].sum(axis=1)
    return df.sort_values('Average_Importance', ascending=False)


def normalize_and_cluster_genes(df, top_genes=50, num_clusters=4):
    """Normalize gene data and perform clustering."""
    print("Running gene clustering analysis...")
    
    # MinMax normalization
    scaler = MinMaxScaler()
    df_copy = df.copy()
    df_copy.columns = df_copy.columns.astype(str)
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_copy),
        columns=df_copy.columns,
        index=df_copy.index
    )
    
    # Select top genes for clustering
    top_genes_list = df_scaled.sort_values('Total_Importance', ascending=False).index[:top_genes].tolist()
    df_top = df_scaled.loc[top_genes_list]
    
    # Hierarchical clustering
    Z = linkage(df_top.values, method='ward')
    cluster_labels = fcluster(Z, t=num_clusters, criterion='maxclust')
    
    # Build gene clusters
    gene_clusters = {}
    for i, label in enumerate(cluster_labels):
        cluster_name = f'Cluster {label}'
        if cluster_name not in gene_clusters:
            gene_clusters[cluster_name] = []
        gene_clusters[cluster_name].append(df_top.index[i])
    
    print(f"Clustering complete: {len(gene_clusters)} clusters")
    return df_scaled, gene_clusters


def create_visualization(df_scaled, gene_clusters, output_path="gene_cluster_heatmap.svg"):
    """Create heatmap and trend plots for gene clusters."""
    print("Generating visualizations...")
    
    # Prepare data
    genes = list(df_scaled.index[:50])
    df_viz = df_scaled.loc[genes].iloc[:, :95]
    
    # Heatmap
    g = sns.clustermap(
        df_viz,
        method='ward',
        metric='euclidean',
        cmap="viridis",
        figsize=(12, 12),
        dendrogram_ratio=(.1, .2),
        cbar_pos=(0.02, 0.8, .03, .15),
        linewidths=.75,
        xticklabels=False,
        yticklabels=True,
        cbar_kws={'label': 'Normalized Score', 'orientation': 'vertical'}
    )
    
    # Layout
    g.gs.update(bottom=0.3)
    heatmap_pos = g.ax_heatmap.get_position()
    ax_trends = g.fig.add_axes([heatmap_pos.x0, 0.1, heatmap_pos.width, 0.15])
    
    # Build trend clusters based on provided gene_clusters to ensure alignment
    # Intersect with genes shown in the heatmap for consistency
    trend_clusters = {}
    for cluster_name, genes_in_cluster in gene_clusters.items():
        if not isinstance(genes_in_cluster, list):
            genes_in_cluster = list(genes_in_cluster)
        intersect_genes = [g for g in genes_in_cluster if g in df_viz.index]
        trend_clusters[cluster_name] = intersect_genes
    
    # Draw trends
    # Sort cluster names like 'Cluster 1', 'Cluster 2', ... if possible
    def _cluster_sort_key(name):
        try:
            return int(str(name).split()[-1])
        except Exception:
            return name
    ordered_trend_items = sorted(trend_clusters.items(), key=lambda kv: _cluster_sort_key(kv[0]))
    
    trend_colors = sns.color_palette("Set2", n_colors=len(ordered_trend_items))
    x_axis = np.arange(len(df_viz.columns))
    
    for i, (cluster_name, genes_in_cluster) in enumerate(ordered_trend_items):
        if len(genes_in_cluster) == 0:
            continue
        
        mean_trend = df_viz.loc[genes_in_cluster].mean(axis=0)
        
        if len(x_axis) > 3:
            f_smooth = interp1d(x_axis, mean_trend, kind='cubic')
            x_smooth = np.linspace(x_axis.min(), x_axis.max(), 300)
            ax_trends.plot(x_smooth, f_smooth(x_smooth), color=trend_colors[i], 
                          linewidth=3, label=cluster_name, alpha=0.9)
        else:
            ax_trends.plot(x_axis, mean_trend, color=trend_colors[i], 
                          linewidth=3, label=cluster_name, alpha=0.9)
        
        std_trend = df_viz.loc[genes_in_cluster].std(axis=0)
        ax_trends.fill_between(x_axis, mean_trend - std_trend, mean_trend + std_trend, 
                              color=trend_colors[i], alpha=0.15)
    
    # Beautify
    ax_trends.set_xlabel('Ordered Clusters', fontsize=20, labelpad=10)
    ax_trends.set_ylabel('Mean Score', fontsize=20)
    ax_trends.tick_params(axis='both', labelsize=20)
    ax_trends.legend(title='Gene Clusters', loc='upper right', fontsize=10, frameon=False)
    sns.despine(ax=ax_trends)
    ax_trends.grid(axis='y', linestyle=':', linewidth=0.6, color='lightgray')
    
    plt.suptitle('Gene Importance Dynamics Across Clusters', fontsize=22, fontweight='bold', y=1.02, x=0.55)
    
    # Save
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.show()
    
    return trend_clusters


def classify_candidate_genes(df_scaled, target_combination, disease_name, gene_clusters, use_user_gene_list=False, user_gene_df=None):
    """Classify candidate genes.
    If use_user_gene_list=True, use the user-provided gene list (DataFrame with 'SYMBOL' column); otherwise filter from target pair data by disease name.
    """
    if use_user_gene_list:
        print("Classifying user-provided candidate genes...")
    else:
        print(f"Classifying candidate genes for {disease_name}...")
    
    # Build candidate gene set
    if use_user_gene_list:
        if user_gene_df is None or 'SYMBOL' not in user_gene_df.columns:
            raise ValueError("user_gene_df must be provided and contain a 'SYMBOL' column")
        processed_df = user_gene_df.copy()
    else:
        disease_genes = target_combination[target_combination['fd_name'] == disease_name]
        candidate_genes = set(disease_genes['genename1']) | set(disease_genes['genename2'])
        processed_df = pd.DataFrame(list(candidate_genes), columns=['SYMBOL'])
    
    # Keep genes present in weight matrix
    df_copy = df_scaled.copy()
    df_copy.index = df_copy.index.str.lower()
    processed_df_gene_symbol = set(df_copy.index) & set(processed_df['SYMBOL'].astype(str).str.lower())
    
    df_candidates = df_copy[df_copy.index.isin(processed_df_gene_symbol)]
    
    if df_candidates.empty:
        print("Warning: No matching candidate genes found")
        return None
    
    # Compute cluster centroids (first 95 columns to match candidate slicing)
    cluster_centroids = {}
    for cluster_name, genes_in_cluster in gene_clusters.items():
        if len(genes_in_cluster) > 0:
            cluster_centroids[cluster_name] = df_scaled.loc[genes_in_cluster].iloc[:, :95].mean(axis=0)
    
    # Classification helper
    def find_best_cluster(gene_profile, centroids):
        min_distance = float('inf')
        best_cluster_name = None
        
        for cluster_name, centroid_profile in centroids.items():
            if len(gene_profile.values) != len(centroid_profile.values):
                print(f"Warning: Dimension mismatch - gene_profile: {len(gene_profile.values)}, centroid: {len(centroid_profile.values)}")
                continue
            distance = np.linalg.norm(gene_profile.values - centroid_profile.values)
            if distance < min_distance:
                min_distance = distance
                best_cluster_name = cluster_name
        
        return best_cluster_name
    
    # Classify
    df_candidates_new = df_candidates.iloc[:, :95]
    assigned_clusters = df_candidates_new.apply(
        find_best_cluster, axis=1, centroids=cluster_centroids
    )
    
    # Results
    classification_results = pd.DataFrame({
        'Gene': assigned_clusters.index,
        'Assigned_Cluster': assigned_clusters.values
    })
    
    print("Candidate gene classification complete")
    return classification_results, df_candidates


def generate_target_combinations(df_scaled, classification_results, target_cluster_name, 
                               gene_clusters, corr_threshold=0.8, top_n=100):
    """Generate target combinations."""
    print(f"Generating combinations for {target_cluster_name}...")
    
    # Genes in target cluster
    genes_in_cluster = gene_clusters[target_cluster_name]
    if not isinstance(genes_in_cluster, list):
        genes_in_cluster = list(genes_in_cluster)
    df_cluster = df_scaled.loc[genes_in_cluster]
    
    # Identify target clusters (columns with max/min sums)
    clusters = df_cluster.columns.tolist()
    col_sum = df_cluster[clusters].sum(axis=0)
    max_col = col_sum.idxmax()
    min_col = col_sum.idxmin()
    target_clusters = [max_col, min_col]
    
    print(f"Target cluster columns: {max_col} (max), {min_col} (min)")
    
    # Candidate genes from classification
    left_gene = list(classification_results[classification_results['Assigned_Cluster'] == target_cluster_name]['Gene'])
    print(f"Candidate genes from classification: {left_gene}")
    
    # Lowercase matching
    df_scaled_lower = df_scaled.copy()
    df_scaled_lower.index = df_scaled_lower.index.str.lower()
    df_candidates = df_scaled_lower.loc[left_gene]
    
    print(f"Matched candidate genes in df_scaled: {len(df_candidates)}")
    if len(df_candidates) > 0:
        print(f"Matched genes: {list(df_candidates.index)}")
    
    if len(df_candidates) < 2:
        print("Warning: Not enough candidate genes")
        return None
    
    # Correlations
    expr_matrix = df_candidates.loc[left_gene]
    gene_corr_signed = expr_matrix.T.corr()
    gene_corr_abs = gene_corr_signed.abs()
    np.fill_diagonal(gene_corr_abs.values, 0.0)
    
    # Pair filter by threshold
    rows, cols = np.where(gene_corr_abs.values >= corr_threshold)
    valid_pairs = set((min(r, c), max(r, c)) for r, c in zip(rows, cols) if r != c)
    
    if not valid_pairs:
        print("Warning: No gene pairs meet the correlation threshold")
        return None
    
    # Scores
    comb_data = []
    gene_names = list(expr_matrix.index)
    
    for i, j in valid_pairs:
        geneA, geneB = gene_names[i], gene_names[j]
        score_a = df_candidates.loc[geneA, target_clusters[0]] + df_candidates.loc[geneB, target_clusters[0]]
        score_b = df_candidates.loc[geneA, target_clusters[1]] + df_candidates.loc[geneB, target_clusters[1]]
        total_score = score_a + score_b
        
        comb_data.append({
            "GeneA": geneA,
            "GeneB": geneB,
            "ClustA_Score": score_a,
            "ClustB_Score": score_b,
            "Total_Score": total_score,
            "PairCorr_Signed": float(gene_corr_signed.iat[i, j]),
            "PairCorr_Abs": float(gene_corr_abs.iat[i, j]),
        })
    
    # Output
    df_comb = pd.DataFrame(comb_data)
    df_comb["sorted_pair"] = df_comb.apply(lambda x: tuple(sorted([x["GeneA"], x["GeneB"]])), axis=1)
    df_comb = (df_comb.sort_values("Total_Score", ascending=False)
                      .drop_duplicates(subset="sorted_pair")
                      .drop(columns="sorted_pair")
                      .head(top_n))
    
    print(f"Target combinations generated: {len(df_comb)} pairs")
    return df_comb


def evaluate_predictions(df_comb, target_combination, disease_name, output_dir):
    """Evaluate prediction overlap with known target combinations."""
    print("Evaluating predictions...")
    
    # Disease-specific known combos
    target_combination_disease = target_combination[target_combination['fd_name'] == disease_name]
    
    # Normalize headers
    df_comb = df_comb.rename(columns={c: c.strip() for c in df_comb.columns})
    target_combination_disease = target_combination_disease.rename(columns={c: c.strip() for c in target_combination_disease.columns})
    
    # Build pair keys
    df_comb["GeneA_up"] = df_comb["GeneA"].astype(str).str.strip().str.upper()
    df_comb["GeneB_up"] = df_comb["GeneB"].astype(str).str.strip().str.upper()
    df_comb["pair_key"] = df_comb.apply(lambda r: tuple(sorted([r["GeneA_up"], r["GeneB_up"]])), axis=1)
    
    # Known pairs
    if {"genename1", "genename2"}.issubset(target_combination_disease.columns):
        g1 = target_combination_disease["genename1"].astype(str).str.strip().str.upper()
        g2 = target_combination_disease["genename2"].astype(str).str.strip().str.upper()
    elif "tp_name" in target_combination_disease.columns:
        tmp = target_combination_disease["tp_name"].astype(str).str.split(";", n=1, expand=True)
        g1 = tmp[0].str.strip().str.upper()
        g2 = tmp[1].str.strip().str.upper()
    else:
        raise ValueError("Target combination table must contain genename1/genename2 or tp_name column")
    
    df2_pair_keys = set(tuple(sorted([a, b])) for a, b in zip(g1, g2))
    
    # Match rates for different TOP_N
    match_stats = []
    for top_n in range(10, 101, 10):
        df_temp = df_comb.head(top_n).copy()
        df_temp["GeneA_up"] = df_temp["GeneA"].astype(str).str.strip().str.upper()
        df_temp["GeneB_up"] = df_temp["GeneB"].astype(str).str.strip().str.upper()
        df_temp["pair_key"] = df_temp.apply(lambda r: tuple(sorted([r["GeneA_up"], r["GeneB_up"]])), axis=1)
        
        mask = df_temp["pair_key"].isin(df2_pair_keys)
        matched_count = mask.sum()
        
        match_stats.append({
            "TOP_N": top_n,
            "Total_Pairs": len(df_temp),
            "Matched_Pairs": matched_count,
            "Match_Rate": matched_count / len(df_temp) if len(df_temp) > 0 else 0
        })
    
    # Save
    match_stats_df = pd.DataFrame(match_stats)
    output_path = os.path.join(output_dir, "match_statistics.csv")
    match_stats_df.to_csv(output_path, index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(match_stats_df["TOP_N"], match_stats_df["Match_Rate"], marker='o')
    plt.xlabel("TOP_N")
    plt.ylabel("Match Rate")
    plt.title(f"{disease_name} - Gene Pair Match Rate vs TOP_N")
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "match_rate_trend.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Evaluation results saved to: {output_dir}")
    return match_stats_df


def main():
    """Main function."""
    # Configuration
    WEIGHTS_PATH = r'D:\基于深度学习模型对靶标组合的预测\target_essential\cal\OV7\OV7_DeepProfile_Ensemble_Gene_Importance_Weights_95L.tsv'
    TARGET_PAIR_PATH = r'D:\基于深度学习模型对靶标组合的预测\target_essential\target pair\TargetPair_DrugCombination_Mapping_Table.txt'
    TARGET_INFO_PATH = r'D:\基于深度学习模型对靶标组合的预测\target_essential\target pair\TargetPair_Information.txt'
    DRUG_COMB_PATH = r'D:\基于深度学习模型对靶标组合的预测\target_essential\target pair\Drug_Combination_Clinicaltrial_Phase_Information.txt'
    
    DISEASE_NAME = 'Ovarian Cancer'
    OUTPUT_DIR = r'D:\基于深度学习模型对靶标组合的预测\target_essential\combine\Ovarian Cancer'
    
    # Mode selection
    print("=" * 60)
    print("Target Combination Prediction System")
    print("=" * 60)
    print("Select run mode:")
    print("1. Prediction mode - customizable clustering parameters for exploratory analysis")
    print("2. Evaluation mode - default settings for validating known results")
    print("-" * 60)
    
    while True:
        mode_choice = input("Enter mode (1 or 2): ").strip()
        if mode_choice in ['1', '2']:
            break
        else:
            print("Invalid choice, please enter 1 or 2")
    
    # Parameterization
    if mode_choice == '1':
        # Prediction mode
        print("\n=== Prediction Mode ===")
        print("Enter custom parameters:")
        
        # Number of clusters
        while True:
            try:
                num_clusters = int(input("Number of clusters (recommended 2-8): "))
                if 2 <= num_clusters <= 10:
                    break
                else:
                    print("Number of clusters should be between 2 and 10")
            except ValueError:
                print("Please enter a valid integer")
        
        # Number of control genes (top genes used to form clusters)
        while True:
            try:
                top_genes = int(input("Number of control genes (recommended 50-200): "))
                if 20 <= top_genes <= 500:
                    break
                else:
                    print("Number of control genes should be between 20 and 500")
            except ValueError:
                print("Please enter a valid integer")
        
        # Correlation threshold
        # New: allow Auto selection to infer threshold from number of candidate genes later
        while True:
            corr_threshold_input = input("Correlation threshold (0.0-1.0, recommended 0.7-0.9). Enter 'auto' to auto-select: ").strip().lower()
            if corr_threshold_input == 'auto':
                use_auto_threshold = True
                corr_threshold = None
                break
            try:
                corr_threshold = float(corr_threshold_input)
                if 0.0 <= corr_threshold <= 1.0:
                    use_auto_threshold = False
                    break
                else:
                    print("Correlation threshold must be between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number or 'auto'")
        
        # Number of output pairs
        while True:
            try:
                top_n = int(input("Number of output combinations (recommended 50-200): "))
                if 10 <= top_n <= 500:
                    break
                else:
                    print("Number of output combinations should be between 10 and 500")
            except ValueError:
                print("Please enter a valid integer")
        
        # Candidate gene source
        print("\nSelect candidate gene source:")
        print("1. Use built-in target pair data (filtered by disease)")
        print("2. Upload custom gene list (Gene Symbol)")
        while True:
            source_choice = input("Enter choice (1 or 2): ").strip()
            if source_choice in ['1', '2']:
                break
            else:
                print("Invalid choice, please enter 1 or 2")
        use_user_gene_list = (source_choice == '2')
        user_gene_df = None
        if use_user_gene_list:
            user_path = input("Enter gene list file path (.csv/.tsv/.txt; if no 'SYMBOL' column, the first column will be used): ").strip()
            try:
                user_gene_df = read_user_gene_list(user_path)
                print(f"Loaded {len(user_gene_df)} custom genes")
            except Exception as e:
                print(f"Failed to read gene list: {e}")
                return
        else:
            print(f"Using built-in target pair data filtered by disease: '{DISEASE_NAME}'")
        
        # Visualization option
        while True:
            create_viz = input("Generate visualization plots? (y/n): ").strip().lower()
            if create_viz in ['y', 'n']:
                create_viz = create_viz == 'y'
                break
            else:
                print("Please enter y or n")
        
        print(f"\nPrediction mode parameters:")
        print(f"- Clusters: {num_clusters}")
        print(f"- Control genes: {top_genes}")
        print(f"- Correlation threshold: {'Auto' if corr_threshold is None else corr_threshold}")
        print(f"- Output combinations: {top_n}")
        print(f"- Visualizations: {'Yes' if create_viz else 'No'}")
        print(f"- Candidate source: {'Custom gene list' if use_user_gene_list else 'Built-in target pair'}")
        
    else:
        # Evaluation mode
        print("\n=== Evaluation Mode ===")
        print("Using default parameters:")
        num_clusters = 4
        top_genes = 50
        corr_threshold = 0.8
        use_auto_threshold = False
        top_n = 100
        create_viz = True
        use_user_gene_list = False
        user_gene_df = None
        print(f"- Clusters: {num_clusters}")
        print(f"- Control genes: {top_genes}")
        print(f"- Correlation threshold: {corr_threshold}")
        print(f"- Output combinations: {top_n}")
        print(f"- Visualizations: Yes")
    
    # Ensure output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # 1. Load data
        df, target_combination = load_and_preprocess_data(
            WEIGHTS_PATH, TARGET_PAIR_PATH, TARGET_INFO_PATH, DRUG_COMB_PATH
        )
        
        # 2. Gene importance
        df_with_stats = compute_gene_importance(df)
        
        # 3. Normalize & cluster
        df_scaled, gene_clusters = normalize_and_cluster_genes(df_with_stats, top_genes, num_clusters)
        
        # 4. Save normalized results
        output_path = os.path.join(OUTPUT_DIR, "Colorectal_target_essential.csv")
        df_scaled.sort_values(by='Total_Importance', ascending=False).to_csv(output_path)
        print(f"Normalized results saved to: {output_path}")
        
        # 5. Visualization
        if create_viz:
            trend_clusters = create_visualization(df_scaled, gene_clusters)
        else:
            print("Skipping visualization...")
            trend_clusters = gene_clusters
        
        # 6. Classify candidate genes
        classification_res = classify_candidate_genes(
            df_scaled, target_combination, DISEASE_NAME, trend_clusters,
            use_user_gene_list=use_user_gene_list, user_gene_df=user_gene_df
        )
        if classification_res is None:
            print("Candidate classification failed, aborting.")
            return
        classification_results, df_candidates = classification_res
        
        # 7. Analyze per-cluster counts and choose the best cluster
        print("\n=== Candidate counts per cluster ===")
        cluster_counts = classification_results['Assigned_Cluster'].value_counts()
        for cluster_name, count in cluster_counts.items():
            print(f"{cluster_name}: {count} candidate genes")
        
        # Choose the cluster with the most candidates
        target_cluster_name = cluster_counts.index[0]
        print(f"\nSelected {target_cluster_name} for analysis ({cluster_counts.iloc[0]} candidate genes)")
        
        # Fallback if too few candidates
        if cluster_counts.iloc[0] < 2:
            print("Warning: Selected cluster has fewer than 2 candidate genes, trying others...")
            for cluster_name, count in cluster_counts.items():
                if count >= 2:
                    target_cluster_name = cluster_name
                    print(f"Selected {target_cluster_name} for analysis ({count} candidate genes)")
                    break
            else:
                print("Error: All clusters have fewer than 2 candidate genes. Cannot generate combinations.")
                return
        
        # New: Auto correlation threshold based on number of candidate genes matched in the matrix
        if use_auto_threshold:
            candidate_count = int(df_candidates.shape[0])
            auto_threshold = round(candidate_count / 200.0, 1)
            auto_threshold = min(0.9, max(0.0, auto_threshold))
            corr_threshold = auto_threshold
            print(f"Auto-selected correlation threshold based on {candidate_count} candidates: {corr_threshold}")
        
        # 8. Generate combinations
        df_comb = generate_target_combinations(
            df_scaled, classification_results, target_cluster_name, 
            trend_clusters, corr_threshold, top_n
        )
        
        if df_comb is None:
            print("Combination generation failed, aborting.")
            return
        
        # 9. Save combinations
        comb_output_path = os.path.join(OUTPUT_DIR, "target_combinations.csv")
        df_comb.to_csv(comb_output_path, index=False)
        print(f"Target combinations saved to: {comb_output_path}")
        
        # 10. Evaluation (skip if custom list was used)
        if not use_user_gene_list:
            match_stats = evaluate_predictions(df_comb, target_combination, DISEASE_NAME, OUTPUT_DIR)
        else:
            print("Custom gene list used. Skipping evaluation against known combinations.")
        
        print("Done.")
        
    except Exception as e:
        print(f"Execution error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()