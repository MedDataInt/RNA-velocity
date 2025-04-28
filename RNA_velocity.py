## python
I have eleven 10x outputs files from scCITE-seq data from different batches and lines, those are PBMC samples from HIV people with ART, the file names are BatchA_Lane2, BatchA_Lane3, BatchA_Lane4,  BatchB_Lane1,BatchB_Lane2, BatchB_Lane3, BatchB_Lane4,BatchC_Lane1,BatchC_Lane2, BatchC_Lane3, BatchC_Lane4. those are saved in my server. and the samples were hashed, for example, in batch 1, hash 1 will be donor ID 0003wk60, and hash 2 is donor ID 00027wk60, and those are HIV samples with virus rebound time information. if rebound greater than 19 days will be considered as late rebound group, and rebound time less than 19 days, will consider early rebound. and there are a total of 27 donor  PBMC samples, which contain T cells, neutrophils, and others. Now I want to run single RNA velocity analysis on only CD8 T cell clusters, and compare the RNA velocity of CD8 T populations  in early rebound time vs. late rebound. Please provide full structure showing how to perform this, including how to prepare my input data? and how to identify CD8 T cells for RNA velocity analysis, and how to do the downstream analysis, the final goal is to see if CD8 T cells differentiaon show difference in early vs. late rebound samples. Please provide me the code in very details.


## Part 1: Data Loading and Integration
import scanpy as sc
import anndata as ad
import scvelo as scv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import sparse
from sklearn.decomposition import PCA
import cellrank as cr

# Set up the environment for analysis
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, figsize=(10, 10))
scv.settings.set_figure_params('scvelo', dpi=100, figsize=(10, 10))
scv.settings.verbosity = 3

# Define paths to data
data_path = "/work/05/gkundu/ATI_wk0_60_CITESeq/"
batches = ["BatchA_Lane2", "BatchA_Lane3", "BatchA_Lane4", 
           "BatchB_Lane1", "BatchB_Lane2", "BatchB_Lane3", "BatchB_Lane4",
           "BatchC_Lane1", "BatchC_Lane2", "BatchC_Lane3", "BatchC_Lane4"]

# Create a mapping of hash IDs to donor IDs and rebound status
donor_info = pd.DataFrame({
    'Hash': ['1', '2', '3', '4'],  # Fill with your hash IDs
    'donor_id': ['0003wk60', '00027wk60', '...'],   # Fill with your donor IDs
    'rebound_time': [25, 15, '...'],                # Fill with rebound times
    'rebound_group': ['late', 'early', '...']       # 'late' if >= 19 days, 'early' if < 19 days
})

# Load all datasets
adata_list = []
for batch in batches:
    print(f"Loading {batch}...")
    path = os.path.join(data_path, batch)
    adata = sc.read_10x_mtx(path, var_names='gene_symbols', cache=True)
    
    # Add batch information
    adata.obs['batch'] = batch
    
    # Extract hash IDs for this batch - you'll need to adjust this based on how your hash info is stored
    hash_path = os.path.join(data_path, f"{batch}_hash_assignment.csv")  # Adjust file name as needed
    if os.path.exists(hash_path):
        hash_info = pd.read_csv(hash_path)
        # Merge hash information with adata.obs
        hash_info.index = hash_info['cell_barcode']  # Adjust column name as needed
        adata.obs = adata.obs.join(hash_info[['hash_id']], how='left')
    
    adata_list.append(adata)

# Concatenate all data
adata = ad.concat(adata_list, join='outer', merge='same')

# Map donor IDs and rebound groups based on hash IDs
adata.obs = adata.obs.join(donor_info.set_index('hash_id'), on='hash_id', how='left')

# Load spliced/unspliced matrices for RNA velocity
for batch in batches:
    print(f"Loading spliced/unspliced data for {batch}...")
    path = os.path.join(data_path, batch)
    
    # This assumes your velocity data is in the same directory as your count matrices
    # If using velocyto output, you might need to adjust the path
    try:
        scv.read(path, cache=True)
    except:
        print(f"Velocity data not found for {batch}, running preprocessing...")
        # If velocity data doesn't exist, process from the raw data
        scv.pp.run_cellranger_to_velocity(path, write_loom=True)
    
# Merge velocity data with our main AnnData object
ldata_list = []
for batch in batches:
    path = os.path.join(data_path, batch)
    # Load the loom file that was created by velocyto or cellranger
    ldata = scv.read(path + '/velocyto/velocyto.loom', cache=True)
    # Add batch information
    ldata.obs['batch'] = batch
    ldata_list.append(ldata)

# Merge all loom data
ldata = ad.concat(ldata_list, join='outer', merge='same')

# Make sure cell barcodes match between adata and ldata
common_cells = np.intersect1d(adata.obs_names, ldata.obs_names)
adata = adata[common_cells].copy()
ldata = ldata[common_cells].copy()

# Transfer spliced/unspliced counts to the main object
adata.layers['spliced'] = ldata.layers['spliced']
adata.layers['unspliced'] = ldata.layers['unspliced']

###### Part 2: Quality Control and Preprocessing
# Basic quality control
sc.pp.calculate_qc_metrics(adata, inplace=True)

# Filter cells based on QC metrics
adata = adata[adata.obs.n_genes_by_counts < 6000, :].copy()  # Remove likely doublets
adata = adata[adata.obs.n_genes_by_counts > 500, :].copy()   # Remove low-quality cells
adata = adata[adata.obs.pct_counts_mt < 20, :].copy()        # Filter cells with high mitochondrial content

# Filter genes
sc.pp.filter_genes(adata, min_cells=10)  # Keep genes expressed in at least 10 cells

# Normalize and log-transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Save the raw data for later use
adata.raw = adata.copy()

# Identify highly variable genes
sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)

# Keep only highly variable genes for further analysis
adata = adata[:, adata.var.highly_variable].copy()

# Scale the data
sc.pp.scale(adata, max_value=10)

# Run PCA
sc.tl.pca(adata, svd_solver='arpack')

# Batch correction using Harmony
import harmonypy as hp
# Get the PCA matrix
X_pca = adata.obsm['X_pca']
# Get batch variable
batch_labels = adata.obs['batch'].values
# Run Harmony
harmony_operator = hp.run_harmony(X_pca, batch_labels, max_iter_harmony=50)
# Store the corrected matrix
adata.obsm['X_harmony'] = harmony_operator.Z_corr.T

# Compute neighbors using harmony corrected PCA
sc.pp.neighbors(adata, use_rep='X_harmony')

# Run UMAP and clustering
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.8)

# Plot initial clustering
sc.pl.umap(adata, color=['leiden', 'batch'], ncols=1


###### Part 3: Cell Type Annotation and CD8 T Cell Identification
# Define marker genes for different cell types
marker_genes = {
    'CD8 T cells': ['CD8A', 'CD8B', 'CD3E', 'CD3D', 'CD3G', 'GZMB', 'GZMK', 'PRF1'],
    'CD4 T cells': ['CD4', 'CD3E', 'CD3D', 'CD3G', 'IL7R'],
    'B cells': ['CD19', 'MS4A1', 'CD79A', 'CD79B'],
    'NK cells': ['NCAM1', 'NKG7', 'GNLY', 'KLRD1', 'KLRF1'],
    'Monocytes': ['CD14', 'FCGR3A', 'MS4A7', 'CST3'],
    'Dendritic cells': ['FCER1A', 'CST3', 'CLEC9A', 'CLEC10A'],
    'Neutrophils': ['FCGR3B', 'CSF3R', 'S100A8', 'S100A9']
}

# Plot expression of marker genes
sc.pl.dotplot(adata, marker_genes, groupby='leiden')
sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', rotation=90)

# Define CD8 T cell clusters based on marker expression
# This is a manual step - examine the plots and identify which clusters express CD8 T cell markers
cd8_clusters = ['0', '2', '5']  # Example - replace with your actual CD8 T cell clusters

# Label the cell types
adata.obs['cell_type'] = 'Unknown'
for cluster in cd8_clusters:
    adata.obs.loc[adata.obs['leiden'] == cluster, 'cell_type'] = 'CD8 T cells'

# Alternatively, use automatic annotation tools
# Import the sctype labeling function (you need to install cellxgene first)
# from cellxgene import auto_annotate
# adata = auto_annotate(adata, organism='human', cluster_col='leiden')

# Extract only CD8 T cells for velocity analysis
adata_cd8 = adata[adata.obs['cell_type'] == 'CD8 T cells'].copy()

# Re-run dimensionality reduction and clustering on CD8 T cells only
sc.pp.neighbors(adata_cd8, use_rep='X_harmony')
sc.tl.umap(adata_cd8)
sc.tl.leiden(adata_cd8, resolution=0.8)  # Higher resolution to identify CD8 subtypes

# Plot CD8 T cell clusters
sc.pl.umap(adata_cd8, color=['leiden', 'rebound_group', 'batch'])

# Define CD8 T cell subtypes based on marker genes
cd8_subtype_markers = {
    'Naive': ['CCR7', 'SELL', 'TCF7', 'LEF1', 'IL7R'],
    'Central Memory': ['CCR7', 'SELL', 'IL7R', 'CD27'],
    'Effector Memory': ['GZMK', 'GZMA', 'CCL5', 'CST7'],
    'Effector': ['GZMB', 'GZMH', 'PRF1', 'CX3CR1', 'FCGR3A'],
    'Tissue Resident Memory': ['CXCR6', 'ITGAE', 'ITGA1'],
    'Exhausted': ['PDCD1', 'HAVCR2', 'LAG3', 'TIGIT', 'TOX']
}

# Plot expression of CD8 subtype markers
sc.pl.dotplot(adata_cd8, cd8_subtype_markers, groupby='leiden')

# Label CD8 subtypes based on marker expression (manual step)
adata_cd8.obs['cd8_subtype'] = 'Unknown'
subtype_mapping = {
    '0': 'Naive',
    '1': 'Central Memory',
    '2': 'Effector Memory',
    '3': 'Effector',
    '4': 'Exhausted'
}  # Example - replace with your actual mapping

for cluster, subtype in subtype_mapping.items():
    adata_cd8.obs.loc[adata_cd8.obs['leiden'] == cluster, 'cd8_subtype'] = subtype

# Plot CD8 subtypes
sc.pl.umap(adata_cd8, color='cd8_subtype')


######  Part 4: RNA Velocity Analysis
# Preprocess the velocity data
scv.pp.filter_and_normalize(adata_cd8, min_shared_counts=20, min_shared_cells=15)
scv.pp.moments(adata_cd8, n_pcs=30, n_neighbors=30)

# Compute velocity
scv.tl.velocity(adata_cd8, mode='stochastic')
scv.tl.velocity_graph(adata_cd8)

# Project the velocities on the UMAP embedding
scv.pl.velocity_embedding_stream(adata_cd8, basis='umap', color='cd8_subtype')
scv.pl.velocity_embedding(adata_cd8, basis='umap', arrow_length=3, arrow_size=2, color='cd8_subtype')

# Perform velocity-based pseudotime analysis
scv.tl.recover_dynamics(adata_cd8)
scv.tl.latent_time(adata_cd8)
scv.pl.scatter(adata_cd8, color='latent_time', cmap='gnuplot')

# Identify terminal states and driver genes
scv.tl.terminal_states(adata_cd8)
scv.pl.scatter(adata_cd8, color=['root_cells', 'end_points'])
scv.tl.rank_velocity_genes(adata_cd8, groupby='cd8_subtype', min_corr=.3)
scv.pl.velocity_genes(adata_cd8)

# Identify top driver genes
top_genes = adata_cd8.var['fit_likelihood'].sort_values(ascending=False).index[:100]
scv.pl.scatter(adata_cd8, basis=top_genes[:15], ncols=5, color='cd8_subtype')

# Use CellRank for lineage analysis
# CellRank extends RNA velocity to predict cell fate and lineage probabilities
cr.tl.terminal_states(adata_cd8, cluster_key='cd8_subtype')
cr.pl.terminal_states(adata_cd8)

# Compute fate probabilities
cr.tl.lineages(adata_cd8)
cr.pl.lineages(adata_cd8)

# Visualize gene expression dynamics along trajectories
cr.pl.gene_trends(adata_cd8, genes=['TCF7', 'GZMB', 'PDCD1']


###### Part 5: Comparing Early vs. Late Rebound Groups

Split the data by rebound group
adata_early = adata_cd8[adata_cd8.obs['rebound_group'] == 'early'].copy()
adata_late = adata_cd8[adata_cd8.obs['rebound_group'] == 'late'].copy()

# Calculate RNA velocity separately for each group
for data, group in [(adata_early, 'early'), (adata_late, 'late')]:
    # Recompute velocity for the subset
    scv.pp.filter_and_normalize(data, min_shared_counts=20, min_shared_cells=15)
    scv.pp.moments(data, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(data, mode='stochastic')
    scv.tl.velocity_graph(data)
    
    # Plot velocity streams
    plt.figure(figsize=(10, 10))
    scv.pl.velocity_embedding_stream(data, basis='umap', color='cd8_subtype', 
                                     title=f"{group} rebound group")

# Compare latent time distributions
scv.tl.latent_time(adata_early)
scv.tl.latent_time(adata_late)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
sns.violinplot(x="cd8_subtype", y="latent_time", data=adata_early.obs, ax=ax1)
ax1.set_title("Early Rebound")
ax1.set_ylim(0, 1)
sns.violinplot(x="cd8_subtype", y="latent_time", data=adata_late.obs, ax=ax2)
ax2.set_title("Late Rebound")
ax2.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# Identify differentially expressed genes between the two groups
sc.tl.rank_genes_groups(adata_cd8, 'rebound_group', method='wilcoxon')
sc.pl.rank_genes_groups(adata_cd8, n_genes=20, sharey=False)

# Export the differentially expressed genes to a CSV file
result = adata_cd8.uns['rank_genes_groups']
groups = result['names'].dtype.names
data = []
for group in groups:
    genes = [result['names'][i][group] for i in range(len(result['names']))]
    scores = [result['scores'][i][group] for i in range(len(result['scores']))]
    pvals = [result['pvals'][i][group] for i in range(len(result['pvals']))]
    pvals_adj = [result['pvals_adj'][i][group] for i in range(len(result['pvals_adj']))]
    logfoldchanges = [result['logfoldchanges'][i][group] for i in range(len(result['logfoldchanges']))]
    
    for i in range(len(genes)):
        data.append([group, genes[i], scores[i], logfoldchanges[i], pvals[i], pvals_adj[i]])

df = pd.DataFrame(data, columns=['group', 'gene', 'score', 'log2fc', 'pval', 'pval_adj'])
df.to_csv("differential_expression_by_rebound_group.csv", index=False)

# Compare the RNA velocity between rebound groups for each CD8 subtype
subtypes = adata_cd8.obs['cd8_subtype'].unique()

for subtype in subtypes:
    # Extract cells of this subtype
    mask_early = (adata_cd8.obs['rebound_group'] == 'early') & (adata_cd8.obs['cd8_subtype'] == subtype)
    mask_late = (adata_cd8.obs['rebound_group'] == 'late') & (adata_cd8.obs['cd8_subtype'] == subtype)
    
    if sum(mask_early) > 10 and sum(mask_late) > 10:  # Ensure we have enough cells
        adata_subtype_early = adata_cd8[mask_early].copy()
        adata_subtype_late = adata_cd8[mask_late].copy()
        
        # Compute velocity-driver genes for each group
        scv.tl.rank_velocity_genes(adata_subtype_early, min_corr=.3)
        early_drivers = adata_subtype_early.var['velocity_genes'].sort_values(ascending=False).index[:50]
        
        scv.tl.rank_velocity_genes(adata_subtype_late, min_corr=.3)
        late_drivers = adata_subtype_late.var['velocity_genes'].sort_values(ascending=False).index[:50]
        
        # Find unique drivers for each group
        early_unique = set(early_drivers) - set(late_drivers)
        late_unique = set(late_drivers) - set(early_drivers)
        
        print(f"\nCD8 {subtype} unique velocity drivers:")
        print(f"Early rebound: {', '.join(list(early_unique)[:10])}")
        print(f"Late rebound: {', '.join(list(late_unique)[:10])}")
        
        # Compare velocity magnitudes
        early_vel = np.linalg.norm(adata_subtype_early.layers['velocity'], axis=1).mean()
        late_vel = np.linalg.norm(adata_subtype_late.layers['velocity'], axis=1).mean()
        
        print(f"Mean velocity magnitude - Early: {early_vel:.4f}, Late: {late_vel:.4f}"
        
        
######## Part 6: Advanced Trajectory Analysis for CD8 Differentiation

# Use CellRank for detailed trajectory analysis
import cellrank as cr

# Extract terminal states (end points of trajectories)
cr.tl.terminal_states(adata_cd8, cluster_key='cd8_subtype')
cr.pl.terminal_states(adata_cd8)

# Identify transition genes - genes that change along trajectories
cr.tl.lineages(adata_cd8)
cr.tl.gene_importance(adata_cd8)
cr.pl.gene_importance(adata_cd8, n_genes=15)

# Compare transition probabilities between rebound groups
def calculate_transition_probs(adata, source_type, target_types):
    transition_probs = {}
    for target in target_types:
        if target != source_type:
            # Get cells of the source type
            source_cells = adata[adata.obs['cd8_subtype'] == source_type].obs_names
            
            # Extract the transition probabilities to the target type
            target_probs = adata.obsp['transition_matrix'][np.ix_(
                np.where(np.isin(adata.obs_names, source_cells))[0],
                np.where(adata.obs['cd8_subtype'] == target)[0]
            )]
            
            # Calculate the mean transition probability
            transition_probs[target] = target_probs.mean()
    
    return transition_probs

# Calculate transition probabilities for different CD8 subtypes in each group
subtypes = list(adata_cd8.obs['cd8_subtype'].unique())
source_subtypes = ['Naive', 'Central Memory']  # Example source states

for source in source_subtypes:
    print(f"\nTransition probabilities from {source}:")
    
    # Early rebound group
    early_probs = calculate_transition_probs(adata_early, source, subtypes)
    print("Early rebound:")
    for target, prob in early_probs.items():
        print(f"  To {target}: {prob:.4f}")
    
    # Late rebound group
    late_probs = calculate_transition_probs(adata_late, source, subtypes)
    print("Late rebound:")
    for target, prob in late_probs.items():
        print(f"  To {target}: {prob:.4f}")

# Plot key differentiation markers along pseudotime
genes_of_interest = [
    'TCF7',      # Naive/memory marker
    'CCR7',      # Naive/memory marker
    'GZMB',      # Effector marker
    'PRF1',      # Effector marker
    'PDCD1',     # Exhaustion marker
    'TOX',       # Exhaustion marker
    'IFNG',      # Cytokine
    'TNF'        # Cytokine
]

# Plot gene trends along pseudotime for both groups
fig, axes = plt.subplots(len(genes_of_interest), 2, figsize=(12, 3*len(genes_of_interest)))

for i, gene in enumerate(genes_of_interest):
    if gene in adata_early.var_names and gene in adata_late.var_names:
        # Early rebound
        ax1 = axes[i, 0]
        sc.pl.scatter(adata_early, x='latent_time', y=gene, color='cd8_subtype', ax=ax1, show=False)
        ax1.set_title(f"{gene} - Early Rebound")
        
        # Late rebound
        ax2 = axes[i, 1]
        sc.pl.scatter(adata_late, x='latent_time', y=gene, color='cd8_subtype', ax=ax2, show=False)
        ax2.set_title(f"{gene} - Late Rebound")

plt.tight_layout()
plt.show()

# Create a summary of differences in CD8 differentiation
summary_data = []

for subtype in subtypes:
    # Count cells in each group
    early_count = np.sum(adata_early.obs['cd8_subtype'] == subtype)
    late_count = np.sum(adata_late.obs['cd8_subtype'] == subtype)
    
    # Calculate proportions
    early_prop = early_count / len(adata_early) * 100
    late_prop = late_count / len(adata_late) * 100
    
    # Calculate average latent time (pseudotime)
    early_time = adata_early[adata_early.obs['cd8_subtype'] == subtype].obs['latent_time'].mean()
    late_time = adata_late[adata_late.obs['cd8_subtype'] == subtype].obs['latent_time'].mean()
    
    summary_data.append({
        'CD8 Subtype': subtype,
        'Early Count': early_count,
        'Early %': early_prop,
        'Early Pseudotime': early_time,
        'Late Count': late_count,
        'Late %': late_prop,
        'Late Pseudotime': late_time,
        'Proportion Difference': late_prop - early_prop,
        'Pseudotime Difference': late_time - early_time
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df)
summary_df.to_csv("cd8_subtype_comparison.csv", index=False)

# Create a comprehensive visualization of the differences
plt.figure(figsize=(12, 6))
sns.barplot(x='CD8 Subtype', y='Proportion Difference', data=summary_df)
plt.title('Difference in CD8 Subtype Proportions (Late - Early)')
plt.axhline(y=0, color='k', linestyle='--')
plt.ylabel('Percentage Difference')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Save the analyzed data
adata_cd8.write('cd8_velocity_analysis.h5ad')


###### Interpretation and Visualization
# Create a comprehensive figure for publication
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 3)

# 1. UMAP with CD8 subtypes
ax1 = plt.subplot(gs[0, 0])
sc.pl.umap(adata_cd8, color='cd8_subtype', ax=ax1, show=False, title='CD8 T Cell Subtypes')

# 2. UMAP with rebound groups
ax2 = plt.subplot(gs[0, 1])
sc.pl.umap(adata_cd8, color='rebound_group', ax=ax2, show=False, title='Rebound Groups')

# 3. RNA velocity stream plot
ax3 = plt.subplot(gs[0, 2])
scv.pl.velocity_embedding_stream(adata_cd8, basis='umap', color='cd8_subtype', ax=ax3, show=False, title='RNA Velocity')

# 4. Latent time distribution
ax4 = plt.subplot(gs[1, 0])
sc.pl.umap(adata_cd8, color='latent_time', cmap='gnuplot', ax=ax4, show=False, title='Pseudotime')

# 5. Velocity-based cell fates
ax5 = plt.subplot(gs[1, 1])
scv.pl.velocity_embedding(adata_cd8, basis='umap', arrow_length=3, arrow_size=2, ax=ax5, show=False, title='Cell Differentiation Vectors')

# 6. Comparison of subtypes between groups
ax6 = plt.subplot(gs[1, 2])
props = pd.DataFrame({
    'Early': adata_early.obs['cd8_subtype'].value_counts().reindex(subtypes) / len(adata_early) * 100,
    'Late': adata_late.obs['cd8_subtype'].value_counts().reindex(subtypes) / len(adata_late) * 100
})
props.plot(kind='bar', ax=ax6)
ax6.set_title('CD8 Subtype Distribution by Rebound Group')
ax6.set_ylabel('Percentage of Cells')
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)

# 7. Key marker genes along pseudotime
ax7 = plt.subplot(gs[2, 0:2])
genes = ['TCF7', 'GZMB', 'PDCD1', 'IFNG']  # Key differentiation markers
scv.pl.scatter(adata_cd8, x='latent_time', y=genes, frameon=False, ax=ax7, show=False, ncols=4)
ax7.set_title('Key Markers Along Pseudotime')

# 8. Transition probability heatmap
ax8 = plt.subplot(gs[2, 2])
transition_matrix = pd.DataFrame(index=subtypes, columns=subtypes)
for source in subtypes:
    for target in subtypes:
        if source != target:
            source_cells = adata_cd8[adata_cd8.obs['cd8_subtype'] == source].obs_names
            target_cells = adata_cd8[adata_cd8.obs['cd8_subtype'] == target].obs_names
            if len(source_cells) > 0 and len(target_cells) > 0:
                trans_prob = adata_cd8.obsp['velocity_graph'][np.ix_(
                    np.where(np.isin(adata_cd8.obs_names, source_cells))[0],
                    np.where(np.isin(adata_cd8.obs_names, target_cells))[0]
                )].mean()
                transition_matrix.loc[source, target] = trans_prob
            else:
                transition_matrix.loc[source, target] = 0
                
sns.heatmap(transition_matrix, cmap='viridis', ax=ax8)
ax8.set_title('Transition Probabilities Between Subtypes')

plt.tight_layout()
plt.savefig('cd8_velocity_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate a comprehensive HTML report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Extract UMAP coordinates
umap_coords = adata_cd8.obsm['X_umap']
df_plot = pd.DataFrame(umap_coords, columns=['UMAP1', 'UMAP2'])
df_plot['CD8 Subtype'] = adata_cd8.obs['cd8_subtype'].values
df_plot['Rebound Group'] = adata_cd8.obs['rebound_group'].values
df_plot['Pseudotime'] = adata_cd8.obs['latent_time'].values

# Create interactive plots
fig1 = px.scatter(df_plot, x='UMAP1', y='UMAP2', color='CD8 Subtype', 
                  title='CD8 T Cell Subtypes', hover_data=['Pseudotime'])

fig2 = px.scatter(df_plot, x='UMAP1', y='UMAP2', color='Rebound Group',
                  title='Early vs Late Rebound', hover_data=['CD8 Subtype', 'Pseudotime'])

fig3 = px.scatter(df_plot, x='UMAP1', y='UMAP2', color='Pseudotime',
                  color_continuous_scale='Viridis', title='Pseudotime')
