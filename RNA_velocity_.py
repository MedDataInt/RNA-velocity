## Overview of the Analysis Plan

#1. Load and QC the data from all batches
#2. Integrate data across batches
#3. Annotate cell types to identify CD8 T cells
#4. Create loom files for RNA velocity analysis
#5. Perform RNA velocity analysis specifically on CD8 T cells
#6. Compare velocity patterns between early and late rebound groups

"""
Explanation of the Analysis Pipeline
1. Data Preparation and Loading
The pipeline begins by loading all 11 10x Genomics output folders from your different batches and lanes. I've included code to:

Load gene expression matrices
Load hashtag oligonucleotide (HTO) data for sample demultiplexing
Match cells with their donor information (ID, viral rebound time, and group) from your sample_information.csv file
Perform QC filtering to remove low-quality cells and genes

2. CD8 T Cell Identification
To identify CD8 T cells, the pipeline:

Normalizes and scales the data
Performs dimensionality reduction (PCA and UMAP)
Runs clustering using the Leiden algorithm
Annotates cell types using canonical marker genes, with specific focus on CD8 T cell markers (CD8A, CD8B, GZMK, GZMB)
Extracts all cells annotated as CD8 T cells for downstream analysis

3. RNA Velocity Analysis
For RNA velocity analysis, the pipeline:

Creates loom files using velocyto for each sample (if they don't already exist)
Loads these loom files and filters for CD8 T cells
Computes RNA velocity using the stochastic model in scVelo
Projects velocity vectors onto UMAP embeddings
Compares velocity patterns between early and late rebound groups
Analyzes differentiation trajectories and pseudotime

4. Comparing Early vs. Late Rebound Groups
The final analysis compares CD8 T cell differentiation between early and late viral rebound groups by:

Separating cells based on rebound status
Calculating velocity pseudotime distributions
Identifying differentially expressed genes along velocity trajectories
Visualizing key driver genes in the differentiation process
Comparing trajectory dynamics between groups

Creating Loom Files in Python
To address your question about creating loom files in Python - yes, you can create them in Python! In this pipeline, I'm using velocyto's command-line tool, which is called from Python using os.system(). This approach is often preferred as velocyto's command-line tool handles the complex task of counting spliced and unspliced reads.
However, if you prefer a pure Python approach, you could use the anndata package to create your own loom files.
"""

# Example of creating a loom file in pure Python
import anndata as ad
import loompy

# Create AnnData object with spliced and unspliced counts
adata = ad.AnnData(X=spliced_counts, 
                  layers={'spliced': spliced_counts, 
                          'unspliced': unspliced_counts,
                          'ambiguous': ambiguous_counts})

# Add necessary attributes and metadata
adata.obs = cell_metadata
adata.var = gene_metadata

# Save as loom file
adata.write_loom("output.loom")




'''
Next Steps and Considerations

Adjust File Paths: Replace /path/to/your/data/ with your actual server path.
QC Parameters: You may need to adjust the QC filtering thresholds based on your specific dataset.
Cell Type Annotation Validation: Double-check the automated cell type annotations by visualizing marker gene expression.
velocyto Installation: Ensure velocyto is installed on your system, as the pipeline calls it via the command line.
Integration Methods: If batch effects are strong, you might want to enhance the integration step using harmony, BBKNN, or scVI.
'''




# Import necessary libraries
import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

# Set up visualization parameters
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100, frameon=False)
scv.settings.verbosity = 3
scv.settings.set_figure_params(dpi=100, frameon=False)
scv.settings.presenter_view = True

# Set up paths
base_dir = "/path/to/your/data/"  # Replace with your actual base directory
output_dir = os.path.join(base_dir, "velocity_output")
os.makedirs(output_dir, exist_ok=True)

# Load sample information
sample_info = pd.read_csv(os.path.join(base_dir, "sample_information.csv"))
print("Sample information loaded:", sample_info.shape)
print(sample_info.head())

# Define batch information
batch_files = {
    "BatchA": ["BatchA_Lane2", "BatchA_Lane3", "BatchA_Lane4"],
    "BatchB": ["BatchB_Lane1", "BatchB_Lane2", "BatchB_Lane3", "BatchB_Lane4"],
    "BatchC": ["BatchC_Lane1", "BatchC_Lane2", "BatchC_Lane3", "BatchC_Lane4"]
}

# Step 1: Load and QC all datasets
adatas = []
for batch, lanes in batch_files.items():
    for lane in lanes:
        # Define file paths for this lane
        data_path = os.path.join(base_dir, lane)
        
        # Load gene expression data
        print(f"Loading {lane} data...")
        adata = sc.read_10x_mtx(
            os.path.join(data_path, "filtered_feature_bc_matrix"),
            var_names='gene_symbols',
            cache=True
        )
        
        # Load HTO (hashtag) data for sample demultiplexing
        hto_path = os.path.join(data_path, "filtered_feature_bc_matrix_HTO")
        if os.path.exists(hto_path):
            hto = sc.read_10x_mtx(
                hto_path,
                var_names='gene_symbols',
                cache=True
            )
            # Extract HTO information and add to main object
            adata.obs['HTO'] = pd.Categorical(
                ["HTO-" + str(np.argmax(hto.X[i].toarray()) + 1) for i in range(hto.n_obs)]
            )
        
        # Add batch information
        adata.obs['batch'] = batch
        adata.obs['lane'] = lane
        
        # Basic QC
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        
        # Add to list of datasets
        adatas.append(adata)

# Step 2: Concatenate all datasets
print("Concatenating all datasets...")
adata_combined = ad.concat(adatas, merge="same")

# Annotate with sample information
adata_combined.obs['donor_id'] = ""
adata_combined.obs['vr_time'] = np.nan
adata_combined.obs['vr_group'] = ""

# Match each cell with its donor information based on batch and hash
for idx, row in adata_combined.obs.iterrows():
    batch = row['batch']
    hto = row['HTO']
    if isinstance(hto, str) and hto.startswith('HTO-'):
        hash_num = int(hto.split('-')[1])
        # Find matching donor in sample info
        match = sample_info[(sample_info['Batch'] == batch) & (sample_info['Hash'] == hash_num)]
        if not match.empty:
            adata_combined.obs.at[idx, 'donor_id'] = match.iloc[0]['Donor_ID']
            adata_combined.obs.at[idx, 'vr_time'] = match.iloc[0]['VR_time']
            adata_combined.obs.at[idx, 'vr_group'] = match.iloc[0]['VR_Group']

# QC filtering
print("Performing QC filtering...")
sc.pp.filter_cells(adata_combined, min_genes=200)
sc.pp.filter_genes(adata_combined, min_cells=3)

# Calculate QC metrics
mito_genes = adata_combined.var_names.str.startswith('MT-')
adata_combined.obs['percent_mito'] = np.sum(adata_combined[:, mito_genes].X, axis=1) / np.sum(adata_combined.X, axis=1) * 100

# Filter cells based on QC metrics
adata_combined = adata_combined[adata_combined.obs['n_genes_by_counts'] < 6000, :]
adata_combined = adata_combined[adata_combined.obs['percent_mito'] < 20, :]

# Normalize data
print("Normalizing data...")
sc.pp.normalize_total(adata_combined, target_sum=1e4)
sc.pp.log1p(adata_combined)

# Feature selection
print("Selecting highly variable genes...")
sc.pp.highly_variable_genes(adata_combined, n_top_genes=3000)

# Scale data
sc.pp.scale(adata_combined, max_value=10)

# Run PCA
print("Running PCA...")
sc.tl.pca(adata_combined, svd_solver='arpack')

# Compute neighbors
print("Computing neighbors...")
sc.pp.neighbors(adata_combined, n_neighbors=10, n_pcs=30)

# Run UMAP
print("Running UMAP...")
sc.tl.umap(adata_combined)

# Leiden clustering
print("Performing clustering...")
sc.tl.leiden(adata_combined, resolution=0.8)

# Step 3: Identify CD8 T cell clusters
# Define marker genes for cell type annotation
marker_genes = {
    'T cell': ['CD3D', 'CD3E', 'CD3G'],
    'CD4 T cell': ['CD4', 'IL7R'],
    'CD8 T cell': ['CD8A', 'CD8B', 'GZMK', 'GZMB'],
    'NK cell': ['GNLY', 'NKG7', 'NCAM1'],
    'B cell': ['CD19', 'MS4A1', 'CD79A'],
    'Monocyte': ['CD14', 'LYZ', 'CST3', 'FCGR3A'],
    'Dendritic cell': ['FCER1A', 'CST3', 'CLEC9A'],
    'Neutrophil': ['S100A8', 'S100A9', 'FCGR3B']
}

# Calculate marker gene expression
print("Calculating marker gene expression...")
sc.tl.rank_genes_groups(adata_combined, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata_combined, n_genes=25, sharey=False)

# Annotate clusters based on marker genes
def calculate_mean_expression(adata, cluster, genes):
    cells = adata[adata.obs['leiden'] == cluster].X
    gene_indices = [adata.var_names.get_loc(gene) for gene in genes if gene in adata.var_names]
    if len(gene_indices) == 0:
        return 0
    return np.mean([np.mean(cells[:, idx]) for idx in gene_indices])

# Automatically annotate clusters
cluster_annotations = {}
for cluster in adata_combined.obs['leiden'].unique():
    scores = {}
    for cell_type, genes in marker_genes.items():
        scores[cell_type] = calculate_mean_expression(adata_combined, cluster, genes)
    cluster_annotations[cluster] = max(scores, key=scores.get)

# Add cell type annotations to the object
adata_combined.obs['cell_type'] = adata_combined.obs['leiden'].map(cluster_annotations)

# Specifically look at clusters with high CD8 T cell marker expression for validation
print("Validating CD8 T cell clusters...")
sc.pl.dotplot(adata_combined, marker_genes['CD8 T cell'] + marker_genes['CD4 T cell'], 
              groupby='leiden', dendrogram=True)

# Extract CD8 T cells for further analysis
adata_cd8 = adata_combined[adata_combined.obs['cell_type'] == 'CD8 T cell'].copy()
print(f"CD8 T cells extracted: {adata_cd8.n_obs} cells")

# Step 4: Create loom files for RNA velocity analysis
print("Preparing data for RNA velocity analysis...")

# Process all batches to create loom files
all_loom_files = []

for batch, lanes in batch_files.items():
    for lane in lanes:
        # Define file paths for this lane
        data_path = os.path.join(base_dir, lane)
        
        # Check if loom file already exists
        loom_file = os.path.join(output_dir, f"{lane}_velocyto.loom")
        all_loom_files.append(loom_file)
        
        if not os.path.exists(loom_file):
            print(f"Creating loom file for {lane}...")
            # Define the command to run velocyto
            cmd = f"velocyto run10x -m {os.path.join(base_dir, 'repeat_msk.gtf')} {data_path} {os.path.join(base_dir, 'refdata-gex-GRCh38-2020-A/genes/genes.gtf')}"
            
            # Execute the command
            print(f"Running command: {cmd}")
            os.system(cmd)
            
            # velocyto typically saves the loom file in the input directory, move it to output_dir
            original_loom = os.path.join(data_path, "velocyto", f"{os.path.basename(data_path)}.loom")
            if os.path.exists(original_loom):
                os.rename(original_loom, loom_file)
        else:
            print(f"Loom file for {lane} already exists.")

# Step 5: Load and process loom files for RNA velocity
print("Loading loom files for RNA velocity analysis...")

# Load all loom files and merge them
loom_data = []
for loom_file in all_loom_files:
    if os.path.exists(loom_file):
        print(f"Loading {loom_file}...")
        try:
            ldata = scv.read_loom(loom_file)
            # Add batch info
            lane_name = os.path.basename(loom_file).replace("_velocyto.loom", "")
            for batch, lanes in batch_files.items():
                if lane_name in lanes:
                    ldata.obs['batch'] = batch
                    ldata.obs['lane'] = lane_name
                    break
            loom_data.append(ldata)
        except Exception as e:
            print(f"Error loading {loom_file}: {e}")

# Concatenate all loom data
print("Concatenating loom data...")
if loom_data:
    adata_loom = ad.concat(loom_data, merge="same")
    print(f"Combined loom data: {adata_loom.shape}")
else:
    print("No loom files could be loaded. Check paths and file integrity.")
    exit(1)

# Merge loom data with cell annotations
# Create a dictionary of cell barcodes to annotations
cell_annotations = adata_cd8.obs[['cell_type', 'vr_group', 'donor_id', 'batch', 'lane']].to_dict('index')

# Match barcodes with loom data
# We need to adjust for potential barcode differences (suffix -1 often added by velocyto)
common_cells = []
for bc in adata_loom.obs.index:
    # Try different barcode formats
    bc_stripped = bc.split('-')[0]
    if bc in cell_annotations:
        common_cells.append(bc)
    elif bc_stripped in cell_annotations:
        common_cells.append(bc)

print(f"Common cells found: {len(common_cells)}")

# Filter loom data to only include CD8 T cells
if common_cells:
    adata_loom_cd8 = adata_loom[adata_loom.obs.index.isin(common_cells)]
    print(f"CD8 T cells in loom data: {adata_loom_cd8.shape}")
    
    # Add cell annotations from the main object
    for key in ['vr_group', 'donor_id', 'batch', 'lane']:
        adata_loom_cd8.obs[key] = ""
        
    for bc in adata_loom_cd8.obs.index:
        bc_stripped = bc.split('-')[0]
        if bc in cell_annotations:
            for key in ['vr_group', 'donor_id', 'batch', 'lane']:
                adata_loom_cd8.obs.loc[bc, key] = cell_annotations[bc][key]
        elif bc_stripped in cell_annotations:
            for key in ['vr_group', 'donor_id', 'batch', 'lane']:
                adata_loom_cd8.obs.loc[bc, key] = cell_annotations[bc_stripped][key]
else:
    print("No common cells found between annotated data and loom files.")
    exit(1)

# Step 6: RNA velocity analysis on CD8 T cells
print("Performing RNA velocity analysis on CD8 T cells...")

# Preprocess the data
scv.pp.filter_and_normalize(adata_loom_cd8)
scv.pp.moments(adata_loom_cd8)

# Compute velocity
print("Computing RNA velocity...")
scv.tl.velocity(adata_loom_cd8, mode='stochastic')
scv.tl.velocity_graph(adata_loom_cd8)

# Visualization and projection
# First, let's calculate embeddings
scv.pp.neighbors(adata_loom_cd8)
scv.tl.umap(adata_loom_cd8)
scv.tl.louvain(adata_loom_cd8)

# Plot velocity stream
print("Plotting velocity streams...")
scv.pl.velocity_embedding_stream(adata_loom_cd8, basis='umap', save=os.path.join(output_dir, 'velocity_stream_cd8_all.png'))

# Separate analysis for early vs late rebound
print("Comparing early vs late rebound...")

# Filter for cells with valid rebound group information
valid_vr = adata_loom_cd8[adata_loom_cd8.obs['vr_group'].isin(['early rebound', 'late rebound'])]

if len(valid_vr) > 0:
    # Plot velocity by rebound group
    scv.pl.velocity_embedding_stream(
        valid_vr, basis='umap', color='vr_group',
        save=os.path.join(output_dir, 'velocity_stream_cd8_by_vr_group.png')
    )
    
    # Calculate velocity pseudotime
    scv.tl.velocity_pseudotime(valid_vr)
    scv.pl.scatter(valid_vr, color='velocity_pseudotime', cmap='gnuplot', 
                   save=os.path.join(output_dir, 'velocity_pseudotime_cd8.png'))
    
    # Compare pseudotime distributions between groups
    vr_groups = valid_vr.obs['vr_group'].unique()
    plt.figure(figsize=(10, 6))
    for vr in vr_groups:
        group_data = valid_vr[valid_vr.obs['vr_group'] == vr]
        sns.kdeplot(group_data.obs['velocity_pseudotime'], label=vr)
    plt.title('Velocity Pseudotime Distribution by Viral Rebound Group')
    plt.xlabel('Velocity Pseudotime')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'pseudotime_distribution_comparison.png'))
    plt.close()
    
    # Differential gene expression along the pseudotime
    scv.tl.rank_velocity_genes(valid_vr, groupby='vr_group')
    scv.pl.velocity_genes(valid_vr, save=os.path.join(output_dir, 'top_velocity_genes_by_group.png'))
    
    # Get top differentially expressed genes between groups
    top_genes = scv.get_df(valid_vr, 'rank_velocity_genes/names')
    
    # Plot phase portraits for top genes
    for group in top_genes.keys():
        genes = list(top_genes[group][:5])  # Top 5 genes
        scv.pl.velocity(valid_vr, genes, ncols=5, 
                        save=os.path.join(output_dir, f'phase_portraits_{group}.png'))
    
    # Run PAGA trajectory analysis
    scv.tl.paga(valid_vr, groups='vr_group')
    scv.pl.paga(valid_vr, basis='umap', size=50, alpha=0.15,
                min_edge_width=2, node_size_scale=1.5,
                save=os.path.join(output_dir, 'paga_trajectory.png'))
    
    # Velocity-inferred driver genes
    scv.tl.recover_dynamics(valid_vr)
    scv.tl.latent_time(valid_vr)
    scv.pl.scatter(valid_vr, color='latent_time', color_map='gnuplot',
                   save=os.path.join(output_dir, 'latent_time.png'))
    
    scv.tl.velocity_confidence(valid_vr)
    keys = 'velocity_length', 'velocity_confidence'
    scv.pl.scatter(valid_vr, c=keys, cmap='coolwarm', perc=[5, 95],
                   save=os.path.join(output_dir, 'velocity_confidence.png'))
    
    # Find top driver genes
    scv.tl.rank_dynamical_genes(valid_vr)
    df = scv.get_df(valid_vr, 'rank_dynamical_genes/names')
    top_genes = df.head(20)
    scv.pl.heatmap(valid_vr, var_names=top_genes, sortby='latent_time', col_color='vr_group',
                   n_convolve=100, save=os.path.join(output_dir, 'driver_genes_heatmap.png'))
    
    # Save the final analyzed object
    valid_vr.write(os.path.join(output_dir, 'cd8_velocity_analyzed.h5ad'))
    
    print("Analysis complete! Results saved to:", output_dir)
else:
    print("Not enough cells with valid viral rebound information for comparison.")

print("Pipeline execution completed!")