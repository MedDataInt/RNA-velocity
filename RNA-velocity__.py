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
adatas_protein = []

for batch, lanes in batch_files.items():
    for lane in lanes:
        # Define file paths for this lane
        data_path = os.path.join(base_dir, lane)
        
        print(f"Loading {lane} data...")
        
        # Load gene expression data (RNA)
        adata_rna = sc.read_10x_h5(
            os.path.join(data_path, "filtered_feature_bc_matrix.h5"),
            gex_only=False  # This will load all modalities
        )
        
        # Check if there's feature_reference.csv to understand feature types
        feature_ref_path = os.path.join(data_path, "feature_reference.csv")
        if os.path.exists(feature_ref_path):
            feature_ref = pd.read_csv(feature_ref_path)
            print(f"Feature reference loaded for {lane}: {feature_ref.shape}")
        
        # Separate RNA and protein data
        # Find if we have 'feature_types' in var to separate modalities
        if 'feature_types' in adata_rna.var:
            # Filter for gene expression data
            gene_mask = adata_rna.var['feature_types'] == 'Gene Expression'
            protein_mask = adata_rna.var['feature_types'] == 'Antibody Capture'
            
            # Create separate objects for RNA and proteins
            adata_genes = adata_rna[:, gene_mask].copy()
            adata_protein = adata_rna[:, protein_mask].copy() if np.any(protein_mask) else None
            
            print(f"Lane {lane}: {adata_genes.n_vars} genes and {adata_protein.n_vars if adata_protein else 0} proteins")
        else:
            # If feature_types not available, try to infer by looking at var_names
            # Commonly antibodies have prefixes like 'ADT-', 'AB-', or 'CD'
            potential_proteins = adata_rna.var_names.str.startswith(('ADT-', 'AB-'))
            if np.any(potential_proteins):
                adata_genes = adata_rna[:, ~potential_proteins].copy()
                adata_protein = adata_rna[:, potential_proteins].copy()
                print(f"Inferred {lane}: {adata_genes.n_vars} genes and {adata_protein.n_vars} proteins")
            else:
                # If we can't detect proteins, assume all are genes
                adata_genes = adata_rna.copy()
                adata_protein = None
                print(f"Could not detect proteins for {lane}, treating all {adata_genes.n_vars} features as genes")
        
        # Add batch and lane information
        adata_genes.obs['batch'] = batch
        adata_genes.obs['lane'] = lane
        
        if adata_protein is not None:
            adata_protein.obs['batch'] = batch
            adata_protein.obs['lane'] = lane
            adatas_protein.append(adata_protein)
        
        # Add to list of datasets
        adatas.append(adata_genes)

# Step 2: Concatenate all datasets
print("Concatenating RNA datasets...")
adata_combined = ad.concat(adatas, merge="same")

# If we have protein data, concatenate it too
if adatas_protein:
    print("Concatenating protein datasets...")
    adata_protein_combined = ad.concat(adatas_protein, merge="same")
    print(f"Combined protein data: {adata_protein_combined.shape}")

# Annotate with sample information using hashtagged information
print("Matching cells with donor information...")

# Extract hashtag information - look for cell barcodes from adata_combined
# and match with sample_info based on batch and hash
# First, identify which columns in our data might contain hash information

# Let's check if we have protein data that might contain hashtag (HTO) information
has_hto = False
hto_proteins = []

if adatas_protein:
    # Look for proteins with HTO or hashtag in the name
    hto_proteins = [name for name in adata_protein_combined.var_names 
                   if 'HTO' in name or 'hashtag' in name.lower()]
    
    if hto_proteins:
        has_hto = True
        print(f"Found HTO proteins: {hto_proteins}")
    else:
        print("No HTO proteins explicitly identified")

# Initialize donor info columns
adata_combined.obs['donor_id'] = "Unknown"
adata_combined.obs['vr_time'] = np.nan
adata_combined.obs['vr_group'] = "Unknown"
adata_combined.obs['hash'] = 0

if has_hto:
    # Process HTO information
    print("Processing HTO information to assign donor IDs...")
    
    # Create temporary object with just HTOs for processing
    adata_hto = adata_protein_combined[:, hto_proteins].copy()
    
    # Normalize HTO counts (CLR transformation is common for HTO data)
    sc.pp.normalize_total(adata_hto)
    sc.pp.log1p(adata_hto)
    
    # Identify which HTO is highest for each cell
    hto_assignments = np.argmax(adata_hto.X, axis=1) + 1  # +1 because Hash IDs typically start at 1
    adata_combined.obs['hash'] = hto_assignments
    
    # Match cells with donor information based on batch and hash
    for idx, row in adata_combined.obs.iterrows():
        batch = row['batch']
        hash_num = row['hash']
        
        # Find matching donor in sample info
        match = sample_info[(sample_info['Batch'] == batch) & (sample_info['Hash'] == hash_num)]
        if not match.empty:
            adata_combined.obs.at[idx, 'donor_id'] = match.iloc[0]['Donor_ID']
            adata_combined.obs.at[idx, 'vr_time'] = match.iloc[0]['VR_time']
            adata_combined.obs.at[idx, 'vr_group'] = match.iloc[0]['VR_Group']
else:
    print("No HTO information available for donor assignment.")
    print("Will need to rely on other methods (e.g., lane information) to identify samples.")
    
    # Try to assign donor info based on lane if HTO data not available
    for batch, lanes in batch_files.items():
        lane_hash_mapping = {}
        for i, lane in enumerate(lanes):
            # Get hash assignments for this lane from sample_info
            lane_hashes = sample_info[sample_info['Batch'] == batch]['Hash'].unique()
            if len(lane_hashes) >= i+1:
                lane_hash_mapping[lane] = lane_hashes[i]
            else:
                lane_hash_mapping[lane] = i+1  # Fallback if not enough hash info
        
        # Now assign donor info based on lane
        for lane, hash_num in lane_hash_mapping.items():
            mask = adata_combined.obs['lane'] == lane
            adata_combined.obs.loc[mask, 'hash'] = hash_num
            
            # Find matching donor in sample info
            match = sample_info[(sample_info['Batch'] == batch) & (sample_info['Hash'] == hash_num)]
            if not match.empty:
                adata_combined.obs.loc[mask, 'donor_id'] = match.iloc[0]['Donor_ID']
                adata_combined.obs.loc[mask, 'vr_time'] = match.iloc[0]['VR_time']
                adata_combined.obs.loc[mask, 'vr_group'] = match.iloc[0]['VR_Group']

# QC filtering
print("Performing QC filtering...")
sc.pp.filter_cells(adata_combined, min_genes=200)
sc.pp.filter_genes(adata_combined, min_cells=3)

# Calculate QC metrics
mito_genes = adata_combined.var_names.str.startswith('MT-')
adata_combined.obs['percent_mito'] = np.sum(adata_combined[:, mito_genes].X, axis=1).A1 / np.sum(adata_combined.X, axis=1).A1 * 100 if sparse.issparse(adata_combined.X) else np.sum(adata_combined[:, mito_genes].X, axis=1) / np.sum(adata_combined.X, axis=1) * 100

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

# Batch correction if needed 
print("Performing batch correction...")
try:
    import harmonypy
    sc.external.pp.harmony_integrate(adata_combined, 'batch')
    use_rep = 'X_pca_harmony'
except ImportError:
    print("harmonypy not installed. Proceeding without harmony batch correction.")
    use_rep = 'X_pca'

# Compute neighbors
print("Computing neighbors...")
sc.pp.neighbors(adata_combined, n_neighbors=10, n_pcs=30, use_rep=use_rep)

# Run UMAP
print("Running UMAP...")
sc.tl.umap(adata_combined)

# Leiden clustering
print("Performing clustering...")
sc.tl.leiden(adata_combined, resolution=0.8)

# Step 3: Add protein information for better cell type annotation
if adatas_protein:
    print("Adding protein information for improved cell type annotation...")
    
    # We'll create a new combined object with both RNA and protein information
    # Get cells that are in both RNA and protein datasets
    common_cells = list(set(adata_combined.obs_names).intersection(set(adata_protein_combined.obs_names)))
    
    if common_cells:
        # Subset both objects to common cells
        adata_rna_common = adata_combined[common_cells].copy()
        adata_protein_common = adata_protein_combined[common_cells].copy()
        
        # Normalize protein data
        sc.pp.normalize_total(adata_protein_common)
        sc.pp.log1p(adata_protein_common)
        
        # Concatenate protein data as a separate layer in RNA object
        adata_rna_common.obsm['protein'] = adata_protein_common.X
        
        # Store protein names
        adata_rna_common.uns['protein_names'] = list(adata_protein_common.var_names)
        
        # Set protein annotations in current object
        adata_combined = adata_rna_common
        print(f"Combined object with RNA and protein data: {adata_combined.shape}")
    else:
        print("No common cells found between RNA and protein data.")

# Define marker genes and proteins for cell type annotation
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

# If we have protein data, define protein markers as well
marker_proteins = {}
if adatas_protein:
    protein_names = list(adata_protein_combined.var_names)
    print("Available proteins:", protein_names)
    
    # Map proteins to cell types (adjust based on your actual protein panel)
    # Check if specific protein markers exist in your data
    cd8_proteins = [p for p in protein_names if 'CD8' in p]
    cd4_proteins = [p for p in protein_names if 'CD4' in p]
    cd3_proteins = [p for p in protein_names if 'CD3' in p]
    
    if cd8_proteins or cd4_proteins or cd3_proteins:
        print("Found protein markers for T cell subsets")
        if cd3_proteins:
            marker_proteins['T cell'] = cd3_proteins
        if cd4_proteins:
            marker_proteins['CD4 T cell'] = cd4_proteins
        if cd8_proteins:
            marker_proteins['CD8 T cell'] = cd8_proteins
    
    # Look for other cell type markers
    for protein in protein_names:
        if 'CD19' in protein or 'CD20' in protein:
            marker_proteins.setdefault('B cell', []).append(protein)
        elif 'CD14' in protein or 'CD16' in protein:
            marker_proteins.setdefault('Monocyte', []).append(protein)
        elif 'CD56' in protein or 'CD57' in protein:
            marker_proteins.setdefault('NK cell', []).append(protein)
        elif 'CD11c' in protein or 'HLA-DR' in protein:
            marker_proteins.setdefault('Dendritic cell', []).append(protein)

# Calculate marker gene expression for cell type annotation
print("Calculating marker gene expression...")
sc.tl.rank_genes_groups(adata_combined, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata_combined, n_genes=25, sharey=False)

# Function to calculate mean expression of markers
def calculate_mean_expression(adata, cluster, genes, use_RNA=True):
    cells = adata[adata.obs['leiden'] == cluster]
    gene_indices = []
    
    if use_RNA:
        # For RNA markers
        gene_indices = [adata.var_names.get_loc(gene) for gene in genes if gene in adata.var_names]
        if len(gene_indices) == 0:
            return 0
        
        # Get expression values
        if sparse.issparse(cells.X):
            return np.mean([np.mean(cells.X[:, idx].A) for idx in gene_indices])
        else:
            return np.mean([np.mean(cells.X[:, idx]) for idx in gene_indices])
    else:
        # For protein markers
        if 'protein_names' not in adata.uns or 'protein' not in adata.obsm:
            return 0
            
        protein_indices = [adata.uns['protein_names'].index(protein) 
                          for protein in genes 
                          if protein in adata.uns['protein_names']]
        
        if len(protein_indices) == 0:
            return 0
            
        # Get protein expression values
        if sparse.issparse(adata.obsm['protein']):
            return np.mean([np.mean(adata.obsm['protein'][:, idx].A) for idx in protein_indices])
        else:
            return np.mean([np.mean(adata.obsm['protein'][:, idx]) for idx in protein_indices])

# Automatically annotate clusters using both RNA and protein markers
cluster_annotations = {}
for cluster in adata_combined.obs['leiden'].unique():
    scores = {}
    
    # Score using RNA markers
    for cell_type, genes in marker_genes.items():
        rna_score = calculate_mean_expression(adata_combined, cluster, genes, use_RNA=True)
        scores[cell_type] = rna_score
    
    # Add protein scores if available
    if marker_proteins and 'protein' in adata_combined.obsm:
        for cell_type, proteins in marker_proteins.items():
            protein_score = calculate_mean_expression(adata_combined, cluster, proteins, use_RNA=False)
            if cell_type in scores:
                # Combine RNA and protein scores (you can adjust weighting)
                scores[cell_type] = scores[cell_type] * 0.7 + protein_score * 0.3
            else:
                scores[cell_type] = protein_score
    
    # Assign cell type based on highest score
    if scores:
        cluster_annotations[cluster] = max(scores, key=scores.get)
    else:
        cluster_annotations[cluster] = "Unknown"

# Add cell type annotations to the object
adata_combined.obs['cell_type'] = adata_combined.obs['leiden'].map(cluster_annotations)

# Visualize cell type annotations on UMAP
sc.pl.umap(adata_combined, color=['cell_type', 'leiden'], save=os.path.join(output_dir, 'umap_celltypes.png'))

# Visualize CD8 marker expression (RNA)
cd8_markers = ['CD8A', 'CD8B']
existing_cd8_markers = [m for m in cd8_markers if m in adata_combined.var_names]
if existing_cd8_markers:
    sc.pl.umap(adata_combined, color=existing_cd8_markers, save=os.path.join(output_dir, 'umap_cd8_markers_rna.png'))

# Visualize CD8 marker expression (protein)
if 'protein_names' in adata_combined.uns:
    cd8_protein_markers = [p for p in adata_combined.uns['protein_names'] if 'CD8' in p]
    if cd8_protein_markers:
        # Create a function to plot protein expression on UMAP
        def plot_protein_on_umap(adata, protein_name, save_path=None):
            if protein_name not in adata.uns['protein_names']:
                print(f"Protein {protein_name} not found")
                return
                
            protein_idx = adata.uns['protein_names'].index(protein_name)
            
            # Extract protein expression
            if sparse.issparse(adata.obsm['protein']):
                protein_expr = adata.obsm['protein'][:, protein_idx].A.flatten()
            else:
                protein_expr = adata.obsm['protein'][:, protein_idx].flatten()
            
            # Create temporary observation with protein expression
            adata.obs[f'protein_{protein_name}'] = protein_expr
            
            # Plot on UMAP
            fig = sc.pl.umap(adata, color=f'protein_{protein_name}', 
                         title=f'Protein: {protein_name}', show=False)
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            
            # Remove temporary column
            del adata.obs[f'protein_{protein_name}']
        
        # Plot each CD8 protein marker
        for i, protein in enumerate(cd8_protein_markers):
            plot_protein_on_umap(adata_combined, protein, 
                              save_path=os.path.join(output_dir, f'umap_protein_{protein.replace("/", "_")}.png'))

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

# Alternative to create loom files entirely in Python (if velocyto command fails)
# This is a simplified example; actual implementation would need spliced/unspliced counts
print("If velocyto command fails, here's an alternative to create loom files in Python:")
print("""
# For each batch/lane:
from anndata import AnnData
import scanpy as sc
import scvelo as scv
import loompy

# If you have access to BAM files, you can use velocyto Python API
# (not shown here as it's complex and requires BAM parsing)

# Or if you already have spliced/unspliced counts:
def create_loom_from_counts(spliced_counts, unspliced_counts, cell_barcodes, gene_ids, gene_names, output_file):
    # Create AnnData object with spliced/unspliced counts
    adata = AnnData(X=spliced_counts)
    adata.layers['spliced'] = spliced_counts
    adata.layers['unspliced'] = unspliced_counts
    
    # Add cell and gene metadata
    adata.obs_names = cell_barcodes
    adata.var['gene_ids'] = gene_ids
    adata.var_names = gene_names
    
    # Save as loom file
    adata.write_loom(output_file)
    return output_file
""")

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
    
    # Further analyze CD8 T cell subpopulations
    print("Analyzing CD8 T cell subpopulations...")
    
    # Perform subclustering on CD8 T cells
    sc.tl.leiden(valid_vr, resolution=1.0, key_added='cd8_subcluster')
    
    # Plot subclusters
    scv.pl.velocity_embedding_stream(
        valid_vr, basis='umap', color='cd8_subcluster',
        save=os.path.join(output_dir, 'velocity_stream_cd8_subclusters.png')
    )
    
    # Define CD8 T cell subpopulation markers
    cd8_subpop_markers = {
        'Naive': ['CCR7', 'SELL', 'TCF7', 'LEF1'],
        'Central Memory': ['CCR7', 'SELL', 'IL7R'],
        'Effector Memory': ['