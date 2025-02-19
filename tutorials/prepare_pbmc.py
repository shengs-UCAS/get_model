from pathlib import Path

import numpy as np
import snapatac2 as snap
from gcell._settings import get_setting
from gcell._settings import update_settings

import pandas as pd
from pyranges import PyRanges as pr



SETTINGS = {
    "annotation_dir":  "/project-whj/pms_lls/get_model/tutorials/annotations",
    "genome_dir": "/project-whj/pms_lls/get_model/tutorials/genomes",
    "cache_dir": "/project-whj/pms_lls/get_model/tutorials/cache",
}
update_settings(SETTINGS)

annotation_dir = get_setting('annotation_dir')
print("gcell currently using annotation directory:", annotation_dir)

cre = pd.read_csv('../data/cCRE_hg38.tsv.gz', sep='\t')
cre = cre.rename(columns={'#Chromosome': 'Chromosome', 'hg38_Start': 'Start', 'hg38_End': 'End'})[['Chromosome', 'Start', 'End']]
cre = pr(cre, int64=True).sort()

import scanpy as sc

ad = sc.read_10x_h5('../data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5', gex_only=False)
ad

peaks = ad.var.query('feature_types == "Peaks"').interval.str.split(':|-').tolist()
peaks = pd.DataFrame(peaks, columns=['Chromosome', 'Start', 'End'])
peaks['Start'] = peaks['Start'].astype(int)
peaks['End'] = peaks['End'].astype(int)
peaks = pr(peaks, int64=True).sort()
peaks.df.shape

# peak / cre + cre 
# find all peaks that don't overlap with cre; -> non_overlap_peaks
non_overlap_peaks = peaks.overlap(cre, invert=True)
# concat with cre and sort
total_peaks = pd.concat([non_overlap_peaks.df, cre.df], ignore_index=True)
total_peaks = pr(total_peaks, int64=True).sort()
# remove chromosome M, Y and those not start with chr
total_peaks = total_peaks.df.query('Chromosome.str.startswith("chr") & ~Chromosome.str.endswith("M") & ~Chromosome.str.endswith("Y") & ~Chromosome.str.startswith("chrUn")')
total_peaks.shape

if not Path('rna.h5ad').exists():
    rna = snap.read(snap.datasets.pbmc10k_multiome(modality='RNA'), backed=None)
    sc.pp.highly_variable_genes(rna, flavor='seurat_v3', n_top_genes=3000)
    rna_filtered = rna[:, rna.var.highly_variable]
    sc.pp.normalize_total(rna_filtered, target_sum=1e4)
    sc.pp.log1p(rna_filtered)
    snap.tl.spectral(rna_filtered, features=None)
    snap.tl.umap(rna_filtered)
    rna_filtered.write('rna.h5ad')
else:
    rna_filtered = sc.read('rna.h5ad')

ad = ad[ad.obs.index.isin(rna_filtered.obs.index.values)]
barcode_to_celltype = rna_filtered.obs.to_dict()['cell_type']
ad.obs['cell_type'] = ad.obs.index.map(barcode_to_celltype)
ad.obs.head()

ad_rna = ad[:, np.where(ad.var.feature_types == 'Gene Expression')[0]]
ad_rna

sc.pp.highly_variable_genes(ad_rna, flavor='seurat_v3', n_top_genes=3000)
ad_rna_filtered = ad_rna[:, ad_rna.var.highly_variable]
sc.pp.normalize_total(ad_rna_filtered, target_sum=1e4)
sc.pp.log1p(ad_rna_filtered)
snap.tl.spectral(ad_rna_filtered, features=None)
snap.tl.umap(ad_rna_filtered)
sc.pl.umap(ad_rna_filtered, color='cell_type')

ad_atac = ad[:, np.where(ad.var.feature_types == 'Peaks')[0]]
ad_atac

cell_number = ad_atac.obs.groupby('cell_type', observed=False).size().to_dict()
print("The following cell types have more than 100 cells and library size > 3M, adding them to celltype_for_modeling")
celltype_for_modeling = []
for cell_type in cell_number:
    if cell_number[cell_type] > 100:
        celltype_for_modeling.append(cell_type)
        libsize = int(ad_atac[ad_atac.obs.cell_type == cell_type].X.sum())
        if libsize > 3000000:
            print(f"{cell_type} number of cells: {cell_number[cell_type]}, library size: {libsize}")

import pandas as pd
from pyranges import PyRanges as pr
def get_peak_from_snapatac(atac: snap.AnnData):
    """
    Get the peak names from the snapatac object.

    Args:
        atac: snapatac2 processed AnnData object

    Returns:
        peak_names: pandas DatasFrame with the peak names
    """
    peak_names = pd.DataFrame(atac.var.index.str.split('[:-]').tolist(), columns=['Chromosome', 'Start', 'End'])
    peak_names['Start'] = peak_names['Start'].astype(int)
    peak_names['End'] = peak_names['End'].astype(int)
    return peak_names

peaks = get_peak_from_snapatac(ad_atac)
peaks.shape


def get_peak_acpm_for_cell_type(atac: snap.AnnData, cell_type: str):
    """
    Get the peak acpm for a given cell type.
    """
    peaks = get_peak_from_snapatac(atac)
    counts = np.array(atac[atac.obs.cell_type == cell_type].X.sum(0)).flatten()
    acpm = np.log10(counts / counts.sum() * 1e5 + 1)
    peaks['aCPM'] = acpm/acpm.max()
    peaks = peaks.query('Chromosome.str.startswith("chr") & ~Chromosome.str.endswith("M") & ~Chromosome.str.endswith("Y") & ~Chromosome.str.startswith("chrUn")')
    peaks = pr(peaks, int64=True).sort().df
    return peaks


for cell_type in celltype_for_modeling:
    peaks = get_peak_acpm_for_cell_type(ad_atac, cell_type)
    peaks.to_csv(f'{cell_type.replace(" ", "_").lower()}.atac.bed', sep='\t', index=False, header=False)

peaks

def get_rna_for_cell_type(rna: snap.AnnData, cell_type: str):
    """
    Get the rna for a given cell type.
    """
    counts = rna[rna.obs.cell_type == cell_type].X.sum(0)
    counts = np.log10(counts / counts.sum() * 1e6 + 1)
    counts = np.array(counts).flatten()
    rna_tpm = pd.DataFrame(counts, columns=['TPM'])
    rna_tpm['gene_name'] = rna.var.index
    return rna_tpm[['gene_name', 'TPM']].sort_values(by='gene_name', ascending=True)


for cell_type in celltype_for_modeling:
    rna_tpm = get_rna_for_cell_type(ad_rna, cell_type)
    rna_tpm.to_csv(f'{cell_type.replace(" ", "_").lower()}.rna.csv', index=False)


# NOTE: tabix has to be >= 1.17

import os
from pathlib import Path

from gcell._settings import get_setting
from preprocess_utils import (
    add_atpm,
    add_exp,
    create_peak_motif,
    download_motif,
    get_motif,
    query_motif,
)

motif_bed_url = "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz"
motif_bed_index_url = "https://resources.altius.org/~jvierstra/projects/motif-clustering/releases/v1.0/hg38.archetype_motifs.v1.0.bed.gz.tbi"


if (
    motif_bed_url
    and motif_bed_index_url
    and not (
        (annotation_dir / "hg38.archetype_motifs.v1.0.bed.gz").exists()
        or (annotation_dir / "hg38.archetype_motifs.v1.0.bed.gz.tbi").exists()
    )
):
    download_motif(motif_bed_url, motif_bed_index_url, motif_dir=annotation_dir)
    motif_bed = str(annotation_dir / "hg38.archetype_motifs.v1.0.bed.gz")
else:
    motif_bed = str(annotation_dir / "hg38.archetype_motifs.v1.0.bed.gz")

peak_bed = "cd4_naive.atac.bed" # since all cell types share the same peak set, when querying motifs, we can just use one cell type to query motifs.
peaks_motif = query_motif(peak_bed, motif_bed)
get_motif_output = get_motif(peak_bed, peaks_motif)

create_peak_motif(get_motif_output, "pbmc10k_multiome.zarr", peak_bed)
celltype_for_modeling = [
    'cd14_mono',
    'cd16_mono',
    'cd4_naive',
    'cd4_tcm',
    'cd4_tem',
    'cd8_naive',
    'cd8_tem_1',
    'cd8_tem_2',
    'intermediate_b',
    'mait',
    'memory_b',
    'naive_b',
    'treg',
    'cdc',
    'gdt',
    'nk',
 ]

for cell_type in celltype_for_modeling:
    add_atpm(
        "pbmc10k_multiome.zarr",
        f"{cell_type}.atac.bed",
        cell_type,
    )

for cell_type in celltype_for_modeling:
    add_exp(
        "pbmc10k_multiome.zarr",
        f"{cell_type}.rna.csv",
        f"{cell_type}.atac.bed",
        cell_type,
        assembly="hg38",
        version=44,
        extend_bp=300, # extend TSS region to 300bp upstream and downstream when overlapping with peaks
    id_or_name="gene_name", # use gene_name or gene_id to match the gene expression data, checkout your rna.csv file column names, should be either [gene_name, TPM] or [gene_id, TPM]
)

for file in [peaks_motif, get_motif_output]:
    os.remove(file)
