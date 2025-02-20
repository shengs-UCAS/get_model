#!/usr/bin/env python
# coding: utf-8

#  # Finetune a GET Model on PBMC 10k Multiome
# 
# 
#  This tutorial demonstrates how to train a GET model to predict expression in ATAC-seq peaks using motif information. We'll cover:
# 
#  1. Loading and configuring the model
# 
#  2. Finetune from a pretrained expression prediction GET model
# 
#  3. Perform various analysis using `gcell` package
# 
# 
# 
#  ## Setup
# 
#  First, let's import the necessary modules and set up our configuration.
#  
#  Note:
#  If you run from a Mac, make sure you use the jupyter notebook rather than the VSCode interactive python editor as the later seems to have issue with multiple workers.
#  If you run from Linux, both should work fine.

# In[1]:


from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from gcell.utils.causal_lib import get_subnet, plot_comm, preprocess_net

from get_model.config.config import load_config
from get_model.run_region import run_zarr as run


#  ## Finetune
# 
# 
# 
#  We'll start by loading a predefined configuration and customizing it for our needs.
# 
#  The base configuration is in `get_model/config/finetune_tutorial_pbmc.yaml`

# In[2]:


if not Path('./checkpoint-best.pth').exists():
    s3_checkpoint_url = "s3://2023-get-xf2217/get_demo/checkpoints/regulatory_inference_checkpoint_fetal_adult/finetune_fetal_adult_leaveout_astrocyte/checkpoint-best.pth"
    get_ipython().system(' aws s3 cp $s3_checkpoint_url ./checkpoint-best.pth --no-sign-request')


# > Note: In the paper, we mainly used binary ATAC signal trained model for motif interpretation analysis. As it's hard to say whether there are mutual causal relationship between transcription and accessibility. If accessibility is added to the model, potentially it will absorb some TF's effect to itself, thereby making the interpretation more difficult. However, if the goal is to represent the cell states as precisely as possible and use the model for other downstream tasks (e.g. enhancer target prediction), adding the accessibility signal is probably better.

# In[3]:


celltype_for_modeling = [
    'memory_b',
    'cd14_mono',
    'gdt',
    'cd8_tem_1',
    'naive_b',
    'mait',
    'intermediate_b',
    'cd4_naive',
    'cd8_tem_2',
    'cd8_naive',
    'cd4_tem',
    'cd4_tcm',
    'cd16_mono',
    'nk',
    'cdc',
    'treg']
cfg = load_config('finetune_tutorial_pbmc') # load the predefined finetune tutorial config
cfg.stage = 'fit'
cfg.run.project_name = 'finetune_pbmc10k_multiome'
cfg.run.run_name = 'training_from_finetune_lora_cd4_tcm_no_chr_split_binary_atac'
cfg.dataset.quantitative_atac = False # We use binary ATAC signal for motif interpretation analysis
cfg.dataset.zarr_path = "./pbmc10k_multiome.zarr" # set the path to the data zarr file we just processed
cfg.dataset.celltypes = ','.join(celltype_for_modeling) # the celltypes you want to finetune
cfg.finetune.checkpoint = "./checkpoint-best.pth" # set the path to the pretrained checkpoint we want to finetune from
cfg.dataset.leave_out_celltypes = 'cd4_tcm' # set the celltypes you want to leave out
cfg.machine.num_devices=1 # use 0 for cpu training; >=1 for gpu training
cfg.machine.batch_size=8 # batch size for training; check `nvidia-smi` to see the available GPU memory
# cfg.machine.output_dir = "./output/"
print(f"output path: {cfg.machine.output_dir}/{cfg.run.project_name}/{cfg.run.run_name}")
print(f"training for {cfg.training.epochs} epochs")


# In[4]:


# export the config to a yaml file
from get_model.config.config import export_config, load_config_from_yaml

export_config(cfg, "exported_training_config.yaml")


# In[5]:


# load the config from the yaml file
cfg = load_config_from_yaml("exported_training_config.yaml")


# In[6]:


print(f"Default checkpoint path is at: {cfg.machine.output_dir}/{cfg.run.project_name}/{cfg.run.run_name}/checkpoints/best.ckpt")
print("The `trainer.checkpoint_callback.best_model_path` variable will be updated to the checkpoint path after training")


# Now we can start the finetuning

# In[7]:


trainer = run(cfg) # run the finetuning, takes around 2 hours on one RTX 3090
print("checkpoint path:", trainer.checkpoint_callback.best_model_path)


# # Interpretation
# 
# After finetuning, we can use the checkpoint to predict expression of all accessible genes and generate jacobian matrix of (peak x motif) for every predicted genes. 
# To start, we need to collect the checkpoint we produced and switch to `predict` stage. Here, let's focus on CD4 Naive cell and we need to set `cfg.leave_out_celltypes` to `cd4_naive` for the model to predict gene expression in this cell type.

# In[3]:


from get_model.config.config import export_config, load_config_from_yaml
from get_model.run_region import run_zarr as run

cfg = load_config_from_yaml("exported_training_config.yaml")
use_checkpoint = f'{cfg.machine.output_dir}/{cfg.run.project_name}/{cfg.run.run_name}/checkpoints/best.ckpt' # or use trainer.checkpoint_callback.best_model_path if you are still in the same session
# note that if you run the training stage multiple times, you can specify the checkpoint you want to use here e.g. best-v1.ckpt, best-v2.ckpt, etc.
cfg.stage = 'predict'
cfg.finetune.resume_ckpt = use_checkpoint
# in predict stage, leave out the celltype you want to interpret
cfg.run.use_wandb=False # disable wandb logging when predicting
cfg.task.layer_names = [] # set to empty list to disable intermediate layer interpretation
cfg.task.gene_list = None # set to None to predict all genes; otherwise you can specify a list of genes as 'MYC,SOX10,SOX2,RET', only genes with promoter open will be used
# loop through all celltypes and run the predict stage
cfg.run.run_name='interpret_training_from_finetune_lora_cd4_tcm_no_chr_split_binary_atac'
# export interpretation config to yaml file
export_config(cfg, "exported_interpretation_config.yaml")
for celltype in ['cd4_naive']:
    cfg.dataset.leave_out_celltypes = celltype
    trainer = run(cfg)





# As you can see, the results is now saved to `finetune_pbmc10k_multiome/interpret_training_from_finetune_lora_cd4_tcm_no_chr_split/cd4_naive.zarr`. Now we can use the `GETHydraCellType` class from `gcell` to load it.
# 
# ## Load interpretation result as `GETHydraCellType`

# In[89]:


# get the celltype object based on the configuration
from gcell.cell.celltype import GETHydraCellType

from get_model.config.config import load_config

cfg = load_config("finetune_tutorial_pbmc.yaml")
celltype = 'cd4_naive' # the celltype you want to load, if you have ran the predict stage multiple times on different celltypes, you can specify the celltype you want to load here
cfg.run.run_name='interpret_training_from_finetune_lora_cd4_tcm_no_chr_split_binary_atac' # the run name you want to load, should be the same as the one you used in the predict stage
hydra_celltype = GETHydraCellType.from_config(cfg, celltype=celltype) # if you ran the predict stage multiple times on different celltypes, you can specify the celltype you want to load here
cd14_mono = GETHydraCellType.from_config(cfg, celltype='cd14_mono')


# We can plot the predicted and observed expression to see whether there is any issue.

# In[3]:


# set dpi to 100 
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.dpi'] = 100
fig, ax = plt.subplots(figsize=(4, 4))
sns.scatterplot(data = hydra_celltype.gene_annot[['obs', 'pred', 'gene_name']].groupby('gene_name').max(), x='obs', y = 'pred', s=1, alpha=0.5, ax=ax)
ax.set_title('CD4 Naive')
ax.set_xlabel('Observed Expression')
ax.set_ylabel('Predicted Expression')
ax.set_aspect('equal')
plt.show()


# ## Save Gene-by-Motif importance matrix and Causal discovery result
# 
# The summarized gene-by-motif importance matrix can be accessed through `hydra_celltype.gene_by_motif` property. The following function will retrieve and save it to disk.

# In[90]:


# takes around 30 min on 16 cores, TODO should be able to run in parallel
hydra_celltype.get_gene_by_motif()
cd14_mono.get_gene_by_motif()


# After gene-by-motif matrix is saved, we can perform causal discovery between different motifs. This is done via `.get_causal()` function. The results will also be saved to disk. Here we will perform 3 independent run with column (motif) permutation.

# In[92]:


# takes around 30 min on 16 cores
hydra_celltype.gene_by_motif.get_causal()
cd14_mono.gene_by_motif.get_causal()


# # Case study on CD4 Naive cells
# Let's have a look at CD4 Naive RNA-seq to find some interesting TF to study: it seems that LEF1 and BACH2 are good marker genes for 'CD4 Naive' (also CD8 Naive)

# In[6]:


# load the scanpy object
import scanpy as sc

rna_filtered = sc.read('rna.h5ad')
# get marker genes for 'CD4 Naive'
sc.tl.rank_genes_groups(rna_filtered, 'cell_type', method='wilcoxon', rankby_abs=False)
sc.pl.rank_genes_groups(rna_filtered, n_genes=20, sharey=False)


# Check which motif cluster LEF1 and BACH2 belongs to...

# In[7]:


hydra_celltype.motif.annotations.query('Motif.str.startswith("BACH")')


# In[8]:


hydra_celltype.motif.annotations.query('Motif.str.startswith("LEF")')


# ## Network Analysis
# 
# Let's use LEF1 and AP1/2 for subnetwork analysis

# In[15]:


from gcell.utils.causal_lib import get_subnet, plot_comm, preprocess_net

g = hydra_celltype.gene_by_motif.get_causal()
g_ = preprocess_net(g, 0.05)
plot_comm(get_subnet(g_.copy(), 'LEF1'), figsize=(5, 5))
# plot_comm(g_, figsize=(20, 20))


# In terms of AP1/2 (BACH2):

# In[14]:


g = hydra_celltype.gene_by_motif.get_causal()
g_ = preprocess_net(g, 0.05)
plot_comm(get_subnet(g_.copy(), 'AP1/2'), figsize=(5, 5))


# ## Protein-protine interaction checking
# 
# Note that BACH2-MAF is a known interaction

# In[13]:


from gcell.protein.stringdb import get_string_network

get_string_network(["BACH2"], species='human', network_type='physical', required_score=400)


# Also NCOR1 is a Nuclear receptor corepressor which binds to many nuclear receptors.

# In[30]:


from IPython.display import Image

get_string_network(["NCOR1"], species='human', network_type='physical', required_score=400, save_image='ncor1_string.png')
# show the image
Image(filename='ncor1_string.png')


# Where RARA belongs to NR/12 motif cluster and predicted to be co-regulating with BACH2.

# In[16]:


genes = hydra_celltype.motif.cluster_gene_list['NR/12']
hydra_celltype.gene_annot.query('gene_name.isin(@genes)')


# We can check the predicted target gene average expression. Here we use the raw counts normalized by library size to compute the average expression, and performed that for AP1/2 and LEF1. 

# In[ ]:


import numpy as np
import scanpy as sc
import snapatac2 as snap
rna_filtered = sc.read('rna.h5ad')

rna = snap.read(snap.datasets.pbmc10k_multiome(modality='RNA'), backed=None)


# In[219]:


import pandas as pd
from scipy.stats import linregress

def get_motif_jacob_residual(motif_name):
    cd4_naive_data = hydra_celltype.gene_by_motif.data[motif_name].copy()
    cd14_mono_data = cd14_mono.gene_by_motif.data[motif_name].copy()
    motif_jacob = pd.merge(cd4_naive_data.rename('cd4_naive'), cd14_mono_data.rename('cd14_mono'), left_index=True, right_index=True)
    motif_jacob.plot.scatter(x='cd14_mono', y='cd4_naive')
    # fit a linear regression model, get residuals
    res = linregress(motif_jacob['cd14_mono'], motif_jacob['cd4_naive'])
    print(res)
    y_ = res.slope*motif_jacob['cd14_mono']+res.intercept
    motif_jacob['cd4_naive_residual'] = motif_jacob['cd4_naive']-y_
    return motif_jacob['cd4_naive_residual']

lef1_residual = get_motif_jacob_residual('LEF1')
ap1_residual = get_motif_jacob_residual('AP1/2')


# In[261]:


sns.scatterplot(x=lef1_residual.rename('LEF1'), y=ap1_residual.rename('AP1/1'), s=2)
common_genes = []
for i in range(len(lef1_residual)):
    if lef1_residual.iloc[i] > lef1_residual.quantile(0.99) and ap1_residual.iloc[i] > ap1_residual.quantile(0.99):
        common_genes.append(lef1_residual.index[i])



# Interestingly, the top CD4 Naive marker gene, FHIT, is also predicted to be regulated by LEF1.

# Let's plot the average expression of predicted targets or non-targets of LEF1 and AP1/2:

# In[262]:


rna_filtered.obs['common_genes_exp'] = X[:, rna.var.index.isin(common_genes)].sum(axis=1)
sc.pl.umap(rna_filtered, color=['common_genes_exp'])


# In[263]:


'FHIT' in common_genes


# It seems FHIT is among one of the potentially commonly regulated genes by AP1/2 and LEF1. We can have a more detailed look into FHIT, the top marker gene for CD4 Naive:

# In[83]:


hydra_celltype.get_gene_jacobian_summary('FHIT').sort_values().tail(20)


# In[205]:


x = hydra_celltype.get_gene_jacobian('FHIT')[0].data.iloc[100, 5:]
y= cd14_mono.get_gene_jacobian('FHIT')[0].data.iloc[100, 5:]
sns.scatterplot(x=x, y=y)


# Many of these are

# In[ ]:


import pandas as pd 

sns.lmplot(data = pd.concat([cd14_mono.get_gene_jacobian_summary('FHIT').rename('cd14_mono'), 
                             hydra_celltype.get_gene_jacobian_summary('FHIT').rename('cd4_naive')], 
                             axis=1), x='cd14_mono', y='cd4_naive')


# In[ ]:


# fit a linear regression model, get residuals
from scipy.stats import linregress
x= cd14_mono.get_gene_jacobian_summary('FHIT')
y= hydra_celltype.get_gene_jacobian_summary('FHIT')
res = linregress(x, y)
print(res)
y_ = res.slope*x+res.intercept
# get the residuals relative to the y_
cd4_naive_residual = y-y_
(cd4_naive_residual/y_).sort_values().tail(20)


# In[208]:


genes = hydra_celltype.motif.cluster_gene_list['ZNF784']
hydra_celltype.gene_annot.query('gene_name.isin(@genes)')[0:6]


# In[211]:


rna_filtered.obs['FOXO1_exp'] = X[:, rna.var.index.isin(['FOXO1'])].mean(axis=1)
rna_filtered.obs['NRF1_exp'] = X[:, rna.var.index.isin(['NRF1'])].mean(axis=1)
rna_filtered.obs['LEF1_exp'] = X[:, rna.var.index.isin(['LEF1'])].mean(axis=1)
rna_filtered.obs['FHIT_exp'] = X[:, rna.var.index.isin(['FHIT'])].mean(axis=1)
sc.pl.umap(rna_filtered, color=['FOXO1_exp', 'NRF1_exp', 'LEF1_exp', 'FHIT_exp'], vmax=3)


# In[163]:


hydra_celltype.focus=100
hydra_celltype.plot_gene_regions('FHIT', plotly=True)


# Alternatively, we can generate a bigwig file for visualize in `genomespy` interactively. Note that

# In[65]:


get_ipython().system(' wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/bedGraphToBigWig')
get_ipython().system(' chmod +x bedGraphToBigWig')


# In[29]:


from gcell._settings import get_setting

str((Path(get_setting('annotation_dir')) / 'hg38.chrom.sizes'))


# In[75]:


import pandas as pd

df = hydra_celltype.get_gene_jacobian_summary('FHIT', "region")
js = hydra_celltype.get_gene_jacobian('FHIT')
df["End"] = hydra_celltype.peak_annot.iloc[df["index"].values].End.values
df = df[["index", "Chromosome", "Start", "End", "Score"]]
r_motif = (
    pd.concat([j.data for j in js], axis=0)
    .drop(["Chromosome", "Start", "End", "Gene"], axis=1, errors="ignore")
    .groupby("index")
    .mean()
    .abs()
)
df['Chromosome'] = df['Chromosome'].apply(lambda x: x.strip(' '))
df = df.merge(r_motif, left_on="index", right_index=True)
df[['Chromosome', 'Start', 'End', 'Score']].to_csv('FHIT_Score.bedgraph', sep='\t', index=False, header=False)
df[['Chromosome', 'Start', 'End', 'LEF1']].to_csv('FHIT_LEF1.bedgraph', sep='\t', index=False, header=False)
df[['Chromosome', 'Start', 'End', 'AP1/2']].to_csv('FHIT_AP1.bedgraph', sep='\t', index=False, header=False)


# In[76]:


get_ipython().system(" ./bedGraphToBigWig FHIT_Score.bedgraph {str(Path(get_setting('annotation_dir')) / 'hg38.chrom.sizes')} FHIT_Score.bw")
get_ipython().system(" ./bedGraphToBigWig FHIT_LEF1.bedgraph {str(Path(get_setting('annotation_dir')) / 'hg38.chrom.sizes')} FHIT_LEF1.bw")
get_ipython().system(" ./bedGraphToBigWig FHIT_AP1.bedgraph {str(Path(get_setting('annotation_dir')) / 'hg38.chrom.sizes')} FHIT_AP1.bw")


# In[77]:


chrom = df.Chromosome.values[0]
start = str(df.Start.min())
end = str(df.End.max())
print(f"{chrom}:{start}-{end}")


# In[79]:


from genomespy import igv

tracks = {
    "Region Score": {
        "url": "FHIT_Score.bw",
        "height": 40,
        "type": "bigwig"
    },
    "LEF1": {
        "url": "FHIT_LEF1.bw",
        "height": 40,
        "type": "bigwig"
    },
    "AP1/2": {
        "url": "FHIT_AP1.bw",
        "height": 40,
        "type": "bigwig"
    },
}
plot = igv(
    tracks,
    region={"chrom": chrom, "start": start, "end": end},
    server_port=18093
)
plot.show()


# In[78]:


plot.close()


# `genomespy` only works in live session, so here is a screenshot as a demo.

# In[80]:


# show genomespy_output.png
Image(filename='genomespy_output.png')


# ##
# We can use tools in `gcell` to get the exact motif matches in the promoter, or any other region of interest.

# In[97]:


from gcell.dna.genome import Genome, GenomicRegionCollection

hg38 = Genome('hg38') # load the reference genome
df_regions = GenomicRegionCollection(hg38, df.iloc[100:101]) # 100 is the index of the promoter region of FHIT
df_seq = df_regions.collect_sequence() # collect the sequence of the promoter region. Note that here you can also specify a list of mutations. Checkout gcell.dna.sequence.DNASequenceCollection for more details.



# In[100]:


motif_cluster = hydra_celltype.motif # motif object is in the celltype object
df_motif = df_seq.scan_motif(motif_cluster, raw=True) # scan the motif on the sequence, raw=True to return every motif match
df_motif



# Filter to just `LEF1` and `AP1/2` cluster:

# In[101]:


df_motif.query('cluster.isin(["LEF1", "AP1/2"])')


# In[14]:


import scanpy as sc
import snapatac2 as snap

rna = snap.read(snap.datasets.pbmc10k_multiome(modality='RNA'), backed=None)
rna_filtered = sc.read('rna.h5ad')
X = sc.pp.normalize_total(rna, inplace=False)['X']
def plot_exp_on_umap(gene_list):
    for gene in gene_list:
        rna_filtered.obs[f'{gene}_exp'] = X[:, rna.var.index.isin([gene])].mean(axis=1)
    sc.pl.umap(rna_filtered, color=[f'{gene}_exp' for gene in gene_list], vmax=5)

plot_exp_on_umap(['FHIT', 'NFATC3', 'NFYB', 'ELF1', 'MXI1', 'SP4', 'ZBTB14', 'E2F8', "NFATC2", "NFATC1"])


# In[24]:


from genomespy import igv

tracks = {
    # data only available in GM12878
    "NFATC3_GM12878": {
        "url": "https://encode-public.s3.amazonaws.com/2020/12/08/8ea9c746-be19-4001-baca-cd48ece97db2/ENCFF179MUG.bigWig?response-content-disposition=attachment%3B%20filename%3DENCFF179MUG.bigWig&AWSAccessKeyId=ASIATGZNGCNX33A5X5WW&Signature=hHHJG06qHqFA0HJlnUgTgv7xKig%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEHgaCXVzLXdlc3QtMiJHMEUCIQDINzaeAc47zRgeeg2mWP3l4VeIdcQevf%2FT353fEKTZmgIgCdkG14fyPq4bcOTCcJjugQlZR8Qx6YKuzquEq83ezPIqswUIYBAAGgwyMjA3NDg3MTQ4NjMiDJS%2B303hofFkb8WxciqQBbao7JhBbche%2BONqPE7mQjroYd5rCYdEHesDQ4twba15t%2BpTy%2FLrQoZ2ca94oVnebnTB0HCFmoluN2OPphhqcCEynyqARAmtRrp4mPjTraOaysR7kEsyGgK%2Fq8S9EPNIai6KircO26NlYgc%2FFcZS%2BZb%2BKdc7foVC3r1lB1IZPfxp%2FzckgzPLZ98F%2F048ftT%2BuSVSr39aizvTwIwgfR%2BJBd3OAMwVdv4K7xIfArmsrS2Zzg00hopbnpNm5huabqZhE4dOpVYz5VqHfYP7BBU44Z0oemwkT7LD6QtWzoHYdXahIAYrmz%2B5siCjuSTBLJYqsjOKluqwcUZmp4x1AFefUb4jsUjLLcW4KlVuTQ7T%2B0jYludXJIA032EObv4qILf%2BjG7WO%2BJkNpzG%2FB9OZomjXiDMdd9e18XZrIzlKKgOHQbs8pmCH3MRMaP%2BlPr92%2Bl%2BKFQclQ7qqaPmLuWrgxHL3ipKkdhrWIbOV3GZ3e8F2ZOKygeUgG4hglufQoxICfRw9NB53S%2B5u5LfJZ6bYaNX%2FsmMS%2BMEo3aiISEy7yLbxxAAD2etOgeAOcyAk4tPOMs7xXo6Zh7%2FT2hKqDofnXi%2BgP%2BvO69b4NA061MX97scUmj%2BIIb0fG3lUNZaPDUDtodj5gS5qhW1X6leeqDJ5Hl0FsuU7mRcbvJA6464ZxagzZQP9tM4l3tzQCv1elbIqVZ%2Bby%2BfXDZ7NPOQyi0bxDcULd1CgIJAzAHd70kszO9IDjcK6m8P7InyywEHWgNs5%2B3fhj7kWf0pP0REx1p7FjxS%2BlirRY1Tsu1HEWvSFOqOtIGsEkqz%2Fm%2BywvRnq5i8jjM%2FW%2Fc27N4CXBaaumAt15E4AF2VbdArz1Rfe2CwSjRToJrtMMKM9bsGOrEB%2BPVNaeqR2wEPUISvMrtGvt0tzKvwgYvh2KTSTw%2Bdb5ml3OzaY0uv5Nm4mgfYnbvIxToq8P%2B9AUmF%2FInThGKjCY9g5MFqUIZ%2B3cFiN5HKPzWKa6xu39nJt1e%2FZgzZXF%2F8NEOmuf9sHNQZWQsxiOGUOZxkGgSQNdOTAY0Goc8vBSnJoipIgJZuvo3eE%2FqJKlReJY%2FKaPXi4NudsZ7nRW5i%2Bv%2FPWq%2Fmlc%2Fq7rX0YcaqGYwp&Expires=1736395830",
        "height": 40,
        "type": "bigwig"
    }

}
plot = igv(tracks, server_port=18090)
plot.show()


# In[23]:


plot.close()


# In[26]:


from IPython.display import Image

Image(filename='NFATC_chip.png')


# In[ ]:


hydra_celltype.gene_by_motif.data
