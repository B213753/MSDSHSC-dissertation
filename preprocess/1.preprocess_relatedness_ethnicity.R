# Script which processes relatedness and ethnicity inclusion criteria
# Outputs list to keep

library(tidyverse)

rel_file_path <- "/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/genotypes/meta_data/ukb19655_rel_s488363.dat"
sample_qc_file_path <- "/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/genotypes/meta_data/ukbb_proj19655_sample_qc.tsv"

  
# Include unrelated (using used.in.pca.calculation in sample qc)- this is a (maximal) subset of unrelated participants after applying some QC filtering.
sample_qc <- read_tsv(sample_qc_file_path, 
                           col_select = c("iid",
                                          "in.white.British.ancestry.subset",
                                          "used.in.pca.calculation"))

inclusion_list <- sample_qc %>% 
  filter(in.white.British.ancestry.subset==1 & used.in.pca.calculation==1) %>% 
  select(iid)

# regenie --keep FID + IID Columns format
regenie_format <- inclusion_list %>% 
  mutate(iid2 = iid)

#337545 samples. 

# # Save
write.table(regenie_format, "white_british_unrelated_ids.txt", 
            row.names=FALSE,col.names = FALSE, quote = FALSE)