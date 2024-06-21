library(tidyverse)
options(scipen = 999)

gwas_path <- "/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/gwas/step2/all_chr.p_formatted.pre_qc.regenie"
hardy_path <- "/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/genotypes/bgen_stats/all_chr_train.hardy"
out_file <-"/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/gwas/step2/all_chr.qc_pass.maf0-01_info0-8.regenie"

gwas <- read.table(gwas_path,
                   sep="\t",
                   header=TRUE,
                   row.names=NULL,
                   )


hardy <- read.table(hardy_path,
                    sep="\t",
                    header=TRUE,
                    row.names=NULL,
) %>% 
  rename(HARDY_P = P)

# Keep only first occurrence of ID
gwas_filtered <- gwas %>% 
  group_by(ID) %>%
  filter(row_number() == 1) %>%
  ungroup()

hardy_filtered <- hardy %>% 
  group_by(ID) %>%
  filter(row_number() == 1) %>%
  ungroup()

# Merge by IDs (deduplicated above)
gwas_merged <- gwas_filtered %>% 
  left_join(hardy_filtered %>% select(ID,HARDY_P), by = "ID")

gwas_qc_pass <- gwas_merged %>% 
  filter(
    A1FREQ >= 0.01,
    INFO >= 0.8,
    HARDY_P >=1e-30
  )

write.table(gwas_qc_pass, file = out_file, sep = "\t", row.names = FALSE, quote = FALSE)