#!/bin/bash
#$ -N regenie.step2
#$ -o eddie_logs/step2.$JOB_ID.$TASK_ID.o
#$ -e eddie_logs/step2.$JOB_ID.$TASK_ID.e
#$ -cwd
#$ -pe sharedmem 4
#$ -l h_vmem=8G
#$ -tc 20

# qsub -t 1-22 regenie.step2.sh

. /etc/profile.d/modules.sh
module load roslin/regenie/3.2.2

geno_dir="/gpfs/igmmfs01/eddie/UK-BioBank-Genotype/imputed/v3"
sample_dir="/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/genotypes/imputed"
pheno_dir="/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/phenotypes"
train_ids="/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/preprocess_gwas/train_ids.txt"
out_dir="/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/gwas/step2/"

chrom=${SGE_TASK_ID}

regenie \
  --step 2 \
  --bgen "$geno_dir/ukb_imp_chr${chrom}_v3.bgen"\
  --ref-first \
  --sample "$sample_dir/ukb19655_imp_chr${chrom}_v3_s487395.sample" \
  --keep "$train_ids" \
  --phenoFile "$pheno_dir/pheno_residuals_train.tsv" \
  --qt \
  --pred "/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/gwas/step1/fit_pred.list" \
  --bsize 400 \
  --out "$out_dir/chr${chrom}"