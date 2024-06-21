#!/bin/bash
#$ -N regenie.step1
#$ -o eddie_logs/step1.$JOB_ID.o
#$ -e eddie_logs/step1.$JOB_ID.e
#$ -cwd
#$ -pe sharedmem 2
#$ -l h_rt=24:0:0,h_vmem=8G

. /etc/profile.d/modules.sh
module load roslin/regenie/3.2.2

geno_dir="/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/genotypes/merged"
pheno_dir="/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/phenotypes"
out_dir="/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/gwas/step1/"

train_ids="/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/preprocess_gwas/train_ids.txt"

mkdir -p step1/tmp

regenie \
  --step 1 \
  --bed  "$geno_dir/all_chr_wb_unrelated"  \
  --phenoFile "$pheno_dir/pheno_residuals_train.tsv" \
  --extract "$geno_dir/qc_pass.snplist" \
  --keep "$train_ids" \
  --bsize 1000  \
  --qt \
  --out "$out_dir/fit" \
  --lowmem --lowmem-prefix "$out_dir/tmp" \
  --loocv