# Takes a subset of columns, and outputs pheno and covariate files
# Creates social support score from questionnaire data

library(data.table)
library(tidyverse)

dataset_tsv_path <- "/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/phenotypes/2019_03_27/dataset_27263/p03_usable_data/ukb27263.tsv"
sample_id_path <- "/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/preprocess_gwas/white_british_unrelated_ids.txt"

sample_ids <- read.table(sample_id_path, 
                         sep = " ", 
                         col.names = c("FID", "IID"))

pheno_fields <- c(
  "f.1031.0.0", # Frequency of friend/family visits
  "f.2110.0.0", # Able to confide
  paste0("f.6160.0.",seq(0,4)), # Number of social activities a week
  "f.709.0.0" # Number in household
  )

covar_fields <- c(
  "f.21003.0.0", #Age when attended assessment center
  "f.738.0.0", #Avg household income
  "f.6142.0.0", #Current employment status
  "f.6138.0.0", #Qualifications 
  "f.1558.0.0", #Alcohol intake freq
  "f.2178.0.0", #overall health rating
  "f.22001.0.0", #genetic sex
  paste0("f.22009.0.",seq(1,20)) #Genetic PCs 1-20
)

dataset_subset_cols <- read.csv("../preprocess_gwas/dataset_subset.csv")

# Select cols from subset
dataset_subset_cols_temp <- dataset_subset_cols %>% 
  select(c("f.eid", pheno_fields, covar_fields))


dataset_subset_cols_encoded <- dataset_subset_cols_temp %>% 
  mutate(
    # Convert covar character columns into factors and use integer encoding
    f.738.0.0 = as.integer(factor(f.738.0.0)),
    f.6142.0.0 = as.integer(factor(f.6142.0.0)),
    f.6138.0.0 = as.integer(factor(f.6138.0.0)),
    f.1558.0.0 = as.integer(factor(f.1558.0.0)),
    f.2178.0.0 = as.integer(factor(f.2178.0.0)),
    f.22001.0.0 = as.integer(factor(f.22001.0.0)),
    # Convert pheno columns into int
    pheno_visits = case_when(
      f.1031.0.0 == "No friends/family outside household" ~ 0,
      f.1031.0.0 =="Never or almost never" ~ 1,
      f.1031.0.0 =="Once every few months" ~ 2,
      f.1031.0.0 =="About once a month" ~ 3,
      f.1031.0.0 =="About once a week" ~ 4,
      f.1031.0.0 =="2-4 times a week" ~ 5,
      f.1031.0.0 =="Almost daily" ~ 6,
      .default = NA
      
    ),
    pheno_confide = case_when(
      f.2110.0.0 == "No friends/family outside household" ~ 0,
      f.2110.0.0 =="Never or almost never" ~ 1,
      f.2110.0.0 =="Once every few months" ~ 2,
      f.2110.0.0 =="About once a month" ~ 3,
      f.2110.0.0 =="About once a week" ~ 4,
      f.2110.0.0 =="2-4 times a week" ~ 5,
      f.2110.0.0 =="Almost daily" ~ 6,
      .default = NA
      
    ),
    pheno_activities = if_else(
      f.6160.0.0 == "None of the above",
      0,
      if_else(
        f.6160.0.0 == "Prefer not to answer" | is.na(f.6160.0.0),
        NA,
        case_when(
          !is.na(f.6160.0.4) ~ 5,
          !is.na(f.6160.0.3) ~ 4,
          !is.na(f.6160.0.2) ~ 3,
          !is.na(f.6160.0.1) ~ 2,
          !is.na(f.6160.0.0) ~ 1
        )
      )
    ),
    pheno_household = case_when(
      f.709.0.0>7 ~ 7, # Trim those above 7
      TRUE ~ f.709.0.0
    )
  ) %>% 
  mutate(
    pheno = rowSums(across(c(pheno_household, pheno_confide, pheno_activities, pheno_visits)),
                    na.rm=FALSE) # If NA in any col, dont sum
  ) %>% 
  # Keep only "pheno" after processing
  select(-pheno_fields,-pheno_household,-pheno_activities,-pheno_visits,-pheno_confide)
  

dataset_subset_sample <- sample_ids %>% left_join(
  dataset_subset_cols_encoded,
  by=c("IID"="f.eid"))

# Covariate sample output
covar_wb_unrelated.tsv <- dataset_subset_sample %>% 
  select(-pheno)

write_tsv(covar_wb_unrelated.tsv, "covar_wb_unrelated.tsv")

# Pheno sample output
pheno_wb_unrelated.tsv <- dataset_subset_sample %>% 
  select(c("FID","IID",pheno))
                                         

write_tsv(pheno_wb_unrelated.tsv, "pheno_wb_unrelated.tsv")
