# Creates a linear model mapping covariates to phenotype
# Resulting in a corrected social support phenotype

library(tidyverse)

covar = read_tsv("covar_wb_unrelated.tsv")
pheno = read_tsv("pheno_wb_unrelated.tsv")

train_id_path = "/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/preprocess_gwas/train_ids.txt"
test_id_path = "/gpfs/igmmfs01/eddie/UK-BioBank-proj19655/gzeng/preprocess_gwas/test_ids.txt"

# Convert categorical cols into factors
covar <- covar %>% 
  mutate(
    f.738.0.0 = factor(f.738.0.0),
    f.6142.0.0 = factor(f.6142.0.0),
    f.6138.0.0 = factor(f.6138.0.0),
    f.1558.0.0 = factor(f.1558.0.0),
    f.2178.0.0 = factor(f.2178.0.0),
    f.22001.0.0 = factor(f.22001.0.0)
  )

# Combine into one df
combined <- covar %>% 
  left_join(pheno, by = "IID")

# Use only training ids (90%)
train_ids <- read.table(train_id_path, 
                       sep = " ", 
                       col.names = c("FID", "IID"))

train_df <- train_ids %>% 
  left_join(combined, by = "IID") %>% 
  select(-FID.y,-FID.x,-FID,-IID)

# Linear model
model <- lm(pheno ~ ., data = train_df, na.action = na.exclude)

summary(model)

# Plot diagnostics
par(mfrow = c(2, 2)) 
plot(model) #Residuals vs Fitted, Normal Q-Q, Scale-Location, and Residuals vs Leverage


#Residuals
residuals <- resid(model)

residuals_df <- data.frame(train_ids)
residuals_df$pheno_res <- residuals

write_tsv(residuals_df, "pheno_residuals_train.tsv")

# Test dataset phenotype residuals
test_ids <- read.table(test_id_path, 
                        sep = " ", 
                        col.names = c("FID", "IID"))

test_df <- test_ids %>% 
  left_join(combined, by = "IID") %>% 
  select(-FID.y,-FID.x,-FID,-IID)

test_df_no_pheno <- test_df %>% 
  select(-pheno)

test_pheno <- test_df %>% 
  select(pheno)

pred <- predict(model, newdata=test_df_no_pheno)

test_residuals <- test_pheno - pred

test_residuals_df <- data.frame(test_ids)
test_residuals_df$pheno_res <- test_residuals$pheno

write_tsv(test_residuals_df, "pheno_residuals_test.tsv")
