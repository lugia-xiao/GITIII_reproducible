setwd("/gpfs/gibbs/project/wang_zuoheng/xx244/GITIII_benchmark")

Sys.setenv(RETICULATE_AUTOCONFIGURE = "FALSE")
Sys.setenv(RETICULATE_PYTHON = "/home/xx244/.conda/envs/benchmark/bin/python")
library(SpaTalk)
library(reticulate)
torch <- import("torch")

py_list <- torch$load("/gpfs/gibbs/project/wang_zuoheng/xx244/GITIII_benchmark/data/NSCLC/genes.pth")
vec <- unlist(py_list)
genes<-ifelse(grepl("^[0-9]", vec), paste0("X", vec), vec)
genes<-gsub("-", ".", genes)

df<-read.csv("/gpfs/gibbs/project/wang_zuoheng/xx244/GITIII_benchmark/data/NSCLC/NSCLC.csv")
rownames(df) <- paste0("C", rownames(df))
df<-df[df$section=="Lung6",]
exp<-t(as.matrix(df[,genes]))

metadata <- data.frame(
  cell = rownames(df),  # Extract row names
  x = df$CenterX_global_px,       # Extract 'centerx' column
  y = df$CenterY_global_px        # Extract 'centery' column
)

obj <- createSpaTalk(st_data = exp,
                     st_meta = metadata,
                     species = "Human",
                     if_st_is_sc = T,
                     spot_max_cell = 1,
                     celltype = df$CellType)

# Filter LRIs with downstream targets
obj <- find_lr_path(object = obj, lrpairs = lrpairs, pathways = pathways)
obj <- dec_cci_all(object = obj)

save(obj, file = "./NSCLC_spatalk.RData")