#!/usr/bin/env Rscript
# =====================================================================
# 02_models.R — 混合効果モデル（Duration / Intensity / F0可用性）
#
# Colab notebook `re-encoding_after_self-repair.ipynb`（Cell 2）のローカル移植版。
# %%R セルの rpy2 依存を除去し、Rscript から直接実行する。
# 入力:  <root>/output/data_for_r.csv（01_extract_pairs.py が生成）
# 出力:  <root>/output/{Table2_Duration_Model, Table3_Intensity_Model,
#                        Supplementary_F0_Model}.txt
#
# 使い方:  Rscript 02_models.R  [out_dir]
# =====================================================================

for (pkg in c("lme4", "lmerTest", "MuMIn")) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, quiet = TRUE, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

args <- commandArgs(trailingOnly = TRUE)
script_dir <- tryCatch(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
  error = function(e) ".")
root <- normalizePath(file.path(script_dir, ".."))
out_dir <- if (length(args) >= 1) args[1] else file.path(root, "output")

data <- read.csv(file.path(out_dir, "data_for_r.csv"))
data$condition <- as.factor(data$condition)
data$speaker   <- as.factor(data$speaker)
data$word      <- as.factor(data$word)
data$utt_pos   <- as.factor(data$utt_pos)

# --- 1. Duration Model (Table 2) ------------------------------------
sink(file.path(out_dir, "Table2_Duration_Model.txt"))
cat("=== TABLE 2: DURATION MODEL ===\n")
m_dur <- lmer(log_duration ~ condition + log_freq + speech_rate + utt_pos +
                (1 | speaker) + (1 | word), data = data, REML = TRUE)
print(summary(m_dur))
print(r.squaredGLMM(m_dur))
sink()

# --- 2. Intensity Model (Table 3) -----------------------------------
sink(file.path(out_dir, "Table3_Intensity_Model.txt"))
cat("=== TABLE 3: INTENSITY MODEL ===\n")
m_int <- lmer(intensity_centered ~ condition + log_freq + speech_rate + utt_pos +
                (1 | speaker) + (1 | word), data = data, REML = TRUE)
print(summary(m_int))
print(r.squaredGLMM(m_int))
sink()

# --- 3. F0 Availability Model (Supplementary) -----------------------
data$f0_fail <- data$f0_n - data$f0_k
sink(file.path(out_dir, "Supplementary_F0_Model.txt"))
cat("=== SUPPLEMENTARY: F0 MODEL ===\n")
m_f0 <- glmer(cbind(f0_k, f0_fail) ~ condition + log_freq + speech_rate + utt_pos +
                (1 | speaker), data = data, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))
print(summary(m_f0))
sink()

cat("All statistical tables saved to:", out_dir, "\n")
