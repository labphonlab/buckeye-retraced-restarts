#!/usr/bin/env Rscript
# =====================================================================
# 03_interactions.R — 交互作用の尤度比検定（LRT）
#
# Colab notebook `re-encoding_after_self-repair.ipynb`（Cell 5）のローカル移植版。
# condition × speech_rate, condition × utt_pos を Duration / Intensity について検定。
# 入力:  <root>/output/data_for_r.csv
# 出力:  標準出力（＋ <root>/output/Interactions_LRT.txt に保存）
#
# 使い方:  Rscript 03_interactions.R  [out_dir]
# =====================================================================

for (pkg in c("lme4", "lmerTest")) {
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

sink(file.path(out_dir, "Interactions_LRT.txt"), split = TRUE)

# --- 1. DURATION MODEL INTERACTIONS ---------------------------------
cat("\n\U0001F535 --- LRT: DURATION INTERACTIONS ---\n")
m0_dur <- lmer(log_duration ~ condition + log_freq + speech_rate + utt_pos +
                 (1 | speaker) + (1 | word), data = data, REML = FALSE)
m1_dur <- lmer(log_duration ~ condition * speech_rate + log_freq + utt_pos +
                 (1 | speaker) + (1 | word), data = data, REML = FALSE)
m2_dur <- lmer(log_duration ~ condition * utt_pos + log_freq + speech_rate +
                 (1 | speaker) + (1 | word), data = data, REML = FALSE)
cat("\n>> Test 1: Condition x Speech Rate (Duration)\n")
print(anova(m0_dur, m1_dur))
cat("\n>> Test 2: Condition x Utt Position (Duration)\n")
print(anova(m0_dur, m2_dur))

# --- 2. INTENSITY MODEL INTERACTIONS --------------------------------
cat("\n\n\U0001F535 --- LRT: INTENSITY INTERACTIONS ---\n")
m0_int <- lmer(intensity_centered ~ condition + log_freq + speech_rate + utt_pos +
                 (1 | speaker) + (1 | word), data = data, REML = FALSE)
m1_int <- lmer(intensity_centered ~ condition * speech_rate + log_freq + utt_pos +
                 (1 | speaker) + (1 | word), data = data, REML = FALSE)
m2_int <- lmer(intensity_centered ~ condition * utt_pos + log_freq + speech_rate +
                 (1 | speaker) + (1 | word), data = data, REML = FALSE)
cat("\n>> Test 3: Condition x Speech Rate (Intensity)\n")
print(anova(m0_int, m1_int))
cat("\n>> Test 4: Condition x Utt Position (Intensity)\n")
print(anova(m0_int, m2_int))

sink()
