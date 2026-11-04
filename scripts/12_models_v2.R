#!/usr/bin/env Rscript
# =====================================================================
# 12_models_v2.R — 修正版データの混合効果モデル（R1 改稿用、DECISIONS.md D8）
#
# 入力:  <root>/output_v2/{tokens,events}.csv（11_extract_v2.py）
# 出力:  <root>/output_v2/models_v2.txt（全モデルの要約）
#        <root>/output_v2/estimates_v2.csv（本文・図で使う推定値）
# =====================================================================
suppressMessages({library(lme4); library(lmerTest); library(glmmTMB)})

script_dir <- tryCatch(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
  error = function(e) ".")
root <- normalizePath(file.path(script_dir, ".."))
out  <- file.path(root, "output_v2")
tok  <- read.csv(file.path(out, "tokens.csv"))
ev   <- read.csv(file.path(out, "events.csv"))

SESOI_DUR <- log(1.10)   # ±10%
SESOI_INT <- 1.0         # ±1 dB
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

# --- データ準備 -------------------------------------------------------
tok <- subset(tok, keep_dur == 1)
evw <- unique(tok$word[tok$condition != "fluent"])
tok <- subset(tok, condition != "fluent" | word %in% evw)
tok$log_dur <- log(tok$duration)
tok$speaker <- factor(tok$speaker); tok$word <- factor(tok$word)
tok$event_id <- factor(tok$event_id)
tok$utt_pos <- factor(tok$utt_pos); tok$prev_pause <- factor(tok$prev_pause)
ok_int <- tok$keep_int == 1
spk_mean  <- tapply(tok$intensity[ok_int], tok$speaker[ok_int], mean)
spk_meanv <- tapply(tok$int_peak_vowel[ok_int], tok$speaker[ok_int], mean, na.rm = TRUE)
tok$int_c  <- ifelse(ok_int, tok$intensity - spk_mean[as.character(tok$speaker)], NA)
tok$intv_c <- ifelse(ok_int, tok$int_peak_vowel - spk_meanv[as.character(tok$speaker)], NA)
tok$f0_avail <- ifelse(tok$f0_n > 0, tok$f0_k / tok$f0_n, NA)
for (v in c("log_freq", "log_freq_subtlex", "rate_syl", "rate_wps")) {
  tok[[paste0(v, "_c")]] <- tok[[v]] - mean(tok[[v]], na.rm = TRUE)
}
tok$log_dur_c <- tok$log_dur - mean(tok$log_dur)

et <- droplevels(subset(tok, condition != "fluent"))
et$condition <- factor(et$condition, levels = c("first", "repeat"))
pair_ids <- names(which(table(et$event_id) == 2))
cp <- droplevels(subset(et, event_id %in% pair_ids))
three <- droplevels(tok)
three$condition <- factor(three$condition, levels = c("fluent", "first", "repeat"))

sink(file.path(out, "models_v2.txt"))
cat("=== 12_models_v2.R ===\n")
cat("Event tokens (dur):", table(et$condition), " complete pairs:", length(pair_ids),
    " fluent:", sum(tok$condition == "fluent"), "\n\n")

# --- 変量効果構造の選択（D8）-----------------------------------------
fit_max <- function(fixed, data, slope = "condition", event = TRUE) {
  ev_term <- if (event) " + (1 | event_id)" else ""
  forms <- c(
    sprintf("%s + (1 + %s | speaker) + (1 + %s | word)%s", fixed, slope, slope, ev_term),
    sprintf("%s + (1 + %s || speaker) + (1 + %s || word)%s", fixed, slope, slope, ev_term),
    sprintf("%s + (1 + %s || speaker) + (1 | word)%s", fixed, slope, ev_term),
    sprintf("%s + (1 | speaker) + (1 + %s || word)%s", fixed, slope, ev_term),
    sprintf("%s + (1 | speaker) + (1 | word)%s", fixed, ev_term))
  for (f in forms) {
    m <- suppressWarnings(suppressMessages(lmer(as.formula(f), data = data, control = ctrl)))
    conv <- is.null(m@optinfo$conv$lme4$messages)
    if (!isSingular(m) && conv) return(m)
  }
  m
}

est <- list()
report <- function(m, term, label, sesoi = NA, transform = identity) {
  cf <- summary(m)$coefficients
  b <- cf[term, "Estimate"]; se <- cf[term, "Std. Error"]; df <- cf[term, "df"]
  p <- cf[term, "Pr(>|t|)"]
  ci95 <- b + c(-1, 1) * qt(.975, df) * se
  ci90 <- b + c(-1, 1) * qt(.95, df) * se
  p_tost <- if (is.na(sesoi)) NA else
    max(pt((b + sesoi) / se, df, lower.tail = FALSE), pt((b - sesoi) / se, df))
  cat(sprintf("%-48s %-18s b=%7.3f SE=%6.3f df=%6.1f t=%6.2f p=%.3f 95%%CI[%6.3f,%6.3f] 90%%CI[%6.3f,%6.3f] TOST p=%s\n",
              label, term, b, se, df, b / se, p, ci95[1], ci95[2], ci90[1], ci90[2],
              ifelse(is.na(p_tost), "-", sprintf("%.4f", p_tost))))
  est[[length(est) + 1]] <<- data.frame(label = label, term = term, b = b, se = se, df = df,
    t = b / se, p = p, lo95 = ci95[1], hi95 = ci95[2], lo90 = ci90[1], hi90 = ci90[2],
    p_tost = p_tost, sesoi = sesoi, formula = paste(deparse(formula(m), width.cutoff = 500), collapse = ""))
}
show_model <- function(m, title) {
  cat("\n---", title, "---\n"); print(formula(m)); print(summary(m)$coefficients, digits = 3)
  print(VarCorr(m), comp = "Std.Dev."); cat("singular:", isSingular(m), "\n")
}

# --- 1. 持続時間: first vs repeat -----------------------------------
FX_DUR <- "log_dur ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
m_dur_all <- fit_max(FX_DUR, et);  show_model(m_dur_all, "DUR all event tokens")
m_dur_cp  <- fit_max(FX_DUR, cp);  show_model(m_dur_cp,  "DUR complete pairs")
m_dur_orig <- lmer(log_dur ~ condition + log_freq_c + rate_wps_c + utt_pos + (1 | speaker) + (1 | word), et)
m_dur_subt <- lmer(log_dur ~ condition + log_freq_subtlex_c + rate_syl_c + utt_pos + prev_pause +
                     (1 | speaker) + (1 | word) + (1 | event_id), et)
cat("\n")
report(m_dur_all, "conditionrepeat", "DUR repeat-first (all tokens)", SESOI_DUR)
report(m_dur_cp,  "conditionrepeat", "DUR repeat-first (complete pairs)", SESOI_DUR)
report(m_dur_orig, "conditionrepeat", "DUR repeat-first (submitted model spec)", SESOI_DUR)
report(m_dur_subt, "conditionrepeat", "DUR repeat-first (SUBTLEX frequency)", SESOI_DUR)
report(m_dur_all, "log_freq_c", "DUR frequency (all tokens)")
report(m_dur_all, "rate_syl_c", "DUR speech rate (all tokens)")

# --- 2. 強度: first vs repeat ---------------------------------------
eti <- droplevels(subset(et, keep_int == 1)); cpi <- droplevels(subset(cp, keep_int == 1))
FX_INT <- "int_c ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
m_int_all <- fit_max(FX_INT, eti); show_model(m_int_all, "INT all event tokens")
m_int_cp  <- fit_max(FX_INT, cpi); show_model(m_int_cp,  "INT complete pairs")
m_int_dur <- fit_max(paste(FX_INT, "+ log_dur_c"), eti); show_model(m_int_dur, "INT + log duration")
m_intv    <- fit_max(sub("int_c", "intv_c", FX_INT), droplevels(subset(eti, !is.na(intv_c))))
show_model(m_intv, "PEAK-VOWEL INT")
cat("\n")
report(m_int_all, "conditionrepeat", "INT repeat-first (all tokens)", SESOI_INT)
report(m_int_cp,  "conditionrepeat", "INT repeat-first (complete pairs)", SESOI_INT)
report(m_int_dur, "conditionrepeat", "INT repeat-first (+log duration)", SESOI_INT)
report(m_intv,    "conditionrepeat", "VOWEL INT repeat-first", SESOI_INT)

# --- 3. F0 可用性（token 単位 beta-binomial）-----------------------
f0d <- droplevels(subset(eti, f0_n > 0))
m_f0 <- glmmTMB(cbind(f0_k, f0_n - f0_k) ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause +
                  (1 | speaker) + (1 | word) + (1 | event_id), data = f0d, family = betabinomial)
cat("\n--- F0 availability, beta-binomial (token-level) ---\n"); print(summary(m_f0)$coefficients$cond, digits = 3)
cf <- summary(m_f0)$coefficients$cond["conditionrepeat", ]
est[[length(est) + 1]] <- data.frame(label = "F0AVAIL repeat-first (logit, beta-binomial)",
  term = "conditionrepeat", b = cf[1], se = cf[2], df = NA, t = cf[3], p = cf[4],
  lo95 = cf[1] - 1.96 * cf[2], hi95 = cf[1] + 1.96 * cf[2], lo90 = NA, hi90 = NA,
  p_tost = NA, sesoi = NA, formula = "betabinomial")
cat("Mean F0 availability:\n"); print(round(tapply(f0d$f0_avail, f0d$condition, mean), 3))

# --- 4. 流暢ベースラインとの比較 -------------------------------------
FX3 <- "~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
m3_dur <- fit_max(paste("log_dur", FX3), three, event = FALSE); show_model(m3_dur, "DUR fluent/first/repeat")
th_i <- droplevels(subset(three, keep_int == 1))
m3_int <- fit_max(paste("int_c", FX3), th_i, event = FALSE); show_model(m3_int, "INT fluent/first/repeat")
m3_intd <- fit_max(paste("int_c", FX3, "+ log_dur_c"), th_i, event = FALSE)
show_model(m3_intd, "INT fluent/first/repeat + log duration")
m3_f0 <- glmmTMB(cbind(f0_k, f0_n - f0_k) ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause +
                   (1 | speaker) + (1 | word), data = droplevels(subset(th_i, f0_n > 0)), family = betabinomial)
cat("\n--- F0 availability fluent/first/repeat ---\n"); print(summary(m3_f0)$coefficients$cond, digits = 3)
cat("\n")
for (tm in c("conditionfirst", "conditionrepeat")) {
  report(m3_dur, tm, paste("DUR vs fluent:", tm))
  report(m3_int, tm, paste("INT vs fluent:", tm))
  report(m3_intd, tm, paste("INT(+dur) vs fluent:", tm))
}
for (tm in c("conditionfirst", "conditionrepeat")) {
  cf <- summary(m3_f0)$coefficients$cond[tm, ]
  est[[length(est) + 1]] <- data.frame(label = paste("F0AVAIL vs fluent:", tm), term = tm,
    b = cf[1], se = cf[2], df = NA, t = cf[3], p = cf[4], lo95 = cf[1] - 1.96 * cf[2],
    hi95 = cf[1] + 1.96 * cf[2], lo90 = NA, hi90 = NA, p_tost = NA, sesoi = NA, formula = "betabinomial")
}

# --- 5. 探索的: 事象レベル変数による repeat−first 差の調整 ------------
w <- reshape(cp[, c("event_id", "speaker", "word", "condition", "log_dur", "int_c")],
             idvar = c("event_id", "speaker", "word"), timevar = "condition", direction = "wide")
evo <- subset(ev, status == "ok"); evo$event_id <- factor(evo$event_id)
w <- merge(w, evo[, c("event_id", "gap_cut_to_repeat", "cut_frag_dur", "retrace_len", "n_between_lex")])
w$d_dur <- w$log_dur.repeat - w$log_dur.first
w$d_int <- w$int_c.repeat - w$int_c.first
w$gap_log <- log(w$gap_cut_to_repeat + 0.05); w$gap_log <- w$gap_log - mean(w$gap_log)
w$frag_c <- w$cut_frag_dur - mean(w$cut_frag_dur)
w$retrace_multi <- as.integer(w$retrace_len > 1)
# 介在語のうち retrace 部分（retrace_len − 1 語）以外の語があれば editing material あり
w$editing <- as.integer(w$n_between_lex - (w$retrace_len - 1) > 0)
m_mod_dur <- lmer(d_dur ~ gap_log + frag_c + retrace_multi + editing + (1 | speaker) + (1 | word), w)
m_mod_int <- lmer(d_int ~ gap_log + frag_c + retrace_multi + editing + (1 | speaker) + (1 | word), w)
show_model(m_mod_dur, "EXPLORATORY: repeat-first log-duration difference ~ event variables")
show_model(m_mod_int, "EXPLORATORY: repeat-first intensity difference ~ event variables")
cat("\n")
for (tm in c("gap_log", "frag_c", "retrace_multi", "editing")) {
  report(m_mod_dur, tm, paste("MOD dur:", tm)); report(m_mod_int, tm, paste("MOD int:", tm))
}

# --- 6. 交互作用（RQ3、原稿との連続性）-------------------------------
m0 <- lmer(log_dur ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause +
             (1 | speaker) + (1 | word) + (1 | event_id), et, REML = FALSE)
cat("\n--- LRT interactions (duration) ---\n")
print(anova(m0, update(m0, . ~ . + condition:rate_syl_c)))
print(anova(m0, update(m0, . ~ . + condition:prev_pause)))
mi0 <- lmer(int_c ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause +
              (1 | speaker) + (1 | word) + (1 | event_id), eti, REML = FALSE)
cat("\n--- LRT interactions (intensity) ---\n")
print(anova(mi0, update(mi0, . ~ . + condition:rate_syl_c)))
print(anova(mi0, update(mi0, . ~ . + condition:prev_pause)))

# --- 7. 記述統計 -----------------------------------------------------
cat("\n--- Descriptives ---\n")
desc <- do.call(rbind, lapply(split(tok, tok$condition), function(d) data.frame(
  condition = d$condition[1], n_dur = nrow(d), dur_mean = mean(d$duration) * 1000,
  dur_sd = sd(d$duration) * 1000, logdur_mean = mean(d$log_dur), logdur_sd = sd(d$log_dur),
  n_int = sum(!is.na(d$int_c)), int_mean = mean(d$int_c, na.rm = TRUE), int_sd = sd(d$int_c, na.rm = TRUE),
  f0_mean = mean(d$f0_avail[!is.na(d$int_c)], na.rm = TRUE),
  f0_sd = sd(d$f0_avail[!is.na(d$int_c)], na.rm = TRUE),
  prev_pause = mean(d$prev_pause == 1), utt_final = mean(d$utt_pos == 1))))
print(desc, digits = 3)
cat("\nPaired (complete pairs): repeat/first ratio =", round(exp(mean(w$d_dur)), 3),
    "; prop repeat shorter =", round(mean(w$d_dur < 0), 3),
    "; mean int diff =", round(mean(w$d_int, na.rm = TRUE), 2),
    "; prop repeat louder =", round(mean(w$d_int > 0, na.rm = TRUE), 3), "\n")
cat("Event variables:\n"); print(summary(evo[, c("cut_frag_dur", "gap_cut_to_repeat", "sil_after_cut")]))
print(table(retrace_len = evo$retrace_len)); print(table(n_between_lex = evo$n_between_lex))
print(table(editing_material = evo$n_between_lex - (evo$retrace_len - 1) > 0))
sink()

write.csv(do.call(rbind, est), file.path(out, "estimates_v2.csv"), row.names = FALSE)
write.csv(desc, file.path(out, "descriptives_v2.csv"), row.names = FALSE)
write.csv(w, file.path(out, "pairs_wide_v2.csv"), row.names = FALSE)
cat("Saved models_v2.txt, estimates_v2.csv, descriptives_v2.csv, pairs_wide_v2.csv to", out, "\n")
