#!/usr/bin/env Rscript
# =====================================================================
# 18_onset_models_v2.R — 語頭を付け直した tokens 表で F–R・流暢ベースラインを再推定（DECISIONS.md D23）
#
# 使い方:  Rscript scripts/18_onset_models_v2.R tokens_onsetfix_v2.csv
#          Rscript scripts/18_onset_models_v2.R tokens_onsetfix_cal_v2.csv
# 入力の tokens 表は 18_onset_apply_v2.py が作る（直後型 R の語頭だけを替えたもの）。
# データ準備・fit_max()・report() は 12_models_v2.R と同一。既存の出力は上書きしない。
# 出力:  <root>/output_v2/onset_models_<入力名>.{txt,csv}
# =====================================================================
suppressMessages({library(lme4); library(lmerTest); library(glmmTMB)})

script_dir <- tryCatch(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
  error = function(e) ".")
root <- normalizePath(file.path(script_dir, ".."))
out  <- file.path(root, "output_v2")
args <- commandArgs(trailingOnly = TRUE)
tok_file <- if (length(args)) args[1] else "tokens_onsetfix_v2.csv"
tag <- sub("\\.csv$", "", sub("^tokens_", "", tok_file))
tok  <- read.csv(file.path(out, tok_file))
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

# --- 変量効果構造の選択（D8）-----------------------------------------
# 順序: 最大構造 → 話者の傾きを外す → 語の傾きの相関を外す → 切片のみ。
# 相関なし（||）の項は因子のままだと全水準の列＋切片で過剰パラメータ化されるので、
# 参照水準以外の数値ダミー列で書く。
fit_max <- function(fixed, data, slope = "condition", event = TRUE) {
  ev_term <- if (event) " + (1 | event_id)" else ""
  lv <- levels(data[[slope]])[-1]
  dums <- paste0("d_", lv)
  for (i in seq_along(lv)) data[[dums[i]]] <- as.numeric(data[[slope]] == lv[i])
  forms <- c(
    sprintf("%s + (1 + %s | speaker) + (1 + %s | word)%s", fixed, slope, slope, ev_term),
    sprintf("%s + (1 | speaker) + (1 + %s | word)%s", fixed, slope, ev_term),
    sprintf("%s + (1 | speaker) + (1 + %s || word)%s", fixed, paste(dums, collapse = " + "), ev_term),
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

evo <- subset(ev, status == "ok")
abut_ids <- as.character(evo$event_id[evo$gap_cut_to_repeat == 0])
et$abut <- factor(as.character(et$event_id) %in% abut_ids, levels = c(FALSE, TRUE), labels = c("gap", "abut"))
eti <- droplevels(subset(et, keep_int == 1))
cpi <- droplevels(subset(cp, keep_int == 1))

sink(file.path(out, paste0("onset_models_", tag, ".txt")))
cat("=== 18_onset_models_v2.R:", tok_file, "===\n")
cat("Event tokens (dur):", table(et$condition), " complete pairs:", length(pair_ids), "\n")
print(table(et$condition, et$abut))

FX_DUR <- "log_dur ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
FX_INT <- "int_c ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
m_dur <- fit_max(FX_DUR, et);  show_model(m_dur, "DUR all event tokens")
m_dcp <- fit_max(FX_DUR, cp);  show_model(m_dcp, "DUR complete pairs")
m_int <- fit_max(FX_INT, eti); show_model(m_int, "INT all event tokens")
m_icp <- fit_max(FX_INT, cpi); show_model(m_icp, "INT complete pairs")
ab <- droplevels(subset(et, abut == "abut")); abi <- droplevels(subset(eti, abut == "abut"))
m_dab <- fit_max(sub(" + utt_pos", "", FX_DUR, fixed = TRUE), ab);  show_model(m_dab, "DUR abutting only")
m_iab <- fit_max(sub(" + utt_pos", "", FX_INT, fixed = TRUE), abi); show_model(m_iab, "INT abutting only")
cat("\n")
report(m_dur, "conditionrepeat", "FIX DUR repeat-first (all tokens)", SESOI_DUR)
report(m_dcp, "conditionrepeat", "FIX DUR repeat-first (complete pairs)", SESOI_DUR)
report(m_int, "conditionrepeat", "FIX INT repeat-first (all tokens)", SESOI_INT)
report(m_icp, "conditionrepeat", "FIX INT repeat-first (complete pairs)", SESOI_INT)
report(m_dab, "conditionrepeat", "FIX DUR repeat-first (abutting only)", SESOI_DUR)
report(m_iab, "conditionrepeat", "FIX INT repeat-first (abutting only)", SESOI_INT)

m0 <- lmer(log_dur ~ condition * abut + log_freq_c + rate_syl_c + utt_pos + prev_pause +
             (1 | speaker) + (1 | word) + (1 | event_id), et, REML = FALSE)
mi0 <- lmer(int_c ~ condition * abut + log_freq_c + rate_syl_c + utt_pos + prev_pause +
              (1 | speaker) + (1 | word) + (1 | event_id), eti, REML = FALSE)
cat("\n--- LRT condition x abut (duration) ---\n"); print(anova(update(m0, . ~ . - condition:abut), m0))
cat("\n--- LRT condition x abut (intensity) ---\n"); print(anova(update(mi0, . ~ . - condition:abut), mi0))
report(m0, "conditionrepeat:abutabut", "FIX DUR interaction repeat x abut (ML)")
report(mi0, "conditionrepeat:abutabut", "FIX INT interaction repeat x abut (ML)")

# --- 流暢ベースライン（12b と同じ事象の切片）-----------------------------
three$event_id <- as.character(three$event_id)
fl <- three$condition == "fluent"
three$event_id[fl] <- paste0("fl", seq_len(sum(fl)))
three$event_id <- factor(three$event_id)
th_i <- droplevels(subset(three, keep_int == 1))
FX3 <- "~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
m3_dur <- fit_max(paste("log_dur", FX3), three); show_model(m3_dur, "DUR fluent/first/repeat + event")
m3_int <- fit_max(paste("int_c", FX3), th_i);    show_model(m3_int, "INT fluent/first/repeat + event")
m3_dur_ri <- lmer(as.formula(paste("log_dur", FX3, "+ (1 | speaker) + (1 | word) + (1 | event_id)")), three, control = ctrl)
cat("\n")
for (tm in c("conditionfirst", "conditionrepeat")) {
  report(m3_dur, tm, paste("FIX DUR vs fluent:", tm))
  report(m3_dur_ri, tm, paste("FIX DUR vs fluent (intercepts only):", tm))
  report(m3_int, tm, paste("FIX INT vs fluent:", tm))
}
sink()
write.csv(do.call(rbind, est), file.path(out, paste0("onset_models_", tag, ".csv")), row.names = FALSE)
cat("Saved onset_models_", tag, ".{txt,csv}\n", sep = "")
