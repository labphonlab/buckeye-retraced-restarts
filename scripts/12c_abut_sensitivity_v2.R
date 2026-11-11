#!/usr/bin/env Rscript
# =====================================================================
# 12c_abut_sensitivity_v2.R — R が cutoff 断片の直後に始まる事象を除いた感度分析
#                             （外部 AI 査読 項目13、DECISIONS.md D19・D20）
#
# 語境界の人手確認（14_boundary_check_*）で、ずれた R の語頭 3 件はいずれも
# 断片の直後（gap_cut_to_repeat == 0）だった。この境界の誤りが F–R の比較を
# 左右していないかを、(a) 直後型を除く、(b) 直後型だけ、(c) 条件×直後型の
# 交互作用、の3通りで確かめる。
# データ準備・fit_max()・report() は 12_models_v2.R と同一。既存の出力は上書きしない。
#
# 入力:  <root>/output_v2/{tokens,events}.csv
# 出力:  <root>/output_v2/abut_sensitivity_v2.{txt,csv}
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

# --- 直後型の事象（R が断片の終わりと同時に始まる）--------------------
evo <- subset(ev, status == "ok")
abut_ids <- as.character(evo$event_id[evo$gap_cut_to_repeat == 0])
et$abut <- factor(as.character(et$event_id) %in% abut_ids, levels = c(FALSE, TRUE), labels = c("gap", "abut"))
eti <- droplevels(subset(et, keep_int == 1))

sink(file.path(out, "abut_sensitivity_v2.txt"))
cat("=== 12c_abut_sensitivity_v2.R ===\n")
cat("Events ok:", nrow(evo), " abutting (gap_cut_to_repeat == 0):", length(abut_ids), "\n")
cat("Duration tokens by condition x abut:\n"); print(table(et$condition, et$abut))
cat("Intensity tokens by condition x abut:\n"); print(table(eti$condition, eti$abut))

FX_DUR <- "log_dur ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
FX_INT <- "int_c ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
fits <- list(
  list("DUR", "excluding abutting", FX_DUR, droplevels(subset(et, abut == "gap")), SESOI_DUR),
  list("DUR", "abutting only",      FX_DUR, droplevels(subset(et, abut == "abut")), SESOI_DUR),
  list("INT", "excluding abutting", FX_INT, droplevels(subset(eti, abut == "gap")), SESOI_INT),
  list("INT", "abutting only",      FX_INT, droplevels(subset(eti, abut == "abut")), SESOI_INT))
# 直後型だけの部分集合では line 末の語がない（utt_pos が1水準）ので、その項を外す
ms <- lapply(fits, function(x) {
  fx <- if (nlevels(x[[4]]$utt_pos) < 2) sub(" + utt_pos", "", x[[3]], fixed = TRUE) else x[[3]]
  m <- fit_max(fx, x[[4]]); show_model(m, paste(x[[1]], x[[2]])); m
})
cat("\n")
for (i in seq_along(fits)) report(ms[[i]], "conditionrepeat", paste(fits[[i]][[1]], "repeat-first,", fits[[i]][[2]]), fits[[i]][[5]])

# --- 交互作用: R−F 差が直後型で異なるか（12_models_v2.R §6 と同じ切片構造、ML）--
m0 <- lmer(log_dur ~ condition * abut + log_freq_c + rate_syl_c + utt_pos + prev_pause +
             (1 | speaker) + (1 | word) + (1 | event_id), et, REML = FALSE)
mi0 <- lmer(int_c ~ condition * abut + log_freq_c + rate_syl_c + utt_pos + prev_pause +
              (1 | speaker) + (1 | word) + (1 | event_id), eti, REML = FALSE)
cat("\n--- LRT condition x abut (duration) ---\n")
print(anova(update(m0, . ~ . - condition:abut), m0))
cat("\n--- LRT condition x abut (intensity) ---\n")
print(anova(update(mi0, . ~ . - condition:abut), mi0))
cat("\n")
report(m0, "conditionrepeat:abutabut", "DUR interaction repeat x abut (ML)")
report(mi0, "conditionrepeat:abutabut", "INT interaction repeat x abut (ML)")

# --- 流暢ベースライン（12b と同じ事象の切片）を直後型の事象を除いて -----------
th <- droplevels(subset(three, !(as.character(event_id) %in% abut_ids)))
th$event_id <- as.character(th$event_id)
fl <- th$condition == "fluent"
th$event_id[fl] <- paste0("fl", seq_len(sum(fl)))
th$event_id <- factor(th$event_id)
th_i <- droplevels(subset(th, keep_int == 1))
FX3 <- "~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
m3_dur <- fit_max(paste("log_dur", FX3), th); show_model(m3_dur, "DUR fluent/first/repeat, excluding abutting")
m3_int <- fit_max(paste("int_c", FX3), th_i); show_model(m3_int, "INT fluent/first/repeat, excluding abutting")
m3_dur_ri <- lmer(as.formula(paste("log_dur", FX3, "+ (1 | speaker) + (1 | word) + (1 | event_id)")), th, control = ctrl)
cat("\n")
for (tm in c("conditionfirst", "conditionrepeat")) {
  report(m3_dur, tm, paste("NOABUT DUR vs fluent:", tm))
  report(m3_dur_ri, tm, paste("NOABUT DUR vs fluent (intercepts only):", tm))
  report(m3_int, tm, paste("NOABUT INT vs fluent:", tm))
}

# --- 事象レベル変数（12_models_v2.R §5）: 介在語は直後型と重なるので、直後型を除いて再推定 --
w <- reshape(et[, c("event_id", "speaker", "word", "condition", "log_dur", "int_c")],
             idvar = c("event_id", "speaker", "word"), timevar = "condition", direction = "wide")
w <- na.omit(w)
evo$event_id <- factor(evo$event_id)
w <- merge(w, evo[, c("event_id", "gap_cut_to_repeat", "cut_frag_dur", "retrace_len", "n_between_lex")])
w$d_dur <- w$log_dur.repeat - w$log_dur.first
w$frag_c <- w$cut_frag_dur - mean(w$cut_frag_dur)
w$retrace_multi <- as.integer(w$retrace_len > 1)
w$editing <- as.integer(w$n_between_lex - (w$retrace_len - 1) > 0)
w$abut <- as.integer(w$gap_cut_to_repeat == 0)
cat("\nEditing material by abut (complete pairs):\n"); print(table(abut = w$abut, editing = w$editing))
wg <- subset(w, abut == 0)
wg$gap_log <- log(wg$gap_cut_to_repeat); wg$gap_log <- wg$gap_log - mean(wg$gap_log)
m_mod_g <- lmer(d_dur ~ gap_log + frag_c + retrace_multi + editing + (1 | speaker) + (1 | word), wg)
show_model(m_mod_g, "EXPLORATORY: R-F log-duration difference ~ event variables, excluding abutting")
m_mod_a <- lmer(d_dur ~ abut + frag_c + retrace_multi + editing + (1 | speaker) + (1 | word), w)
show_model(m_mod_a, "EXPLORATORY: R-F log-duration difference ~ abut + event variables")
cat("\n")
for (tm in c("gap_log", "frag_c", "retrace_multi", "editing")) report(m_mod_g, tm, paste("NOABUT MOD dur:", tm))
for (tm in c("abut", "editing")) report(m_mod_a, tm, paste("MOD dur with abut:", tm))
sink()

write.csv(do.call(rbind, est), file.path(out, "abut_sensitivity_v2.csv"), row.names = FALSE)
cat("Saved abut_sensitivity_v2.{txt,csv} to", out, "\n")
