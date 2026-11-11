#!/usr/bin/env Rscript
# =====================================================================
# 17_formants_models_v2.R — 母音の中央化による F–R・流暢ベースライン比較（DECISIONS.md D23）
#
# 17_formants_v2.py の central_dist（話者の母音空間の中心からの距離、Lobanov 正規化）を目的変数にする。
# 小さいほど中央化（縮約）。母音中点の値なので、語頭境界のずれの影響をほとんど受けない。
# データ準備・fit_max()・report() は 12_models_v2.R と同一。既存の出力は上書きしない。
#
# 入力:  <root>/output_v2/{tokens,events,formants_v2}.csv
# 出力:  <root>/output_v2/formants_models_v2.{txt,csv}
# =====================================================================
suppressMessages({library(lme4); library(lmerTest); library(glmmTMB)})

script_dir <- tryCatch(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))),
  error = function(e) ".")
root <- normalizePath(file.path(script_dir, ".."))
out  <- file.path(root, "output_v2")
tok  <- read.csv(file.path(out, "tokens.csv"))
ev   <- read.csv(file.path(out, "events.csv"))
fm   <- read.csv(file.path(out, "formants_v2.csv"))
# 公開表では母音ラベルを符号化し、二重母音かどうかを diphthong 列で持つ（make_public_tables.py）
if (!"diphthong" %in% names(fm)) fm$diphthong <- as.integer(fm$vowel %in% c("ay", "aw", "oy", "ey", "ow"))
tok$tok_row <- seq_len(nrow(tok)) - 1
tok <- merge(tok, fm, by = "tok_row", all.x = TRUE)

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
etf <- droplevels(subset(et, !is.na(central_dist)))
etf$abut <- as.character(etf$event_id) %in% abut_ids

sink(file.path(out, "formants_models_v2.txt"))
cat("=== 17_formants_models_v2.R ===\n")
cat("Event tokens with central_dist:", table(etf$condition), "\n")
cat("Median central_dist by condition (all tokens):\n")
print(round(tapply(tok$central_dist, tok$condition, median, na.rm = TRUE), 3))

FX <- "central_dist ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
sets <- list(
  list("all events", etf),
  list("excluding abutting", droplevels(subset(etf, !abut))),
  list("abutting only", droplevels(subset(etf, abut))),
  list("monophthongs only", droplevels(subset(etf, diphthong == 0))))
for (s in sets) {
  fx <- if (nlevels(s[[2]]$utt_pos) < 2) sub(" + utt_pos", "", FX, fixed = TRUE) else FX
  m <- fit_max(fx, s[[2]]); show_model(m, paste("CENTRAL F-R,", s[[1]]))
  cat("\n"); report(m, "conditionrepeat", paste("CENTRAL repeat-first,", s[[1]]))
}

# --- 流暢ベースライン（12b と同じ事象の切片）----------------------------
th <- droplevels(subset(three, !is.na(central_dist)))
th$event_id <- as.character(th$event_id)
fl <- th$condition == "fluent"
th$event_id[fl] <- paste0("fl", seq_len(sum(fl)))
th$event_id <- factor(th$event_id)
FX3 <- "central_dist ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
m3 <- fit_max(FX3, th); show_model(m3, "CENTRAL fluent/first/repeat + event")
m3v <- fit_max(paste(FX3, "+ log_dur_c"), th); show_model(m3v, "CENTRAL fluent/first/repeat + log duration + event")
cat("\n")
for (tm in c("conditionfirst", "conditionrepeat")) {
  report(m3, tm, paste("CENTRAL vs fluent:", tm))
  report(m3v, tm, paste("CENTRAL(+dur) vs fluent:", tm))
}
report(m3, "log_freq_c", "CENTRAL frequency (per log10 unit), baseline model")
report(m3v, "log_dur_c", "CENTRAL log duration, baseline model")
cat("\nSD of central_dist, fluent tokens:", round(sd(th$central_dist[fl]), 3), "\n")
sink()
write.csv(do.call(rbind, est), file.path(out, "formants_models_v2.csv"), row.names = FALSE)
cat("Saved formants_models_v2.{txt,csv} to", out, "\n")
