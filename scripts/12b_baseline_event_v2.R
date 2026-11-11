#!/usr/bin/env Rscript
# =====================================================================
# 12b_baseline_event_v2.R — 流暢ベースライン比較に事象の切片を入れた再分析
#                           （外部 AI 査読 項目6、DECISIONS.md D17）
#
# 12_models_v2.R の §4 は fluent/first/repeat の3条件モデルから (1 | event_id) を
# 外していた。F と R は同じ事象に属するので、ここでは事象の切片を入れる。
# 流暢トークンは対にならないので、各トークンに固有の event 水準を与える
# （元データでは流暢トークンの event_id はすべて -1）。
# データ準備・fit_max()・report() は 12_models_v2.R と同一。既存の出力は上書きしない。
#
# 入力:  <root>/output_v2/{tokens,events}.csv
# 出力:  <root>/output_v2/baseline_event_v2.txt
#        <root>/output_v2/baseline_event_v2.csv
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

# --- 流暢トークンに固有の event 水準を与える ---------------------------
three$event_id <- as.character(three$event_id)
fl <- three$condition == "fluent"
three$event_id[fl] <- paste0("fl", seq_len(sum(fl)))
three$event_id <- factor(three$event_id)
th_i <- droplevels(subset(three, keep_int == 1))

sink(file.path(out, "baseline_event_v2.txt"))
cat("=== 12b_baseline_event_v2.R ===\n")
cat("Tokens:", table(three$condition), " event levels:", nlevels(three$event_id), "\n")

FX3 <- "~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause"
m3_dur  <- fit_max(paste("log_dur", FX3), three);              show_model(m3_dur,  "DUR fluent/first/repeat + event")
m3_int  <- fit_max(paste("int_c", FX3), th_i);                 show_model(m3_int,  "INT fluent/first/repeat + event")
m3_intd <- fit_max(paste("int_c", FX3, "+ log_dur_c"), th_i);  show_model(m3_intd, "INT fluent/first/repeat + log duration + event")
f0b <- droplevels(subset(th_i, f0_n > 0))
m3_f0 <- glmmTMB(cbind(f0_k, f0_n - f0_k) ~ condition + log_freq_c + rate_syl_c + utt_pos + prev_pause +
                   (1 | speaker) + (1 | word) + (1 | event_id), data = f0b, family = betabinomial)
cat("\n--- F0 availability fluent/first/repeat + event ---\n"); print(summary(m3_f0)$coefficients$cond, digits = 3)
print(VarCorr(m3_f0))
m3_dur_ri <- lmer(as.formula(paste("log_dur", FX3, "+ (1 | speaker) + (1 | word) + (1 | event_id)")), three, control = ctrl)
m3_int_ri <- lmer(as.formula(paste("int_c", FX3, "+ (1 | speaker) + (1 | word) + (1 | event_id)")), th_i, control = ctrl)
show_model(m3_dur_ri, "SENSITIVITY: DUR, random intercepts only + event")
show_model(m3_int_ri, "SENSITIVITY: INT, random intercepts only + event")

cat("\n")
for (tm in c("conditionfirst", "conditionrepeat")) {
  report(m3_dur_ri, tm, paste("EV DUR vs fluent (intercepts only):", tm))
  report(m3_int_ri, tm, paste("EV INT vs fluent (intercepts only):", tm))
}
for (tm in c("conditionfirst", "conditionrepeat")) {
  report(m3_dur, tm, paste("EV DUR vs fluent:", tm))
  report(m3_int, tm, paste("EV INT vs fluent:", tm))
  report(m3_intd, tm, paste("EV INT(+dur) vs fluent:", tm))
}
for (tm in c("conditionfirst", "conditionrepeat")) {
  cf <- summary(m3_f0)$coefficients$cond[tm, ]
  est[[length(est) + 1]] <- data.frame(label = paste("EV F0AVAIL vs fluent:", tm), term = tm,
    b = cf[1], se = cf[2], df = NA, t = cf[3], p = cf[4], lo95 = cf[1] - 1.96 * cf[2],
    hi95 = cf[1] + 1.96 * cf[2], lo90 = NA, hi90 = NA, p_tost = NA, sesoi = NA, formula = "betabinomial + event")
}
# R − F（3条件モデル内）: 12_models_v2.R の F–R モデルとの整合確認用
cat("\n--- R - F within the three-condition models ---\n")
for (m in list(m3_dur, m3_int)) {
  L <- setNames(rep(0, length(fixef(m))), names(fixef(m))); L["conditionrepeat"] <- 1; L["conditionfirst"] <- -1
  print(contest(m, L, joint = FALSE))
}
sink()

write.csv(do.call(rbind, est), file.path(out, "baseline_event_v2.csv"), row.names = FALSE)
cat("Saved baseline_event_v2.txt, baseline_event_v2.csv to", out, "\n")
