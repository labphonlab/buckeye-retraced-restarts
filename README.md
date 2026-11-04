# Retraced same-word restarts in the Buckeye Corpus: analysis code and measurements

Code and token-level measurements for:

> Ishihara, T. *Same plan, same word? Duration and intensity of retraced same-word restarts in spontaneous American English.* Manuscript under review at *Speech Communication* (SPECOM-D-26-00036, revision 1).

The study compares the first production of a word with its repetition after a cutoff (*so **we** \<ju-\> **we** just had…*) in 542 repair events from 40 speakers of the Buckeye Corpus, and compares both with fluent tokens of the same words by the same speakers.

## A correction you can inspect

The originally submitted version of this study read the time stamps in the Buckeye `.words` files as word **onsets**. They mark word **offsets**: an entry starts where the preceding entry ends. As a result, the earlier pipeline measured the interval from the end of each target word to the end of the next lexical word, never the target word itself. The reported effects (repetitions 32% shorter and 2.3 dB louder) were artifacts of this error.

Both pipelines are included so that the error and its correction can be checked:

| Script | Status |
|---|---|
| `scripts/01_extract_pairs.py` | Original extraction. `load_words_robust()` contains the boundary error. The corrected pipeline still imports its event-detection functions (`scan_candidates`, `normalize_token`), which are unaffected. |
| `scripts/legacy/02_models.R`, `03_interactions.R`, `04_f0_supplementary.py` | Original models, kept for reference only. |
| `scripts/11_extract_v2.py` | Corrected extraction and acoustic measurement (boundaries read as offsets). |
| `scripts/12_models_v2.R` | Mixed-effects models, equivalence tests, fluent-baseline comparisons. |
| `scripts/13_figures_v2.py` | Figures 1, 2 and S1. |

## What is (and is not) in this repository

The Buckeye Corpus is licensed by The Ohio State University and may not be redistributed. This repository therefore contains **no audio, transcriptions, word labels or corpus time stamps**. It contains only the measurements made for this study (supplemental annotations):

- `output_v2/tokens.csv`: one row per analysed token (repair tokens and fluent baseline). Tokens are identified by recording (`basename`) and the 0-based index of the entry in that recording's `.words` file (`words_idx`). Word types are replaced by codes (`word` = `w001`…`w126`), assigned in the R collation order of the original labels.
- `output_v2/events.csv`: one row per candidate repair event, with event-level variables (fragment duration, cutoff-to-restart interval, retrace length, intervening words) and the selection status.
- `output_v2/selection_counts.csv`, `models_v2.txt`, `estimates_v2.csv`, `descriptives_v2.csv` and the figures: outputs of the analysis as reported in the manuscript.

Column definitions follow `scripts/11_extract_v2.py`. Durations are in seconds, intensities in dB, and `f0_k` is the number of the word's `f0_n` analysis frames that yielded a valid F0 estimate.

## Reproducing the analysis

**Models and figures only (no corpus needed).** The public tables are sufficient to re-run the models and figures:

```bash
Rscript scripts/12_models_v2.R
python3 scripts/13_figures_v2.py
```

This reproduces `output_v2/models_v2.txt` and `estimates_v2.csv` (estimates agree with the reported ones to within 1e-6).

**Full pipeline (licensed copy of Buckeye required).**

1. Obtain the corpus from <https://buckeyecorpus.osu.edu/>.
2. Make a flat directory of the `.txt`, `.words`, `.phones` and `.wav` files, for example with `scripts/link_raw.sh /path/to/buckeye`, which creates symbolic links in `raw/`.
3. Run `python3 scripts/11_extract_v2.py --raw raw --out output_v2`. On first use the script downloads SUBTLEX-US frequencies (Brysbaert & New, 2009), which are used only in a sensitivity analysis.
4. Run the two commands above.

To attach word labels and time stamps to the public table from your own copy of the corpus:

```bash
python3 scripts/restore_from_corpus.py --raw raw --data output_v2
```

`scripts/make_public_tables.py` is the script that produced the public tables from the full ones.

## Software

Analyses were run with Python 3.14 (pandas 3.0.3, NumPy 2.4.4, Parselmouth 0.4.7 / Praat 6.1.38) and R 4.6.1 (lme4 2.0.6, lmerTest 3.2.1, glmmTMB 1.1.14).

The random-effects structure of each model is selected automatically by `fit_max()` in `12_models_v2.R`, in a fixed order: maximal structure, then without by-speaker slopes, then without correlations among the by-word slopes, then random intercepts only. Models for the comparison with the fluent baseline are also reported with random intercepts only, as a sensitivity analysis (`SENSITIVITY` sections of `models_v2.txt`).

## Citation

Please cite the article (see `CITATION.cff`) and the corpus:

- Pitt, M. A., Dilley, L., Johnson, K., Kiesling, S., Raymond, W., Hume, E., & Fosler-Lussier, E. (2007). *Buckeye Corpus of Conversational Speech* (2nd release). Columbus, OH: Department of Psychology, Ohio State University (Distributor).
- Pitt, M. A., Johnson, K., Hume, E., Kiesling, S., & Raymond, W. (2005). The Buckeye corpus of conversational speech: Labeling conventions and a test of transcriber reliability. *Speech Communication, 45*(1), 89–95.

## Licence

Code: MIT License (`LICENSE`). Measurement tables and figures: CC BY 4.0. The Buckeye Corpus itself is not covered by these licences.
