# Retraced same-word restarts in the Buckeye Corpus: analysis code and measurements

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24139970.svg)](https://doi.org/10.5281/zenodo.24139970)

Archived on Zenodo: version 1.1.0, https://doi.org/10.5281/zenodo.24139970 (all versions: https://doi.org/10.5281/zenodo.24139969).

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
| `scripts/12b_baseline_event_v2.R` | Fluent-baseline models with a random intercept for repair event (the baseline estimates reported in the manuscript, Table 3 and Figure 2). |
| `scripts/12c_abut_sensitivity_v2.R` | Sensitivity analyses with and without the events in which the repetition directly follows the cutoff fragment. |
| `scripts/13_figures_v2.py` | Figures 1, 2 and S1. |
| `scripts/14_boundary_check_*.py`, `tools/boundary_check.html` | Listening check of 50 word boundaries (sampling, blind judging tool, summary). Requires the corpus. |
| `scripts/15_phone_check_v2.py` | Segment-level duration ratios from the corpus phone labels (summary values only). Requires the corpus. |
| `scripts/16_onset_kit.py`, `16_onset_serve.py`, `tools/onset_mark.html` | Blind re-marking of repetition onsets (kit and judging tool). Requires the corpus. |
| `scripts/17_formants_v2.py`, `17_formants_models_v2.R` | Vowel centralization: measurement (requires the corpus) and models. |
| `scripts/18_onset_apply_v2.py`, `18_onset_models_v2.R` | Re-measurement with the re-marked onsets (requires the corpus and the judgements) and models. |

## What is (and is not) in this repository

The Buckeye Corpus is licensed by The Ohio State University and may not be redistributed. This repository therefore contains **no audio, transcriptions, word labels or corpus time stamps**. It contains only the measurements made for this study (supplemental annotations):

- `output_v2/tokens.csv`: one row per analysed token (repair tokens and fluent baseline). Tokens are identified by recording (`basename`) and the 0-based index of the entry in that recording's `.words` file (`words_idx`). Word types are replaced by codes (`word` = `w001`…`w126`), assigned in the R collation order of the original labels.
- `output_v2/events.csv`: one row per candidate repair event, with event-level variables (fragment duration, cutoff-to-restart interval, retrace length, intervening words) and the selection status.
- `output_v2/tokens_onsetfix_v2.csv`, `tokens_onsetfix_cal_v2.csv`: the same rows as `tokens.csv`, with the measurements of the directly following repetitions taken from their hand-re-marked onsets (uncalibrated, and calibrated by the median shift of the control items). Coded in the same way as `tokens.csv`.
- `output_v2/formants_v2.csv`: vowel formants and the centralization index, keyed by the row of `tokens.csv` (`tok_row`, 0-based). Vowel labels are replaced by codes (`v01`…), with a `diphthong` flag.
- `output_v2/selection_counts.csv`, `models_v2.txt`, `estimates_v2.csv`, `descriptives_v2.csv`, `baseline_event_v2.*`, `abut_sensitivity_v2.*`, `formants_models_v2.*`, `onset_models_onsetfix{,_cal}_v2.*`, `onset_agreement{,_interim}_v2.txt`, `phone_check_v2.txt` and the figures: outputs of the analysis as reported in the manuscript.

The kits for the listening check and the re-marking contain audio clips and corpus time stamps and are therefore not included.

Column definitions follow `scripts/11_extract_v2.py`. Durations are in seconds, intensities in dB, and `f0_k` is the number of the word's `f0_n` analysis frames that yielded a valid F0 estimate.

## Reproducing the analysis

**Models and figures only (no corpus needed).** The public tables are sufficient to re-run the models and figures:

```bash
Rscript scripts/12_models_v2.R
Rscript scripts/12b_baseline_event_v2.R
Rscript scripts/12c_abut_sensitivity_v2.R
Rscript scripts/17_formants_models_v2.R
Rscript scripts/18_onset_models_v2.R tokens_onsetfix_v2.csv
Rscript scripts/18_onset_models_v2.R tokens_onsetfix_cal_v2.csv
python3 scripts/13_figures_v2.py
```

These commands overwrite the model outputs in `output_v2/`. The estimates agree with the reported ones to within 1e-5, and the figures are identical.

**Full pipeline (licensed copy of Buckeye required).**

1. Obtain the corpus from <https://buckeyecorpus.osu.edu/>.
2. Make a flat directory of the `.txt`, `.words`, `.phones` and `.wav` files, for example with `scripts/link_raw.sh /path/to/buckeye`, which creates symbolic links in `raw/`.
3. Run `python3 scripts/11_extract_v2.py --raw raw --out output_v2`. On first use the script downloads SUBTLEX-US frequencies (Brysbaert & New, 2009), which are used only in a sensitivity analysis.
4. Run the commands above. `15_phone_check_v2.py` and `17_formants_v2.py` also run on the corpus; `18_onset_apply_v2.py` additionally needs your own re-marked onsets from `16_onset_kit.py` and `16_onset_serve.py`.

To attach word labels and time stamps to the public table from your own copy of the corpus:

```bash
python3 scripts/restore_from_corpus.py --raw raw --data output_v2
```

`scripts/make_public_tables.py` is the script that produced the public tables from the full ones.

## Software

Analyses were run with Python 3.14 (pandas 3.0.3, NumPy 2.4.4, Parselmouth 0.4.7 / Praat 6.1.38) and R 4.6.1 (lme4 2.0.6, lmerTest 3.2.1, glmmTMB 1.1.14).

The random-effects structure of each model is selected automatically by `fit_max()` in `12_models_v2.R`, in a fixed order: maximal structure, then without by-speaker slopes, then without correlations among the by-word slopes, then random intercepts only. Models for the comparison with the fluent baseline are also reported with random intercepts only, as a sensitivity analysis (`SENSITIVITY` sections of `models_v2.txt`).

## Citation

Please cite the article and this archive (see `CITATION.cff`), and the corpus:

- Ishihara, T. (2026). *Retraced same-word restarts in the Buckeye Corpus: Analysis code and measurements* (Version 1.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.24139970

- Pitt, M. A., Dilley, L., Johnson, K., Kiesling, S., Raymond, W., Hume, E., & Fosler-Lussier, E. (2007). *Buckeye Corpus of Conversational Speech* (2nd release). Columbus, OH: Department of Psychology, Ohio State University (Distributor).
- Pitt, M. A., Johnson, K., Hume, E., Kiesling, S., & Raymond, W. (2005). The Buckeye corpus of conversational speech: Labeling conventions and a test of transcriber reliability. *Speech Communication, 45*(1), 89–95.

## Licence

Code: MIT License (`LICENSE`). Measurement tables and figures: CC BY 4.0. The Buckeye Corpus itself is not covered by these licences.
