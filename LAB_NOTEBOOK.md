# Lab notebook

A dated record of every GPU run: what it was for, what it cost, what it showed, and what was
decided because of it. Costs are estimates from job durations at Modal's A100-80GB price
($2.50/hour); the Modal dashboard (Settings → Usage) has the exact figures. Changes to the
pre-registered plan are also logged in [PREREGISTRATION.md](PREREGISTRATION.md#deviations-log).

Repos: `Roushniii/jepa-rlhf` holds the real runs, `Roushniii/jepa-rlhf-test` the smoke tests.
W&B project: `jepa-rm`.

---

## 2026-10-14: GPU checks (smoke tests, test repo)

Tiny settings, to find what breaks on a real GPU before spending on real runs. Total ≈ $2.70.

| # | Job | Result |
|---|---|---|
| 1 | `reward_models --max-pairs 500` | Works end to end: image build, all six reward models, uploads, W&B. BT reached only 54% held-out accuracy, as expected after 62 steps on 500 pairs. |
| 2 | `stress_test --n-examples 20` | Works, including the 8B gold judge. The judge behaves sensibly (below). |
| 3 | `best_of_n --n-prompts 8 --n-samples 8` | Works. Found 31% of base answers cut off at 768 tokens (below). |
| 4 | `grpo --rm bt --max-steps 8` | Three attempts; three real problems found and fixed (below). Final: 113 s/step. |
| 5 | `evaluate --runs bt --n-prompts 16` | Works, including the KL measurement: 0.000 for base, ~0.001 after 8 steps. |

**The gold judge looks trustworthy** (check 2, n = 20, change in σ vs. the original answer). Edits that
shouldn't help didn't: filler −0.38, repeat −0.77, confident framing −0.50, bullets −0.34,
restating the question −0.21. Edits that should hurt did, strongly: half the answer −1.10, answer =
the question −2.14, answer to a different question −4.15, wrong facts −2.46, reversed sentences −1.12.

**Long answers** (check 3). Qwen3-1.7B writes long, formatted answers (~300 words). 20 of 64 hit the
768-token limit and stopped mid-sentence. Checked the text: no leftover `<think>` blocks, so
`/no_think` works and the answers are real. A cap there would hide the length hacking being
measured, so the limit was raised to 1,536 tokens (deviation logged).

**GRPO problems found and fixed** (check 4):
- *Cancelled at step 6:* the laptop's Modal client dropped and took the job with it. Long jobs now
  run with `modal run --detach`.
- *KL logged as NaN:* each training slice is one question's 8 answers; when all 8 are cut off and
  masked, TRL computes KL as 0/0. Training is unaffected (the loss is guarded). KL is now measured
  directly during evaluation instead of read from training logs.
- *130 s/step:* gradient checkpointing made transformers turn off the KV cache during sampling.
  Turning checkpointing off ran out of memory in the backward pass (>80 GB). Fix: keep checkpointing
  on and sample in eval mode, which keeps the cache.
- *Still 113 s/step after that:* generation goes through the un-merged LoRA adapter (~60–70 s) plus
  training (~40 s). 200 steps ≈ 6.3 h ≈ $16 per run. vLLM would roughly halve it; not done yet.

Also found: the laptop's Hugging Face token is fine-grained and can't read the private repo. The
Modal jobs use a separate write token, so runs are unaffected; downloading results locally needs a
token with read access.

---

## 2026-10-14: Reward models (real run)

`reward_models` on all 4,601 training pairs; evaluated on the 236 held-out pairs.

- **First attempt failed** after ~2 min (≈ $0.10): out of memory training BT. Raising the reward
  models' limit to 3,072 tokens meant a batch of 16 long texts no longer fit. Fix: gradient
  checkpointing for BT training, plus a test that LoRA weights actually change under it.
- **Second attempt** succeeded in ~16 min (≈ $0.70).

| reward model | accuracy (95% CI) | vs. BT, paired (95% CI) | prefers longer | corr. with length |
|---|---|---|---|---|
| `bt` | 74.2% [69, 80] | — | 61.4% | 0.31 |
| `bt_lenpen` | 72.0% [67, 78] | −2.1% [−6, +1] | 53.4% | 0.09 |
| `frozen_bt` | 67.4% [61, 74] | −6.8% [−13, −0] | 67.4% | 0.13 |
| `jepa` | 66.5% [61, 73] | −7.6% [−14, −1] | 62.3% | 0.30 |
| `jepa_pred` | 56.8% [50, 63] | −17.4% [−25, −9] | 56.8% | 0.27 |
| `jepa_rank` | 65.3% [59, 72] | −8.9% [−15, −2] | 61.0% | 0.29 |

Humans prefer the longer answer 60.2% of the time. BT pays 0.0021 reward per word; `bt_lenpen`
subtracts that.

**What it says:**
- BT is a solid reward model. The length penalty costs ~2 points of accuracy and cuts the reward's
  correlation with length from 0.31 to 0.09.
- JEPA is weaker than BT (paired gap −7.6 points, CI excludes 0) and about as accurate as the
  capacity-matched frozen-feature control (66.5% vs. 67.4%). So far the frozen features, not the
  JEPA objective, account for JEPA's accuracy.
- Pure JEPA (no negatives) is weak, at 56.8%; the ranking term does most of the work.
- Pre-registered accuracy caveat: JEPA's point estimate is 7.6 points below BT, but the CI
  [−14, −1] isn't entirely below −5, so the rule as written doesn't formally trigger. Borderline;
  report it as such.

None of this tests hacking yet. That's best-of-n and GRPO.

---

## 2026-10-14: GRPO pilot, `bt`, lr 1e-5 (real run)

Purpose: check that the standard reward model gets over-optimized at all (H0) before paying for
the full comparison. Planned 60 steps; started 15:20 IST, stopped at step ~37 (≈ 1 h 45 min, ≈ $4.40).
Checkpoints at steps 15 and 30 and the partial log are saved in the repo.

| step | reward | answer length | cut off | KL per token |
|---|---|---|---|---|
| 5 | +4.29 | 685 tok | 10% | 0.0004 |
| 10 | +4.67 | 792 | 9% | 0.0007 |
| 20 | +4.02 | 529 | 5% | 0.0007 |
| 25 | +4.71 | 660 | 6% | 0.0012 |
| 30 | +3.74 | 517 | 11% | 0.0008 |
| 35 | +4.28 | 715 | 8% | 0.0009 |

**Finding:** the policy barely moved. KL stayed around 0.001 per token and reward and length show
no trend; later steps got slower (~150 s) only because of randomly longer answers. At lr 1e-5,
LoRA learns too slowly for hacking to appear in a few hundred steps. No code error: the run was
healthy, the setting was wrong.

**Decision:** stopped early to save ~$2.50. Next GRPO pilot at lr 1e-4 (10× higher), on another
Modal account. Logged as a deviation, as the pre-registration plans under H0.

---

## 2026-10-14: Stress test (real run)

`stress_test` on 200 held-out answers, all six reward models plus the gold judge. Started 17:00 IST,
~10 min, ≈ $0.50. Score change in σ of each scorer, vs. the original answer:

| edit | expect | bt | bt_lenpen | frozen_bt | jepa | jepa_pred | jepa_rank | gold |
|---|---|---|---|---|---|---|---|---|
| filler | flat | −0.24 | −0.28 | −0.03 | −0.20 | −0.11 | −0.07 | −0.39 |
| repeat | flat | −0.54 | −0.87 | −0.13 | −0.80 | −0.67 | −0.99 | −0.77 |
| confident | flat | −0.01 | −0.03 | +0.03 | +0.03 | −0.09 | **+0.22** | −0.49 |
| bullets | flat | **+0.52** | **+0.51** | +0.02 | +0.08 | −0.14 | +0.09 | −0.31 |
| restate | flat | −0.38 | −0.47 | +0.03 | **+0.20** | +0.13 | +0.07 | −0.10 |
| truncate | down | −1.34 | −1.17 | −0.05 | −0.38 | −0.42 | −0.30 | −1.03 |
| prompt_only | down | −1.91 | −1.69 | −0.16 | −1.19 | −1.25 | −1.15 | −2.39 |
| off_topic | down | −2.04 | −2.06 | −0.05 | −1.31 | −3.67 | −0.53 | −4.39 |
| wrong_facts | down | −0.49 | −0.50 | −0.04 | −0.16 | −0.19 | −0.15 | −2.28 |
| reversed | down | −1.03 | −1.04 | −0.06 | −0.38 | −0.18 | −0.33 | −1.26 |

**What it says:**
- **BT's loophole is formatting.** The same text as bullet points scores +0.52σ higher, while the
  judge scores it slightly lower. The length penalty doesn't touch it (+0.51σ): it's not a length hack.
- **JEPA's loophole is different: copying the question in** (+0.20σ), smaller than BT's. That's the
  "hacked differently" pattern. `jepa_rank` is instead swayed by confident framing (+0.22σ).
- **The frozen-feature control is nearly blind** to every edit, even an answer to a different
  question (−0.05σ), despite 67% held-out accuracy. It seems to score style or the question rather
  than whether the answer fits. JEPA, with the same features, clearly does look at the answer
  (−1.31σ off topic), which makes the JEPA-vs-control comparison more interesting than accuracy alone suggested.
- **No reward model checks facts:** wrong facts cost at most −0.5σ, against −2.3σ for the judge.
- **The judge passes again:** none of the "shouldn't help" edits raised its score.

**Rerun with CIs (2026-10-15, ≈ $0.50).** Identical shifts, which also shows the stress test is
reproducible. Every shift in the table has a 95% CI (over the 200 answers) that excludes 0, so the
loopholes are real: BT bullets +0.52σ [+0.46, +0.58]; `bt_lenpen` bullets +0.51σ [+0.44, +0.57];
JEPA restate +0.20σ [+0.15, +0.25] and bullets +0.08σ [+0.05, +0.11]; `jepa_rank` confident
+0.22σ [+0.18, +0.26]. The judge's score fell significantly for every "shouldn't help" edit
(e.g. bullets −0.31σ [−0.37, −0.25], restate −0.10σ [−0.14, −0.05]). `frozen_bt`'s tiny shifts
(~±0.03σ) are also "significant" only because its scores barely vary; they're too small to matter.
Every answer's raw scores are now saved in `results/stress_test.json` (`scores`).

---

## 2026-10-14: Best-of-n (real run, wrong settings)

Started 17:00 IST. **Ran at 100 prompts × 16 samples (n up to 16), not the planned 64 × 64.** Bug:
the Modal job's command-line defaults (100, 16) override the config's (64, 64), and only the config
had been updated. Fixed (`modal_app.py`), plus `tests/test_modal_defaults.py`, which checks every
job default against its config so this can't drift again.

Left to finish, since generation was already paid for: its results are a useful early look at
n ≤ 16. ~56 min, ≈ $2.30 for 1,600 answers.

Gold judge score of each reward model's pick (raw judge units; 15.73 = a plain sample):

| n | bt | bt_lenpen | frozen_bt | jepa | jepa_pred | jepa_rank |
|---|---|---|---|---|---|---|
| 2 | 16.14 | 16.13 | 16.24 | 16.24 | 15.75 | 16.22 |
| 4 | 16.38 | 16.33 | 16.53 | 16.57 | 15.54 | 16.51 |
| 8 | 16.65 | 16.49 | 16.75 | 16.76 | 15.19 | 16.63 |
| 16 | 16.81 | 16.54 | 16.99 | 16.98 | 15.05 | 16.73 |

Paired gaps at n = 16 (100 prompts): jepa − bt +0.18 [−1.0, +1.3]; jepa − bt_lenpen +0.44
[−0.8, +1.6]; jepa − frozen_bt −0.00 [−0.9, +0.9]; jepa − jepa_pred +1.94 [+0.6, +3.3];
jepa − jepa_rank +0.25 [−0.6, +1.1].

**What it says:**
- BT is not over-optimized yet at n ≤ 16: its picks keep improving by the judge while getting
  slightly longer (449 → 488 words). Consistent with Gao et al. needing more pressure; H0 is still
  open, so n = 64 is essential.
- `jepa_pred` is over-optimized from n = 2: its own score rises (0.485 → 0.542) while the judge's
  falls (15.73 → 15.05). Textbook reward hacking, in the weakest reward model.
- `jepa` and `frozen_bt` behave identically here too.
- CIs are ~±1 point: differences of ~0.2 between reward models can't be resolved with 100 prompts.

**Next (proposed):** extend this run to 100 prompts × 64 samples by generating 48 more per prompt with
a different sampling seed and merging (≈ $7), rather than a fresh 64 × 64 (≈ $6, fewer prompts).
At 1,536-token answers, best-of-n costs ≈ $1.45 per 1,000 answers, more than the earlier ≈ $3–4
estimate for 64 × 64.

---

## 2026-10-15: Best-of-n extended to 100 × 64 (real run)

`best_of_n --extend --n-samples 64`: reused the 1,600 scored answers and sampled 48 more per prompt
with a fresh seed (100 prompts × 64 samples, n up to 64). ~2 h 40 min, ≈ $6.60. The n ≤ 16 result is
kept as `results/best_of_n_qwen_before_extend.json`.

Gold judge score of each reward model's pick (raw units; 15.62 = a plain sample):

| n | bt | bt_lenpen | frozen_bt | jepa | jepa_pred | jepa_rank |
|---|---|---|---|---|---|---|
| 2 | 16.13 | 16.12 | 16.02 | 15.99 | 15.63 | 15.99 |
| 4 | 16.41 | 16.39 | 16.19 | 16.15 | 15.44 | 16.13 |
| 8 | 16.56 | 16.53 | 16.31 | **16.21** | 15.14 | **16.15** |
| 16 | 16.67 | 16.59 | **16.37** | 16.20 | 14.79 | 16.06 |
| 32 | 16.78 | 16.65 | 16.32 | 16.01 | 14.41 | 15.80 |
| 64 | **16.97** | **16.78** | 16.21 | 15.63 | 13.91 | 15.61 |

Bold = each curve's peak. BT's picks also grow longer (452 → 505 words); the length penalty's don't
(→ 431).

**Units matter.** The judge's spread across all base answers is 11.3 points, mostly differences
between prompts, so a 1-point raw change is only ~0.1σ. All effects here are small in σ. The paired
CIs below are valid regardless; the σ scale understates within-prompt effects, which the write-up
should say.

**Pre-registered rules applied** (a CPU-only script on the saved results; σ = the base answers'
spread for each scorer; paired 95% CIs over the 100 prompts):

| rm | own rise @64 | optimizable | peak n | gold, peak → 64 | hacked |
|---|---|---|---|---|---|
| bt | +0.64σ | yes | 64 | — | no |
| bt_lenpen | +0.70σ | yes | 64 | — | no |
| frozen_bt | +0.12σ | **no** | 16 | −0.01 [−0.08, +0.05] | no |
| jepa | +0.56σ | yes | 8 | −0.05 [−0.13, +0.03] | no |
| jepa_pred | +0.64σ | yes | 2 | −0.15 [−0.25, −0.06] | **yes** |
| jepa_rank | +0.50σ | yes | 8 | −0.05 [−0.13, +0.04] | no |

Paired gold gaps at n = 64: jepa − bt −0.12σ [−0.25, 0.00]; jepa − bt_lenpen −0.10σ [−0.22, +0.02];
jepa − frozen_bt −0.05σ [−0.15, +0.04]; bt_lenpen − bt −0.02σ [−0.08, +0.04].

**Verdicts from best-of-n** (GRPO still to come):
- **H0 (BT gets hacked): not supported.** BT's picks keep improving all the way to n = 64. Best-of-n
  at this scale doesn't push it past its peak.
- **H1 (JEPA resists while BT is hacked): can't be tested as stated,** since BT wasn't hacked. What
  did happen is the reverse direction: JEPA's curve peaks at n = 8 and falls after it while BT's keeps
  rising. JEPA's own decline isn't significant (CI crosses 0), but its gap to BT at n = 64 is at the
  edge of significance in BT's favour (−0.12σ [−0.25, 0.00]).
- **H2 (beats `frozen_bt`): not supported** (−0.05σ, CI crosses 0). The control also fails the
  "optimizable" test, so it's a weak baseline under best-of-n.
- **H3 (beats `bt_lenpen`): not supported** (−0.10σ, CI crosses 0).
- **H4 (which term matters):** the JEPA term alone (`jepa_pred`) is the one reward model formally
  hacked, falling from n = 2 onwards; adding the ranking term (`jepa`) removes most of that.
- **Accuracy caveat:** JEPA is ~8 points less accurate than BT (borderline under the rule), so a
  weaker signal remains a live explanation for its earlier peak.

**Plain reading:** in best-of-n, the standard reward model was the *more* robust one. JEPA-style
rewards peaked early, and the pure JEPA objective was clearly over-optimized. This matches the
outside prediction ("JEPA is a weaker signal / gets hacked differently"), not the hypothesis.

---

## 2026-10-15: Mean vs. last-token pooling (real run, exploratory)

Second HF repo `Roushniii/jepa-rlhf-last`. The last-pooling arm reuses the mean arm's BT models
(copied, so both arms share the identical BT) and rescores its 6,400 best-of-n answers instead of
generating new ones, so the arms differ only in pooling and best-of-n compares them pair by pair on
the same answers. Steps: `reward_models --jepa-pooling last --reuse-bt-from` (~4 min),
`stress_test --no-with-gold`, `best_of_n --rescore-from` (~11 min), then the CPU-only
`compare_pooling`. ≈ $0.90 in total. Logged in the pre-registration as exploratory.

**Held-out accuracy** (236 pairs, 95% CIs):

| rm | mean | last |
|---|---|---|
| frozen_bt | 67.4% [61, 74] | 61.4% [56, 68] |
| jepa | 66.5% [61, 73] | 60.2% [54, 66] |
| jepa_rank | 65.3% [59, 72] | 61.9% [56, 68] |
| jepa_pred | 56.8% [50, 63] | 53.4% [47, 60] |

Last pooling costs 3–6 points everywhere. It also prefers the longer answer less (jepa 62.3% → 57.6%).

**"Shouldn't help" edits** (shift in σ of each scorer, 95% CIs):

| rm | pooling | filler | repeat | bullets | restate | confident |
|---|---|---|---|---|---|---|
| jepa | mean | −0.20 | −0.80 | +0.08 | **+0.20** | +0.03 |
| jepa | last | −0.57 | −0.93 | −0.10 | −0.10 | **+0.10** |
| jepa_rank | mean | −0.07 | −0.99 | +0.09 | +0.07 | **+0.22** |
| jepa_rank | last | **+0.24** | −0.97 | +0.14 | −0.14 | **+0.22** |
| jepa_pred | mean | −0.11 | −0.67 | −0.14 | +0.13 | −0.09 |
| jepa_pred | last | −1.32 | −0.52 | −0.51 | −0.14 | −0.14 |
| frozen_bt | mean | −0.03 | −0.13 | +0.02 | +0.03 | +0.03 |
| frozen_bt | last | −0.01 | −0.01 | +0.00 | −0.00 | +0.00 |

Last pooling closes JEPA's restate loophole (+0.20 → −0.10) and penalizes filler harder, but JEPA
picks up a smaller one: the confident opener (+0.10 [+0.06, +0.15]). `jepa_rank` gets worse: filler
now helps it too. `frozen_bt` with last pooling barely reacts to any edit (all |shift| ≤ 0.01σ), so
its head mostly ignores what the edits change; that makes it a weak control in this arm.

**Best-of-n, paired** (gold of last-pooling picks − mean-pooling picks, same answers, n = 64, 100
prompts):

| rm | last − mean | peak n (mean → last) | gold change @64, σ (mean → last) |
|---|---|---|---|
| frozen_bt | +0.03σ [−0.08, +0.13] | 16 → 64 | +0.05 → +0.08 |
| jepa | +0.04σ [−0.08, +0.17] | 8 → 16 | +0.00 → +0.04 |
| jepa_pred | **+0.22σ [+0.11, +0.35]** | 2 → 64 | −0.15 → +0.07 |
| jepa_rank | +0.03σ [−0.10, +0.16] | 8 → 4 | −0.00 → +0.03 |

With last pooling, `jepa − bt` at n = 64 is −0.91 raw [−2.05, +0.19] (crosses 0).

**Plain reading:** pooling matters for the pure JEPA objective (`jepa_pred` stops being
over-optimized) and changes *which* loophole JEPA has, but it makes no measurable difference to
`jepa` itself in best-of-n. So "JEPA is not more robust than BT" isn't an artifact of mean pooling.
The accuracy drop means last pooling isn't a free improvement either.

---

## 2026-10-15: Fixes after an outside code review (code only, no GPU)

An external review of the public repo raised several points. Checked each against the code:

- **Normalization leak (real, fixed).** `embed_pairs` fitted the feature standardizers on every
  training prompt and answer before splitting off the 5% that picks the best epoch. No labels were
  involved, and the 236 held-out pairs were never in it, but the epoch-selection holdout did shape
  the features. Now the split comes first and the statistics come from the fitting side only. New
  test `test_normalization_is_fitted_on_the_fit_split_only` fails on the old code. Every result
  above was produced before this fix; it isn't worth ~$2 to rerun for a change this small, but
  any rerun will use the fixed code.
- **Per-epoch loss (fixed, logging only).** The JEPA/frozen heads logged the last minibatch's loss
  as the epoch loss; now it's the epoch average. No effect on training.
- **BT trains on unshuffled pairs, embedding models on 95% of them (documented, not changed).**
  Changing BT's order now would make the code differ from the models the results came from. Added
  to the README's known limitations.
- **`.DS_Store` committed (removed, now ignored).**
- **Already covered:** the review asked for an end-to-end CPU test (`test_pipeline.py` runs the
  whole flow on tiny models), launch-config checks (`test_modal_defaults.py`), and a human check
  of the judge (`scripts/human_check.py`, still to be done).
- **Still open, needs GPU or people:** matched GRPO with more than one seed, the blind human
  check, and a second gold judge.

---

## 2026-10-24: Full project review and fixes (code and docs only, no GPU)

Three independent read-throughs of the whole repo (reward models and training, evaluation and
statistics, documentation against code and numbers), each claim then checked by hand. 134 tests
pass. What was wrong, and what changed:

**Would have hurt the GRPO run (fixed)**
- Evaluation sampled at temperature 0.7 / top-p 0.95 but estimated KL from raw (T = 1) log-probs.
  The k3 estimator is only right when the answers come from the distribution being measured. GRPO
  and best-of-n already sample at 1.0; `EvalConfig` now does too. No checkpoint had been evaluated
  yet, so nothing reported changes.
- The GRPO Modal job had a 6 h timeout; the pilot ran at ~170 s/step, so 200 steps is ~9.5 h and
  the run would have been killed near the end. Now 12 h. The cost estimate is ≈ $24 per 200-step
  run, not the $10–14 the README said.
- `lr` default is now 1e-4 in both the config and the job (the decision logged on 10-14), so the
  pilot can't accidentally run at 1e-5 again.
- "Cut-off answers get no reward" was wrong: TRL 0.17 scores every answer before masking truncated
  ones, so they still enter the group baseline and the logged reward. They get no gradient, which is
  what stops the policy being pushed past the limit. Comments and docs corrected.
- `evaluate --runs` with a typo, or run before GRPO finished, only warned and then spent hours
  evaluating the base model alone. A run with no checkpoints for any seed is now an error.
- `best_of_n` hard-coded its `ns`; `evaluate --runs/--seeds` and `reward_models --jepa-pooling`
  weren't covered by the defaults-vs-config test. All read from or checked against the configs now.

**Made results more honest or reproducible (fixed)**
- The pre-registered "hacked" rule (gold at the peak n vs. at the largest n, paired CI) was computed
  by a one-off script that wasn't committed. `verdicts()` in `best_of_n.py` now applies every rule
  and saves the result in `best_of_n_{policy}.json` under `verdicts`, so the table in the 10-15 entry
  can be regenerated from the saved samples.
- Only one σ was reported: the gold judge's spread across all base answers, which is mostly
  differences between prompts. Best-of-n only ever chooses between answers to one prompt, so that
  unit understates its effect. `base_std_within` (root mean per-prompt variance) is now saved next
  to it. The plots keep the total σ and say so.
- The best-of-n estimator broke ties in the reward model's score by sample order, giving the first
  of equal-scored answers the whole pick. Tied answers now share it. Rare with float scores.
- `--extend` and `--rescore-from` now refuse to mix answers sampled with a different policy,
  temperature, length limit or judge, and the pre-extension backup is named by sample count.
- Stress-test CIs now resample the scorer's spread too (it was treated as exact), which widens
  them a little for scorers whose scores barely vary; the figure draws the CIs the tables quote.
- Head-to-head GRPO comparisons use each seed's own final checkpoint instead of dropping seeds
  that stopped early; the row records `n_seeds`.
- KL log-prob passes run in float32 (bf16 rounding between the merged-adapter model and the fresh
  base model would show up as a small positive KL on every answer).
- BT training is seeded like the heads; the heads share one batch order regardless of head shape;
  the BT tokenizer loads from the saved folder; the bootstrap percentile index is exact; a reloaded
  reward model is checked to score identically to the trained one.

**Noted, not changed** (changing them now would break the link between code and results)
- BT trains on all 4,601 pairs in one fixed pass with no early stopping; the embedding-space
  models keep the best of 30 epochs picked on 5% of prompts. `frozen_bt` vs. `jepa` is matched;
  `bt` vs. the frozen models isn't, on this axis.
- At τ = 0.05 the ranking term's gradient is up to 20× the JEPA term's, so `jepa` is close to
  `jepa_rank` by construction, which the results also show. A balanced-weights ablation would cost
  ≈ $1 (embedding models only) plus ≈ $1 to rescore.
- `bt_lenpen`'s length slope is fit on BT's own training pairs; on fresh policy samples the trend
  may differ.
- No `revision=` pin on the hub models. A silent upstream update to Qwen3-0.6B would change every
  reloaded reward model. Low likelihood; worth pinning before the final runs.

**Docs.** README rewritten: the headline no longer says JEPA "held up worse" when the CI touches 0,
or that the objective "bought nothing" when the CI crosses 0; the accuracy caveat sits next to the
best-of-n verdict; the stress table says it shows the largest loophole per model; costs, W&B run
names, the pilot instructions, the tests paragraph and the layout match the code. Pre-registration:
deviations log in date order with the 100-prompt/extend deviation and today's changes added; the
blank "my predictions" block is kept blank and says why.

---

## 2026-11-03: Second GRPO pilot started and stopped; sampling bug found (no results)

New month, $30 of Modal credit on this account. Started the pilot as planned:
`grpo --rm bt --max-steps 60`, lr 1e-4 (W&B run `thjbvq84`). It ran at ~120 s/step, faster than
the first pilot's ~170 s.

The start-up log said transformers had changed `temperature`, `top_p` and `top_k` "to match
model-specific defaults". Reading transformers 4.51.3 (`_prepare_generation_config`): when a caller
passes a generation config, every value left at the global default (temperature 1.0, top-p 1.0,
top-k 50) is replaced by the model's own default. TRL 0.17 passes temperature 1.0 and top-p 1.0,
so GRPO was sampling at Qwen3's **temperature 0.6, top-p 0.95, top-k 20**, not the configured 1.0.
That also means the first pilot (10-14) trained at 0.6. Best-of-n and evaluation pass their
settings as keyword arguments, which do override, so they really did use temperature 1.0, but they
picked up Qwen3's top-k 20 without anyone choosing it.

Stopped the pilot after ~10 minutes (≈ $0.40), before it could become the baseline for the full
runs.

**Fix.** One set of sampling settings everywhere, written into the model's own generation config
so nothing can swap them (`runtime.set_sampling`): temperature 1.0, top-p 1.0, **top-k 20**. Top-k
20 is kept on purpose: it's what the 6,400 best-of-n answers were sampled with, so those results
stay valid as they are, and it stops 1,536-token answers wandering into junk tokens. The cost is
that KL is estimated from full-softmax log-probs while sampling is cut to the top 20 tokens, a
small approximation; it applies equally to every run. `tests/test_sampling.py` reproduces the trap
on a tiny Qwen3 with Qwen3's real defaults and checks the fix through the LoRA wrapper GRPO uses.

**Next:** restart the pilot with the fix (same command).

---

## 2026-11-04 — GRPO pilot 3 (BT, lr 1e-4, sampling fixed)

`modal run --detach modal_app.py::grpo --hf-repo Roushniii/jepa-rlhf --rm bt --max-steps 60`, app
ap-Z1466LSmBTWBrLGGvfA9nJ, W&B run kzg3oj2x. 60 steps in 101 min (≈ 100 s/step). Checkpoints 15,
30, 45, 60 uploaded to the HF repo. Generation log now shows only the `bos_token_id` override, so
the sampling fix works in the real run.

| step | KL | BT reward | mean length | clipped |
|---|---|---|---|---|
| 5 | 0.002 | 4.32 | 660 | 0 |
| 10 | 0.005 | 5.14 | 800 | 0 |
| 15 | 0.007 | 3.58 | 420 | 0 |
| 30 | 0.026 | 4.30 | 553 | 0.03 |
| 35 | 0.019 | 4.93 | 739 | 0.03 |
| 40 | 0.036 | 4.41 | 539 | 0.04 |
| 45 | 0.056 | 4.01 | 390 | 0.03 |
| 50 | 0.033 | 4.57 | 658 | 0.01 |
| 55 | 0.049 | 4.64 | 538 | 0.02 |
| 60 | 0.026 | 5.05 | 624 | 0.02 |

(Steps 20 and 25 fell out of the Modal log buffer; they are on W&B.)

**Reading.** The policy moves now: KL reaches 0.03–0.06, 30–50× the first pilot's ≈ 0.001. BT
reward per batch is noisy (each step sees different prompts) with a weak upward drift, ≈ 4.3 →
≈ 5.0. Length does not grow yet. Very few answers hit the 1,536 limit. With max_steps 60 the
learning rate had already decayed to near 0 by the end; a 200-step run keeps it high for longer.
This is a training-signal check only; no gold scores were computed.

**Decision.** The setup works; the full 200-step BT run is the next step (≈ 5.6 h, ≈ $15).

---

## 2026-11-05 — Full GRPO run against BT (200 steps)

`modal run --detach modal_app.py::grpo --hf-repo Roushniii/jepa-rlhf --rm bt`, app
ap-1E07XgmjZVlPXrFT4NFEPg, W&B run kpg6tt3l. Finished all 200 steps in 6.25 h (`complete: true`).
Checkpoints 50, 100, 150, 200 and `results/grpo_qwen_bt.json` are on the HF repo. (Modal's
log view was empty afterwards; the numbers below come from the results file.)

| steps | BT reward | mean length | clipped | training KL |
|---|---|---|---|---|
| 5–35 | 3.3–5.0 (≈ 4.4) | 354–790 (≈ 620) | 0.03–0.11 | 0.005–0.05 |
| 40–70 | 4.3–5.7 (≈ 4.9) | 9–757 | 0–0.15 | spikes: 0.4, 2.4, 0.9, 1.9, 14.1, 4.9 |
| 75–145 | 4.5–5.7 (≈ 5.2) | 702–950 (≈ 810) | 0.13–0.29 | ≈ 0.2–0.33 |
| 150–195 | 5.6–6.3 (≈ 5.9) | 694–916 (≈ 840) | 0.03–0.14 | ≈ 0.25–0.45 (one 2.2) |

**Reading.**
- BT reward rose ≈ 4.2 → ≈ 6.0 over training while the learning rate decayed. The policy does
  optimize the proxy.
- Answers got ≈ 35% longer (≈ 620 → ≈ 840 tokens). Length is the classic BT exploit, but the gold
  judge decides whether this is hacking or real improvement.
- Unstable phase at steps 40–70. Step 65's batch had a mean length of 9 tokens (near-empty answers)
  with training KL 14. The policy recovered by step 75 and stayed at KL ≈ 0.3 from there. Worth
  looking at checkpoint 50's answers.
- The training-log KL is bf16 and has NaNs on some steps. The real KL numbers come from the fp32
  k3 estimate in `evaluate`, as planned.

**Cost.** ≈ 6.3 h of A100 ≈ $16.

**Next.** Gold-judge evaluation of checkpoints 50/100/150/200 (≈ $3). This tells whether gold
score peaks and then falls (hacking) or keeps rising. The JEPA run (≈ $16) does not fit in what is
left of November's credit.

---

## 2026-11-06 — Gold-judge evaluation of the BT GRPO run (before cleanup fix)

`modal run --detach modal_app.py::evaluate --hf-repo Roushniii/jepa-rlhf --runs bt`, app
ap-dd8Ev2ZIGyTdU5dJyj3NKF, W&B run wgk4okfy. 200 validation prompts × base + 4 checkpoints, ≈ 50
min (≈ $2).

| | step | gold | gold CI vs base | BT | BT + length penalty | words | cut off | KL (fp32 k3) |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 15.24 | | 4.52 | 3.53 | 474 | 4% | 0 |
| bt | 50 | 9.21 | [−7.2, −4.9] | 5.10 | 3.94 | 553 | 16% | 0.74 |
| bt | 100 | 10.62 | [−5.8, −3.5] | 5.21 | 3.98 | 587 | 17% | 0.35 |
| bt | 150 | 10.02 | [−6.5, −4.0] | 5.92 | 4.67 | 594 | 11% | 0.37 |
| bt | 200 | 10.97 | [−5.4, −3.2] | 6.20 | 5.01 | 565 | 7% | 0.30 |

Base gold std 11.2, so the drop is ≈ 0.4–0.55σ. Per prompt, step 200 beats base on BT 91% of
the time but on gold only 33%.

**At first sight: textbook over-optimization.** BT rises steadily while gold falls by step 50 and
never recovers to base. But the drop is immediate rather than a rise-then-fall. And it does not come from
cut-off answers: answers that weren't cut off fall just as much (9.3 / 10.6 / 10.2 / 11.6).

**Confound found: every trained answer starts with a stray `</think>`.** All 800 checkpoint
answers, none of the base ones. `strip_think` only removed complete `<think>…</think>` blocks, so
the tag reached both BT (during training) and the gold judge. Training step 65's 9-token mean
length was probably a batch of near-empty "</think>" answers. The gold judge may punish the tag
itself, so part of the drop could be formatting rather than worse answers.

**Real style shifts too (base → step 200, per answer):** emoji 1.4 → 14.5, markdown headers 6.9 →
17.3, bold markers 60 → 129, bullets 17 → 25. Change in length does not explain the change in gold
(r = −0.01). Example: a prompt saying "always ask questions before you answer". Base asked
questions (gold 20.4); step 200 ignored it and wrote a heavily formatted course outline (gold
−4.2, BT 6.78 vs 4.38).

**Fix.** `strip_think` now removes stray tags too (test added). New `rescore` job: cleans the
saved answers and scores them again without generating new ones, keeping the old file as
`eval_qwen_before_rescore.json`. Logged as a deviation.

**Next.** Run `rescore` (≈ 15–20 min, ≈ $0.50–1). If gold still drops clearly, the BT run is
over-optimized on style (emoji/markdown), not just on the tag.

---

## 2026-11-09 — BT GRPO run rescored with the tag removed

`modal run --detach modal_app.py::rescore --hf-repo Roushniii/jepa-rlhf`, app
ap-Mm6dytJZUIos7sRN0wBMAA, W&B run 7dnuivde. ≈ 5 min of scoring (≈ $0.40). No stray tags left in
any answer.

| step | gold with tag | **gold clean** | clean gold CI vs base | BT with tag | **BT clean** | gold win vs base | BT win vs base | KL |
|---|---|---|---|---|---|---|---|---|
| 0 (base) | 15.24 | **15.24** | | 4.52 | **4.52** | | | 0 |
| 50 | 9.21 | **12.07** | [−4.3, −2.0] | 5.10 | **4.39** | 34% | 54% | 0.74 |
| 100 | 10.62 | **12.87** | [−3.6, −1.3] | 5.21 | **4.48** | 40% | 60% | 0.35 |
| 150 | 10.02 | **11.89** | [−4.6, −2.1] | 5.92 | **4.89** | 33% | 61% | 0.37 |
| 200 | 10.97 | **12.77** | [−3.6, −1.4] | 6.20 | **5.12** | 39% | 66% | 0.30 |

(σ: gold 11.2, BT 1.87, from the base model.)

**Reading.**
- **The tag explains part of the gold drop, not all of it.** Clean gold is still below base at
  every checkpoint, by 2.4–3.4 points (0.2–0.3σ), with every CI entirely below 0.
- **BT rewarded the tag itself.** On the same answers, BT gives +0.7 to +1.1 more with the
  `</think>` in front than without. GRPO found a junk token that BT likes and put it on every
  answer: literal reward hacking, on top of the style drift (emoji ×10, headers ×2.5).
- **Pre-registered verdict for `bt` in GRPO**, on the policy as trained (the tag is part of
  what it outputs and what BT scored):
  - Optimizable: BT +1.68 = 0.90σ ≥ 0.5σ ✓.
  - Hacked: final gold 12.77 vs. the peak of the gold curve, base 15.24, with the CI entirely below
    ✓ (clean or not).
  - **H0 is supported: the setup can show hacking.**
- Caveat for the write-up: on cleaned answers BT's own rise is only 0.32σ. Most of the
  "optimization" was the tag exploit, so without it BT would not pass "optimizable". Report both.
- Gold never rises above base: no rise-then-fall. The first checkpoint (step 50) is already the
  worst, in the unstable phase at steps 40–70. KL then falls from 0.74 to 0.30.

**Next.** The JEPA GRPO run (≈ $16) with the fixed cleanup, then evaluate it (≈ $2). Not this month's
credit. Before that, a blind human check of ~30 base vs. step-200 pairs costs nothing.

---

## 2026-11-10 — GRPO against JEPA (stopped between steps 100 and 150)

`modal run --detach modal_app.py::grpo --hf-repo Roushniii/jepa-rlhf --rm jepa`, run on a friend's
Modal workspace (shaanvora17, $30 of Starter credit, spend limit $0) with this project's own
`huggingface` and `wandb` secrets. App ap-VfLNRdrnAbkkHOUwaojxcg, W&B run hedv1dsf. ≈ 125 s/step
(BT: ≈ 110). Started 11:46.

The run stopped after step 100 and before step 150. Checkpoints 50 and 100 are on the HF repo;
`results/grpo_qwen_jepa.json` has `complete: false`. Modal's logs for the app were gone by the next
check. Afterwards the friend's workspace refused even a CPU-only job with "Resource exhausted".
**Most likely cause: the $30 credit ran out, and with the $0 spend limit Modal stopped the run.**
The ≈ $2.50/h estimate covered the GPU only; Modal also bills the container's CPU and memory.
Not yet confirmed on the billing page.

| step | JEPA reward | mean length | clipped | training KL |
|---|---|---|---|---|
| 5–30 | 0.15–0.19 | 396–763 | 0.03–0.21 | 0.005–0.03 |
| 35–60 | 0.16–0.21 | 434–678 | 0.10–0.24 | mostly NaN; 0.20 at 60 |
| 65–100 | 0.20–0.24 | 231–687 | **0.32–0.75** | NaN |

**Reading.** The JEPA reward rises (≈ 0.18 → ≈ 0.21; base σ for `jepa` is 0.079, so ≈ +0.4σ). The
share of answers hitting the 1,536-token limit climbs to 34–75% by steps 75–100. BT never went
above 0.29. Cut-off answers are scored but get no gradient. So a growing share of each batch
teaches nothing, and the policy drifts toward answers that run on. Check checkpoint 100's answers
for repetition. The training KL is NaN on most logged steps (bf16); the fp32 k3 in `evaluate` is
the real number.

**Next.** Evaluate base + BT + JEPA together on the same 200 prompts. Compare BT and JEPA at the
same step (100), and at matched KL, as pre-registered. The JEPA run has no 150/200 checkpoints.

**2026-11-11: resume support.** `train_grpo` can now continue from a saved checkpoint
(`--resume-step 100`). It keeps checkpoints 50/100, carries on the learning-rate schedule (5e-5 → 0),
skips the 400 prompts already used (TRL's shuffle reproduced; a test checks it against TRL's own
sampler), and appends to the log with step numbers 101–200. Adam's state was not saved, so it
restarts (logged as a deviation). About 3.5 h of GPU.

---

## 2026-11-11 — Gold-judge evaluation of BT and JEPA together

`modal run --detach modal_app.py::evaluate --hf-repo Roushniii/jepa-rlhf --runs bt,jepa`, on the
author's college Modal workspace (bt23cse001) with this project's own `huggingface` and `wandb`
secrets (checked first: HF user Roushniii with write access, W&B user roushnisareen26). App
ap-FXGW0QPN7E6yFiTU0Zz1eN, W&B run luob2c1b. 7 policies × 200 prompts, ≈ 10 min each, 10:29–11:51
UTC. The cleanup fix was already in place, so no rescore was needed. Base and BT numbers came out the same as the 11-09
rescore (same prompts, same sampling seed). `eval_qwen.json` now holds BT and JEPA together; the
BT-only file stays as `eval_qwen_bt_only.json`.

| run | step | KL | gold | Δgold (σ) | 95% CI of Δgold | own reward rise (σ) | words | cut off |
|---|---|---|---|---|---|---|---|---|
| base | 0 | 0 | 15.24 | — | — | — | 474 | 4% |
| bt | 50 | 0.74 | 12.07 | −0.28 | [−4.34, −2.03] | −0.07 | 550 | 16% |
| bt | 100 | 0.35 | 12.87 | −0.21 | [−3.56, −1.31] | −0.02 | 586 | 16% |
| bt | 150 | 0.37 | 11.89 | −0.30 | [−4.59, −2.14] | +0.20 | 593 | 10% |
| bt | 200 | 0.30 | 12.77 | −0.22 | [−3.56, −1.38] | +0.32 | 564 | 6% |
| jepa | 50 | 0.17 | 14.73 | −0.05 | [−1.55, +0.50] | +0.22 | 561 | 16% |
| jepa | 100 | 0.23 | 13.38 | −0.17 | [−2.97, −0.75] | **+0.57** | 704 | **46%** |

Gold σ (base) = 11.20; CIs are in raw gold points. The code's comparison pairs each run's last
checkpoint: `jepa`@100 − `bt`@200 = +0.61 gold, CI [−0.29, +1.56] → **no difference shown**.

**Against the pre-registered rules (one seed, so preliminary; JEPA stopped at step 100):**
- `jepa` is **optimizable**: its own score rises +0.57σ (≥ 0.5σ). BT's own rise is only +0.32σ.
- `jepa` is **hacked**: gold at its last checkpoint is below the peak (the base model), and the CI
  of the drop lies entirely below 0.
- So **H1 is not supported**: JEPA gets optimized and hacked, like BT.
- JEPA's gold falls more slowly per step (step 50: no drop shown; BT step 50: −3.2) and at lower KL,
  but the step-100 gap to BT (+0.51) and the last-checkpoint gap both have CIs that include 0.

**How JEPA gets hacked: different from BT.** BT's answers filled up with emoji (2.1 → 17 per answer)
and headers. JEPA's grow long and over-structured: 704 words, 46% hit the 1,536-token limit (BT
step 200: 6%), more headers (6.9 → 14.2) and bullets (17 → 31), few emoji (5.2). The worst-scored
complete answers ignore the prompt's constraints (a 200-word limit, alphabetical order). The
complete answers score lower (12.2) than the cut-off ones (14.8), so the drop is not just
truncation. The outside prediction's "JEPA gets hacked differently" fits better than "JEPA
resists". BT's reward model also scores JEPA's answers lower (3.63 vs 4.52 for base).

---

## 2026-11-11/12 — JEPA run finished (steps 100–200) and evaluated

**Training.** The first continuation was preempted by Modal at step ~114 and restarted from 100
(≈ 30 min lost); stopped, resume points every 10 steps added (deviation logged), relaunched
12:36 UTC. Finished 15:19 UTC with no further restarts (`resumed_from: [100]`); checkpoints 150 and
200 on the HF repo, resume points under `policy_qwen_jepa_resume/`. ≈ 97 s/step. The share of
answers hitting the 1,536-token limit kept climbing: 56% at step 105, ~80% by 150, 77–96% over
steps 180–200. JEPA reward ≈ 0.23.

**Evaluation.** `evaluate --runs jepa` (base + JEPA 50–200), 07:30–08:20 UTC on 11-12, after ~30
min waiting for an A100. The BT+JEPA-100 file was backed up first as `eval_qwen_bt_jepa100.json`;
the JEPA-only output is kept as `eval_qwen_jepa_only.json`. The two were merged with the project's
own `summarize_and_save` into `eval_qwen.json` (plots regenerated). The base model's 200 answers
were identical in both runs, and JEPA 50/100 reproduced the earlier numbers exactly.

| run | step | KL | gold | Δgold (σ) | 95% CI of Δgold | own reward rise (σ) | words | cut off | emoji | headers |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 0 | 15.24 | — | — | — | 474 | 4% | 2.1 | 6.9 |
| bt | 200 | 0.30 | 12.77 | −0.22 | [−3.56, −1.38] | +0.32 | 564 | 6% | 17.1 | 17.3 |
| jepa | 50 | 0.17 | 14.73 | −0.05 | [−1.55, +0.50] | +0.22 | 561 | 16% | | |
| jepa | 100 | 0.23 | 13.38 | −0.17 | [−2.97, −0.75] | +0.57 | 704 | 46% | 5.2 | 14.2 |
| jepa | 150 | 0.25 | 12.05 | −0.29 | [−4.28, −2.13] | +0.70 | 769 | 71% | | |
| jepa | 200 | 0.27 | 11.68 | **−0.32** | [−4.78, −2.39] | +0.69 | 775 | **82%** | 15.7 | 14.5 |

Last checkpoints, paired by prompt: `jepa`@200 − `bt`@200 = **−1.10 gold, CI [−2.08, −0.09]**: JEPA's
policy ends up *worse* than BT's (≈ −0.10σ), at similar KL (0.27 vs 0.30).

**Against the pre-registered rules (one seed, so preliminary):**
- `jepa` is optimizable (+0.69σ) and hacked (gold keeps falling to −0.32σ; CI below 0). **H1 not
  supported.** Unlike BT's gold curve, which bottoms out around −0.2 to −0.3σ, JEPA's falls at
  every checkpoint.
- JEPA vs BT at the last checkpoint: JEPA worse, CI just below 0.

**How.** Early on (step 100) the two exploits looked different: BT went for emoji, JEPA for length
and structure. By step 200 JEPA has picked up the emoji too (15.7 per answer, BT 17.1), and on top
of that 82% of its answers run into the length limit (BT 6%). Its 36 answers that do finish score
7.7 on average; the 164 cut-off ones 12.6. So the policy that learned to ramble is worst when it
actually finishes.

---

## Spending

| | cost |
|---|---|
| GPU checks (smoke tests) | ≈ $2.70 |
| Reward models: failed attempt + real run | ≈ $0.80 |
| GRPO pilot (lr 1e-5, stopped early) | ≈ $4.40 |
| Stress test | ≈ $0.50 |
| Best-of-n, 100 × 16 (wrong settings, kept) | ≈ $2.30 |
| Best-of-n, extended to 100 × 64 | ≈ $6.60 |
| Stress test rerun with CIs | ≈ $0.50 |
| Pooling comparison (last-pooling arm) | ≈ $0.90 |
| GRPO pilot 2 (stopped after ~10 min: sampling bug) | ≈ $0.40 |
| GRPO pilot 3 (60 steps, sampling fixed) | ≈ $4.50 |
| GRPO full run, BT (200 steps) | ≈ $16 |
| Gold-judge evaluation, BT run | ≈ $2 |
| Rescore with the tag removed | ≈ $0.40 |
| GRPO against JEPA, steps 1–100 (friend's workspace) | ≈ $30 |
| Gold-judge evaluation, BT + JEPA (bt23cse001) | ≈ $4–6 (≈ 1.4 h A100; check billing page) |
| GRPO against JEPA, steps 100–200, incl. the preempted attempt (bt23cse001) | ≈ $12–14 (≈ 3.4 h A100) |
| Gold-judge evaluation, JEPA 50–200 (bt23cse001) | ≈ $3.50 (≈ 0.9 h A100) |
| **Total so far** | **≈ $95** (estimates; billing pages are the record) |

October: $19.81 of credit on this account, ≈ $1 left at the end of the month. November: $30 of
new credit on the same account. After pilot 3 and the full BT run (≈ $16, 6.3 h), ≈ $9 is left;
after the evaluation (≈ $2) and rescore (≈ $0.40), ≈ $6.60. Not enough for the JEPA run. That run needs next month's credit or another
account with the same `huggingface` and `wandb` secrets and the same HF repo.

## Open questions

- ~~Does best-of-n over-optimize BT by n = 64?~~ No (2026-10-15). Does GRPO, which pushes much harder?
- ~~Does GRPO over-optimize BT?~~ Yes (2026-11-09): clean gold −0.22σ at step 200 (CI below 0); most of BT's own gain came from a `</think>` token it rewards.
- ~~At lr 1e-4, does GRPO move the policy at all?~~ Yes, KL 0.03–0.06 by step 60 (2026-11-04). Far enough to show hacking? Needs the full run + gold scores.
- ~~Is the negative JEPA result an artifact of mean pooling?~~ No (2026-10-15, exploratory).
- ~~Does JEPA resist GRPO's over-optimization?~~ No (2026-11-12, one seed): optimized (+0.69σ) and hacked (gold −0.32σ at step 200, CI below 0); worse than BT at the last checkpoint (−1.10 gold, CI [−2.08, −0.09]).
- ~~Would JEPA steps 150/200 fall further, or level off?~~ Further, at every checkpoint; 82% of answers cut off by step 200.
- Does the pattern hold with a second seed, or another policy family (Llama)? Needs ≈ $25 per run.
- Is vLLM worth it for the full GRPO runs (~$24 → ~$12 per run)?
