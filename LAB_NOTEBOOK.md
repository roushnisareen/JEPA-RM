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
| **Total so far** | **≈ $17.80** |

This account had $19.81 of October credit, so ≈ $2 remains. Everything from here (GRPO) runs on
another Modal account. GRPO moves to another Modal account; it needs the same `huggingface` and `wandb`
secrets and the same HF repo, so it can find the reward models trained here.

## Open questions

- ~~Does best-of-n over-optimize BT by n = 64?~~ No (2026-10-15). Does GRPO, which pushes much harder?
- At lr 1e-4, does GRPO move the policy far enough (KL ≫ 0.001) to show hacking?
- Is vLLM worth it for the full GRPO runs (~$16 → ~$8 per run)?
