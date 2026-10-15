# Pre-registration

Fill in the blanks marked `___`, then commit this file **before** running any GPU job. The commit
date is the record that the hypotheses and decision rules came before the data. Anything changed
afterwards goes in the deviations log at the bottom, with the reason, not silently into this text.

## Question

Does a JEPA-style reward model resist reward hacking better than a standard Bradley-Terry reward
model, and if so, is it the JEPA objective that does it?

## What gets compared

| kind | what it is | role |
|---|---|---|
| `bt` | Qwen3-0.6B + LoRA + scalar head, Bradley-Terry loss | standard reward model |
| `bt_lenpen` | `bt` with its linear length trend removed | the cheap known fix |
| `frozen_bt` | frozen encoder + MLP head, Bradley-Terry loss | control: same features and capacity as JEPA |
| `jepa` | frozen encoder + predictor, JEPA + ranking loss | the idea under test |
| `jepa_pred`, `jepa_rank` | JEPA loss only / ranking loss only | ablations |

Data: HelpSteer2 preference subset (4,601 training pairs, 236 held-out pairs, no shared prompts).
Gold judge: Skywork-Reward-V2-Llama-3.1-8B, never used in training.

## Evidence, in order of priority

1. **Best-of-n** (`best_of_n`): 64 validation prompts, 64 samples each, n = 2 ... 64. Primary,
   because it covers every reward model kind cheaply.
2. **GRPO** (`grpo` + `evaluate`): `bt` vs. `jepa`, compared at matched KL to the base model.
3. Held-out accuracy (`reward_models`) and the stress test (`stress_test`), as context.

All numbers below are gold-judge score changes in units of the base model's standard deviation
(σ), with 95% bootstrap CIs over prompts, as the code reports them.

## Definitions (fixed now)

- **Optimizable:** the reward model's own score of its picks rises by at least 0.5σ by the
  largest n (or the final GRPO checkpoint). A reward model that can't be optimized can't be hacked
  either, and resisting hacking only means something for one that can.
- **Hacked:** the gold score of its picks at the largest n (or final checkpoint) is below the peak
  of its own gold curve, with the 95% CI of that point lying entirely below the peak value.
- **Better than X:** the paired gold gap `jepa − X` at the largest n has a 95% CI above 0
  (`comparisons` in `best_of_n_qwen.json`). CIs that include 0 mean **no difference shown**, not
  "equal".

## Hypotheses and decision rules

| | hypothesis | supported if | if not |
|---|---|---|---|
| H0 | the setup can show hacking at all | `bt` is hacked in best-of-n or GRPO | the experiment can't test resistance; say so, then (planned, once) rerun GRPO with a higher `--lr`, and report both |
| H1 | JEPA resists hacking | `jepa` is optimizable and not hacked, while `bt` is hacked | report which condition failed |
| H2 | the JEPA objective is responsible | `jepa` better than `frozen_bt` | the frozen features, not the objective, explain any difference |
| H3 | it beats the simple fix | `jepa` better than `bt_lenpen` | a length penalty does as well; report it |
| H4 | which loss term matters | `jepa` vs. `jepa_pred` and vs. `jepa_rank` | report the pattern, no claim |

**Accuracy caveat, decided now:** if `jepa`'s held-out accuracy is more than 5 points below `bt`'s
(paired CI below −5 points), every resistance claim is reported alongside "and it's a weaker reward
model", because a weaker signal is less exploitable for that reason alone.

**Seeds:** a GRPO claim counts only if it holds in every seed run. With one seed, GRPO results
are reported as preliminary.

## What won't change after seeing results

The judge, the prompts (`load_prompts` with seed 0), n values, decision rules, σ units, and which
comparisons are primary (H1 to H3). If any of these change, it goes in the deviations log.

## Outside prediction: Claude (AI coding assistant), 2026-10-14, before any data

Written by the AI assistant that helped build this codebase, at the author's request, as a
separate prediction to compare against. These are its guesses, not the author's.

- **H0, `bt` gets hacked:** likely in best-of-n (~65%), less likely in GRPO at the default
  settings (~40%). The proxy is a 0.6B model trained on 4.6k pairs, which is weak enough to be
  over-optimized by n = 64; 200 LoRA steps at lr 1e-5 may not move the policy far enough, so the
  pilot will probably call for a higher learning rate.
- **H1, `jepa` resists:** unlikely (~25%). More likely it fails the "optimizable" test, or it gets
  hacked a different way: generic, on-topic answers that sit near the predicted embedding
  without being correct (the wrong-facts and copy-the-question edits should show this first).
- **H2, beats `frozen_bt`:** unlikely (~25%). With the same frozen features and capacity, the two
  probably differ by less than the CI; if anything, `frozen_bt` will be the more accurate.
- **H3, beats `bt_lenpen`:** unlikely (~20%). Length is the biggest single hack in this setting,
  and removing it gets most of the benefit.
- **Held-out accuracy:** `bt` ~68–72%, `frozen_bt` ~60–66%, `jepa` ~58–64%, `jepa_pred` lowest.
- **Most likely overall outcome:** "JEPA is a weaker signal" (~50%), then "JEPA gets hacked
  differently" (~30%), then "JEPA resists" (~20%).

## My predictions (before any data)

- H0, `bt` gets hacked: ___ (likely / unlikely), because ___
- H1, `jepa` resists: ___, because ___
- H2, beats `frozen_bt`: ___
- H3, beats `bt_lenpen`: ___
- Most likely overall outcome: ___

Date: ___   Commit: (the commit that adds this file)

## Deviations log

| date | what changed | why |
|---|---|---|
| 2026-10-14 | GRPO pilot (`bt`, lr 1e-5) stopped at step ~37 of 60; next pilot at lr 1e-4 | KL to the base model stayed ~0.001 per token through step 35 and reward and answer length showed no trend, so at lr 1e-5 the policy barely moves and H0 can't be tested. Raising the learning rate once is the response planned under H0. |
| 2026-10-14 | Answer length limit raised from 768 to 1,536 tokens | Before any real run, a GPU check found 31% of the base model's answers cut off at 768 tokens, which would cap the length hacking being measured. |
| 2026-10-14 | Best-of-n run before further GRPO | Budget: best-of-n is the primary evidence and needs no tuning; GRPO resumes once a pilot shows the policy moving. |
