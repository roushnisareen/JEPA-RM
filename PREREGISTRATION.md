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

1. **Best-of-n** (`best_of_n`): 64 validation prompts, 64 samples each, n = 2 ... 64 (as run: 100
   prompts; see the deviations log). Primary,
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

## The author's own predictions

Not recorded in time. The template for them (H0–H3 and an overall guess) was left blank when the
first GPU job ran on 2026-10-14, so the only prediction that provably came before the data is the
outside one above. Filling the blanks afterwards would make them look like forecasts, so they stay
empty. (Noted 2026-10-24.)

## Deviations log

| date | what changed | why |
|---|---|---|
| 2026-10-14 | Answer length limit raised from 768 to 1,536 tokens | Before any real run, a GPU check found 31% of the base model's answers cut off at 768 tokens, which would cap the length hacking being measured. |
| 2026-10-14 | GRPO pilot (`bt`, lr 1e-5) stopped at step ~37 of 60; next pilot at lr 1e-4 (now the default) | KL to the base model stayed ~0.001 per token through step 35 and reward and answer length showed no trend, so at lr 1e-5 the policy barely moves and H0 can't be tested. Raising the learning rate once is the response planned under H0. |
| 2026-10-14 | Best-of-n run before further GRPO | Budget: best-of-n is the primary evidence and needs no tuning; GRPO resumes once a pilot shows the policy moving. |
| 2026-10-14/15 | Best-of-n used 100 prompts, not 64, and was extended from 16 to 64 samples per prompt with a second sampling seed | The job's command-line default (100 × 16) overrode the config (64 × 64), which a test now prevents. The 100 × 16 answers were already paid for and give tighter CIs than a fresh 64 × 64 run, so they were kept and 48 more answers per prompt were added with a fresh seed. The prompts are still `load_prompts` with seed 0, i.e. the first 100 of the same ordering. |
| 2026-10-15 | Added a mean vs. last-token pooling comparison (exploratory, not pre-registered) | Mean pooling ignores repetition and word order by construction, the most obvious objection to a negative JEPA result. Its results are reported as exploratory: no hypothesis above depends on them. |
| 2026-10-24 | Policy evaluation samples at temperature 1.0 / top-p 1.0 instead of 0.7 / 0.95 | The KL estimator only measures the right thing when the answers come from the distribution being measured. GRPO and best-of-n already sampled at 1.0; evaluation was the odd one out. No GRPO checkpoint had been evaluated yet, so no result changes. |
| 2026-10-24 | "Cut-off answers get no reward" corrected to "get no gradient" | In TRL 0.17 every answer is scored before truncated ones are masked, so a cut-off answer still enters the group baseline and the logged reward. The policy still gets no direct push to ramble past the limit. Wording only; no setting changed. |
| 2026-10-24 | Two training asymmetries noted, not changed: BT trains on all 4,601 pairs in one fixed pass, the embedding-space models on ~95% with the best of 30 epochs picked on the rest; and at τ = 0.05 the ranking term's gradient is up to 20× the JEPA term's, so `jepa` sits close to `jepa_rank` by construction | Changing either now would break the link between the code and the results above. `frozen_bt` vs. `jepa` stays a matched pair; `bt` vs. any frozen model is not matched on early stopping. Reported as limitations. |
| 2026-11-03 | Sampling made explicit everywhere: temperature 1.0, top-p 1.0, top-k 20 | transformers 4.51 silently replaced GRPO's requested temperature 1.0 / top-p 1.0 with Qwen3's defaults (0.6 / 0.95, plus top-k 20), so the 10-14 pilot actually trained at temperature 0.6, and best-of-n and evaluation were getting top-k 20 without saying so. Top-k 20 is kept (it's how the 6,400 best-of-n answers were sampled, so they stay valid); temperature 1.0 now actually applies in GRPO. The second pilot, started the same day under the old code, was stopped after ~10 minutes. |
| 2026-11-06 | `strip_think` also removes a lone `<think>` / `</think>` tag; evaluation answers rescored | After 200 GRPO steps against BT, all 800 checkpoint answers began with a stray `</think>`, which the old cleanup (complete `<think>…</think>` blocks only) left in. The gold judge scored them with the tag. The first BT evaluation's scores are kept as `eval_qwen_before_rescore.json`; the rescored ones replace them. The BT GRPO run itself was trained with the tag visible to BT; the JEPA run will be trained with the fixed cleanup, so its reward models never see the tag. |
| 2026-11-11 | The JEPA GRPO run continues from its step-100 checkpoint (`--resume-step 100`) | The run stopped between steps 100 and 150, most likely when the Modal workspace it ran on ran out of credit. Only the LoRA adapter is saved at a checkpoint, not the optimizer, so the continuation restarts Adam's moments at step 100. It keeps the same linear learning-rate schedule from where it was (5e-5 → 0 over steps 100–200), and uses only the 1,600 prompts the first 100 steps hadn't reached (TRL's own shuffle order, reproduced and tested). The BT run went through in one piece, so this is an asymmetry between the two; BT vs. JEPA at step 100 is unaffected by it. |
| 2026-11-11 | The continuation also saves a resume point every 10 steps (kept apart from the checkpoints) and, if Modal preempts it, restarts from the latest one | The first continuation was preempted at step ~114 and Modal restarted it from step 100, costing ~30 min of a tight budget. Each further restart again resets Adam's moments; the prompts stay exactly the unseen ones (each restart reproduces TRL's shuffle once more, tested). The steps it restarted at are saved in `grpo_qwen_jepa.json` as `resumed_from`. |
