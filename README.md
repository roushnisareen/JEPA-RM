# jepa-rm

Can a JEPA-style reward model resist reward hacking during RLHF?

A normal reward model reads an answer and gives it a score. During RLHF the policy learns to push
that score up, and a lot of the time it does that by gaming the reward model rather than by getting
better: longer answers, more confident tone, more formatting. That's reward hacking.

The idea here is to score answers by *meaning* instead. A frozen encoder embeds the prompt and the
answer separately, and a small predictor learns to guess what a good answer's embedding should look
like from the prompt alone. The reward is how close the actual answer lands to that guess. If
padding an answer out or changing its tone doesn't move its embedding much, this should be harder
to game.

## Results so far

**Short version: no.** On this setup the JEPA reward model is a weaker judge, has its own loophole,
and under best-of-n selection it held up *worse* than the standard Bradley-Terry (BT) reward model.
It also did no better than a control with the same frozen features and capacity, so the JEPA
objective itself bought nothing. The RL step (GRPO) is still to come. Every run, its cost and the
decisions it led to are in [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md); the hypotheses were written down
first in [PREREGISTRATION.md](PREREGISTRATION.md).

### 1. How good are the reward models?

Agreement with human annotators on the 236 held-out HelpSteer2 pairs (95% bootstrap CIs; the gap to
BT is paired over the same pairs):

| reward model | accuracy | vs. BT | prefers the longer answer |
|---|---|---|---|
| `bt` | 74.2% [69, 80] | — | 61.4% |
| `bt_lenpen` | 72.0% [67, 78] | −2.1 [−6, +1] | 53.4% |
| `frozen_bt` | 67.4% [61, 74] | −6.8 [−13, −0] | 67.4% |
| `jepa` | 66.5% [61, 73] | −7.6 [−14, −1] | 62.3% |
| `jepa_rank` | 65.3% [59, 72] | −8.9 [−15, −2] | 61.0% |
| `jepa_pred` | 56.8% [50, 63] | −17.4 [−25, −9] | 56.8% |

Humans prefer the longer answer 60.2% of the time. JEPA is level with the frozen-feature control
and ~8 points behind BT; without the ranking term (`jepa_pred`) it's near chance.

### 2. Each reward model has its own loophole

200 good answers, each edited in ways that shouldn't help (top) or should hurt (bottom, sanity
checks). Shifts are in each scorer's own standard deviations:

![Stress test](docs/figures/stress_test.png)

Edits that raised a reward model's score while the gold judge lowered it (all CIs exclude 0):

| reward model | edit | its shift | gold judge |
|---|---|---|---|
| `bt` | same text as bullet points | +0.52σ [+0.46, +0.58] | −0.31σ |
| `bt_lenpen` | same text as bullet points | +0.51σ [+0.44, +0.57] | −0.31σ |
| `jepa_rank` | open with "Absolutely! As an expert…" | +0.22σ [+0.18, +0.26] | −0.49σ |
| `jepa` | copy the question into the answer | +0.20σ [+0.15, +0.25] | −0.10σ |

BT's blind spot is formatting, which a length penalty doesn't touch. JEPA's is restating the
question, as its design predicts. No reward model checks facts: changing numbers and flipping claims
cost them at most 0.5σ, against 2.3σ for the judge. The judge lowered its score for every edit that
shouldn't help.

### 3. Best-of-n: the standard reward model held up better

Sample n answers per prompt (100 prompts × 64 samples), let each reward model pick its favourite,
and score the pick with the gold judge. Larger n = more optimization pressure:

![Best-of-n](docs/figures/best_of_n_qwen.png)

BT's picks kept improving up to n = 64 (getting longer, 452 → 505 words). JEPA's peaked at n = 8 and
fell back to the level of a plain sample. Paired over the same prompts at n = 64:

| comparison | gold-judge gap (95% CI) |
|---|---|
| `jepa` − `bt` | −0.12σ [−0.25, 0.00] |
| `jepa` − `bt_lenpen` | −0.10σ [−0.22, +0.02] |
| `jepa` − `frozen_bt` | −0.05σ [−0.15, +0.04] |
| `jepa_pred`, its peak → n = 64 | −0.15σ [−0.25, −0.06], the only reward model formally over-optimized |

The JEPA ablations (`jepa_pred`, `jepa_rank`) side by side:
[docs/figures/best_of_n_ablation_qwen.png](docs/figures/best_of_n_ablation_qwen.png).

**Against the pre-registration:** BT was never over-optimized under best-of-n, so the main hypothesis
(JEPA resists hacking that BT suffers) can't be tested yet; H2 (beats `frozen_bt`) and H3 (beats
`bt_lenpen`) are not supported. Effects are small (~0.1σ): the σ here is the judge's spread across
all answers, mostly differences between prompts, so it understates within-prompt effects.

**Next:** GRPO against `bt` and `jepa`. A trained policy can learn to write everything as bullets,
which best-of-n can't, so BT may yet be hacked under RL.

### 4. Does it depend on how JEPA pools the embedding? (exploratory)

Mean pooling averages token embeddings, so it can't see repetition or word order. The obvious
objection is that a negative JEPA result comes from that. To check it, every embedding-space reward
model was retrained using the last token instead. They then rescored the same 6,400 best-of-n
answers, so the two versions are compared pair by pair on identical answers:

| reward model | accuracy, mean → last | loophole, mean → last | best-of-n @64, last − mean |
|---|---|---|---|
| `jepa` | 66.5% → 60.2% | restate +0.20σ → confident opener +0.10σ | +0.04σ [−0.08, +0.17] |
| `jepa_pred` | 56.8% → 53.4% | restate +0.13σ → none | **+0.22σ [+0.11, +0.35]** |
| `jepa_rank` | 65.3% → 61.9% | confident +0.22σ → confident +0.22σ, filler +0.24σ | +0.03σ [−0.10, +0.16] |
| `frozen_bt` | 67.4% → 61.4% | none → none (barely reacts to any edit) | +0.03σ [−0.08, +0.13] |

Last pooling closes the restate loophole and stops the pure JEPA objective from being
over-optimized, but it costs 3–6 points of accuracy and makes no measurable difference to `jepa`
under best-of-n. The negative result doesn't depend on pooling. Details in the
[lab notebook](LAB_NOTEBOOK.md); this comparison was added after the pre-registration.

Raw results (every answer and score) are in the run's Hugging Face repo under `results/`; training
curves and tables are in the W&B project `jepa-rm`.

## Plan

0. Write down predictions and decision rules in [PREREGISTRATION.md](PREREGISTRATION.md) and
   commit it before any GPU run.
1. Train six reward models on the preference subset of
   [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2), where annotators compared two
   answers to the same prompt directly. After dropping ties and multi-turn prompts that's 4,601
   training and 236 held-out pairs, with no prompt in both. The RL and evaluation prompts come from
   the same two splits, so no prompt a reward model trained on is used to judge it.

   | kind | what it is | why it's here |
   |---|---|---|
   | `bt` | Qwen3-0.6B + LoRA + scalar head, Bradley-Terry loss | the standard reward model |
   | `bt_lenpen` | `bt` with its linear length trend subtracted | the cheap, known fix: JEPA should beat it |
   | `frozen_bt` | frozen encoder + MLP head, Bradley-Terry loss | control: same features and capacity as JEPA, so a difference from it is down to the JEPA objective |
   | `jepa` | frozen encoder + predictor, JEPA + ranking loss | the idea under test |
   | `jepa_pred` / `jepa_rank` | JEPA loss only / ranking loss only | ablations: which term matters |

   Each is checked on the held-out pairs: how often it picks the better answer (with a 95% CI, and
   a paired CI on its gap to `bt`), and how often it just picks the longer one.
2. Stress-test every reward model before any RL: edit good answers in ways that shouldn't help
   (padding, repetition, flattery, formatting, copying the question in) or should hurt (cutting
   it in half, wrong facts, shuffled reasoning), and see which scorers move. The gold judge gets
   the same test, since the rest of the evaluation relies on it.
3. Best-of-n, still with no training: sample 64 answers per prompt from the base model, let each
   reward model pick its favourite of n (up to 64), and score the pick with the judge. Larger n is
   more optimization pressure, so a reward model that's being over-optimized shows up as the
   judge's score of its picks flattening or falling as n grows. This is the other method Gao et al.
   (2022) use alongside RL, so it's an independent check on step 4, and it covers all six reward
   models for a few dollars. JEPA is also compared head to head with each baseline on the same
   prompts (paired gold gaps with 95% CIs).
4. Run GRPO on Qwen3-1.7B against `bt` and against `jepa`. The KL penalty is tiny (1e-3), so if the
   policy can hack its reward model it should start doing it within a couple hundred steps. It's
   nonzero only so KL to the base model gets measured. Pilot the `bt` run first (below): if the
   standard reward model never gets hacked, there's nothing to compare.
5. Score the base model and every saved checkpoint with a much stronger judge that neither run
   trained against ([Skywork-Reward-V2-Llama-3.1-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Llama-3.1-8B)).
   Reward hacking shows up as the policy's own reward going up while the judge's score doesn't.
   This is plotted against both training steps and KL to the base model, the axis used in
   [Gao et al. (2022)](https://arxiv.org/abs/2210.10760).
6. If there's budget left: a DPO baseline (no reward model at all) and the same runs on
   Phi-4-mini to check the result isn't specific to Qwen.

If the JEPA reward model can't tell good answers from bad ones in step 1, that's a result too,
and it's cheaper to find out there than after the GRPO runs.

## Running it

Everything runs on [Modal](https://modal.com) with one A100. Checkpoints go through a private
Hugging Face repo instead of a Modal volume, so different steps can run on different Modal accounts.

```bash
pip install -e ".[dev]"
modal setup
modal secret create huggingface HF_TOKEN=hf_...   # token needs write access
modal secret create wandb WANDB_API_KEY=...        # from wandb.ai/authorize

modal run modal_app.py::reward_models --hf-repo USER/jepa-rlhf
modal run modal_app.py::stress_test --hf-repo USER/jepa-rlhf
modal run modal_app.py::best_of_n --hf-repo USER/jepa-rlhf
modal run modal_app.py::grpo --hf-repo USER/jepa-rlhf --rm bt --max-steps 60   # pilot: is bt hacked?
modal run modal_app.py::grpo --hf-repo USER/jepa-rlhf --rm bt
modal run modal_app.py::grpo --hf-repo USER/jepa-rlhf --rm jepa
modal run modal_app.py::evaluate --hf-repo USER/jepa-rlhf

# optional
modal run modal_app.py::dpo --hf-repo USER/jepa-rlhf
modal run modal_app.py::evaluate --hf-repo USER/jepa-rlhf --runs bt,jepa,dpo
modal run modal_app.py::grpo --hf-repo USER/jepa-rlhf --rm bt --policy phi
```

Use `modal run --detach ...` for anything longer than a few minutes: without it, a laptop that
sleeps or drops its connection cancels the job (a GPU check lost a GRPO run that way).

For a cheap first run to catch bugs: `--max-pairs 500`, `--n-examples 20`, `--max-steps 8`,
`--n-prompts 16`. Each GRPO or DPO run clears its own old checkpoints before it starts, so a smoke
test never leaks into the real run's results, but pointing smoke tests at a separate repo
(`--hf-repo USER/jepa-rlhf-test`) keeps things tidier still.

Running a stage before the one it depends on stops right away with a message saying which job to
run first.

### Seeds

Each seed saves to its own folder (`policy_qwen_bt_s1`, ...; seed 0 keeps the plain name), so
seeds can run on different accounts at the same time:

```bash
modal run modal_app.py::grpo --hf-repo USER/jepa-rlhf --rm bt --seed 1
modal run modal_app.py::grpo --hf-repo USER/jepa-rlhf --rm bt --seed 2
modal run modal_app.py::evaluate --hf-repo USER/jepa-rlhf --seeds 0,1,2
```

The figures then show the mean over seeds with a shaded band from the lowest to the highest seed.
Seeds that don't exist for a run are skipped, so runs can have different numbers of seeds.

Rough costs on one A100 ($2.50/hour on Modal): `reward_models` and `stress_test` around $1 each,
`best_of_n` $6–8 (4,096 answers of up to 1,536 tokens), each `grpo` run $10–14 (generation
dominates), `evaluate` $2–4. Answers can run to 1,536 tokens because Qwen3-1.7B writes long,
formatted answers: at 768, a GPU check found 31% of them cut off mid-sentence, which would cap the
very length hacking the experiment measures. `reward_models` uses all 4,601 training pairs unless `--max-pairs` caps it.

**The pilot.** Run `grpo --rm bt --max-steps 60` first and look at KL and gold vs. proxy in W&B.
If the BT policy isn't being over-optimized, raise `--lr` (e.g. `3e-5`) or `--max-steps` before the
full runs, and note it in the pre-registration's deviations log.

Each run uploads its metrics, training log and the exact config it used to `results/` in the HF
repo. To pull them down:

```bash
huggingface-cli download USER/jepa-rlhf --include "results/*" --local-dir .
```

## Figures

Each figure goes to `results/` in the HF repo and to W&B:

- `stress_test.png` (from `stress_test`): how far each scorer moves for each edit.
- `best_of_n_{policy}.png` (from `best_of_n`): the same three panels as below, against best-of-n KL
  instead of training, for `bt`, `bt_lenpen`, `frozen_bt` and `jepa`. Reward hacking is the middle
  panel falling as n grows. `best_of_n_ablation_{policy}.png` does the same for the JEPA ablations.
- The head-to-head gaps (JEPA minus each baseline, paired over prompts) are in the `comparisons`
  field of `best_of_n_{policy}.json` and `eval_{policy}.json`, and printed at the end of each job.
- `reward_hacking_{policy}.png` (from `evaluate`): three panels over training (each run's own reward
  model score, the gold judge's score with a 95% CI, and answer length). Reward hacking is the left
  panel rising while the middle one doesn't.
- `reward_hacking_kl_{policy}.png`: the same, against KL to the base model. This is the fairer
  comparison, because a reward model that's just a weaker signal moves its policy less and would
  look "unhacked" at any given step.
- `training_{policy}.png`: GRPO training reward and answer length per step.

To restyle or rebuild them for the blog without a GPU:

```bash
huggingface-cli download USER/jepa-rlhf --include "results/*" --local-dir .
python scripts/make_figures.py results --policy qwen   # writes PNG + PDF to figures/
```

## Weights & Biases

Every job is a run in the `jepa-rm` W&B project, with its full config attached:

| run | what's logged |
|---|---|
| `reward-models` (`reward-models-last` for the pooling ablation) | BT loss and learning rate, JEPA loss and holdout accuracy per epoch, the held-out comparison table |
| `stress-test` | the stress test figure and a score change per edit and scorer |
| `best-of-n-{policy}` | the best-of-n figure and a table of every reward model's picks for each n |
| `grpo-{policy}-{rm}` | reward, answer length, share of answers cut off (`completions/clipped_ratio`), KL to the base model and loss per step, plus sample answers |
| `dpo-{policy}` | DPO loss and reward margins |
| `eval-{policy}` | the figures, a score table per checkpoint (including KL and the share of answers cut off), and every generated answer with its scores |

To log under a team or a different project name, add `WANDB_ENTITY=...` or `WANDB_PROJECT=...` to
the `wandb` secret. The GRPO runs in particular are worth watching live: if the reward climbs while
answer length climbs with it, that's the hacking this project is looking for.

## Known limitations

These are design choices that affect what the results can show, so they belong next to them:

- **Mean pooling ignores repetition by construction.** The JEPA reward averages token embeddings,
  so saying everything twice barely changes its score. That isn't robustness. The stress test
  includes two edits averaging can't hide (wrong facts in the same style, and the same sentences
  in reverse order), and the pooling ablation trains a JEPA reward model on the last token
  instead, which sees the whole sequence:
  `modal run modal_app.py::reward_models --jepa-pooling last --hf-repo USER/jepa-rlhf-last`
  (a separate repo, so the two experiments never overwrite each other). The full mean-vs-last
  comparison reuses the mean run's BT models and its best-of-n answers, so the two arms differ only
  in pooling and the best-of-n comparison is paired (same answers, same gold scores):

  ```bash
  modal run --detach modal_app.py::reward_models --hf-repo USER/jepa-rlhf-last --jepa-pooling last --reuse-bt-from USER/jepa-rlhf
  modal run --detach modal_app.py::stress_test --hf-repo USER/jepa-rlhf-last --no-with-gold
  modal run --detach modal_app.py::best_of_n --hf-repo USER/jepa-rlhf-last --rescore-from USER/jepa-rlhf
  modal run modal_app.py::compare_pooling --mean-repo USER/jepa-rlhf --last-repo USER/jepa-rlhf-last
  ```

  This comparison was added after the pre-registration and is reported as exploratory; results
  are in [Results so far](#results-so-far).
- **The JEPA reward model embeds the answer without the question.** Its score is a fixed direction
  per prompt, so it may reward "sounds like a good answer to this kind of question" rather than
  correctness. The wrong-facts edit tests for this.
- **Reward model strength differs.** The BT reward model fine-tunes its backbone, while JEPA only
  trains a small predictor on a frozen one. If JEPA's held-out accuracy is much lower, "harder to
  hack" may just mean "weaker signal". Three things guard against that: `frozen_bt` has JEPA's exact
  features and capacity, so comparing against it isolates the objective; accuracy (with CIs) is
  reported next to every hacking result; and the KL-axis figures compare policies that moved the
  same distance from the base model.
- **The reward models don't see exactly the same training pairs.** BT trains on all 4,601 pairs
  in one pass, in dataset order. The embedding-space models hold out 5% of prompts to pick their
  best epoch, so they train on ~95% of the pairs. Either way the held-out comparison uses the same
  236 pairs, which no model ever saw.
- **Results so far predate a normalization fix.** Before 2026-10-15 the embedding models fitted
  their feature normalization (a mean and spread per dimension, no labels) on all training
  prompts, including the 5% used to pick the best epoch. It now uses the fitting prompts only.
  The 236 held-out pairs were never included, so the reported accuracies and best-of-n results
  are unaffected by leakage; retraining with the fix would move them only slightly.
- **Is it really JEPA?** The predictor targets a frozen encoder's embedding rather than an EMA
  target encoder, and the default loss adds a ranking term. `jepa_pred` (JEPA loss only, no
  negatives) is the purest version; the ablation figure shows how much each term contributes.
- **Seeds and the judge.** The gold-score changes come with paired bootstrap 95% CIs over prompts,
  which cover sampling noise in the evaluation but not run-to-run variance in GRPO itself. Only
  several seeds measure that (see Seeds above), so treat a single-seed difference as preliminary.
  Check the judge by hand too (below).

## Checking the judge by hand

Every result leans on one judge model, so it's worth checking a sample yourself. This writes a
sheet of blind pairs (the base model's answer vs. a trained checkpoint's, in random order):

```bash
python scripts/human_check.py export results/eval_qwen.json --run bt --step 200 --n 50
# fill the "better" column in human_check.csv with A, B or tie, without opening the key file
python scripts/human_check.py score human_check.csv human_check_key.json
```

It reports how often you agreed with the judge, and how often you preferred the trained model. If
you prefer the base model where the judge doesn't, the judge was fooled too.

## Tests

GitHub Actions runs both suites on every push (`.github/workflows/tests.yml`).

```bash
pytest -m "not integration"   # unit tests, a few seconds
pytest                        # also runs the whole pipeline on tiny models on CPU (~1 min)
ruff check .
```

The integration tests (`tests/test_pipeline.py`) run every stage, from training both reward
models through the stress test, GRPO (two seeds, after planting a stale checkpoint that must be
cleared), DPO and evaluation, on tiny random Qwen3 models with a `LocalStore` in place of the HF
repo, plus GRPO and generation with a tiny Phi-3 model (the architecture Phi-4-mini uses). It exercises the real TRL, PEFT and transformers code paths,
so library mismatches show up on a laptop for free rather than an hour into a GPU run. It
downloads about 60 MB of tiny models the first time.

## Layout

```
modal_app.py                 GPU jobs (thin wrappers around the package)
src/jepa_rm/
    config.py                model names and every hyperparameter, as dataclasses
    data.py                  HelpSteer2 pairs and prompts
    artifacts.py             where each checkpoint lives; HubStore (HF repo) and LocalStore (a folder)
    tracking.py              W&B logging (does nothing when no run is active)
    runtime.py               picks GPU or CPU, so the same code runs on Modal and on a laptop
    rewards/
        __init__.py          the shared interface, and the list of reward model kinds
        bradley_terry.py     standard reward model, and its length-penalized version
        embeddings.py        frozen encoder and training loop shared by the embedding-space kinds
        frozen_bt.py         capacity-matched control
        jepa.py              JEPA reward model and its ablations
        suite.py             trains all six kinds, sharing one embedding pass
    training/
        grpo.py              GRPO against either reward model
        dpo.py               DPO baseline
    evaluation/
        reward_models.py     accuracy and length bias on held-out pairs, with CIs
        stats.py             bootstrap CIs and paired head-to-head gaps
        policies.py          generation + gold-judge scoring
        stress_test.py       edits that shouldn't help, and which scorers they fool
        best_of_n.py         over-optimization without training: each reward model picks the best of n
        pooling.py           mean vs. last-token pooling, side by side and paired
        human_check.py       blind pairs for checking the judge by hand
        plots.py             all figures
scripts/make_figures.py      rebuild the figures locally from downloaded results
scripts/human_check.py       export a blind rating sheet, then score it
tests/                       unit tests, plus a tiny-model run of the whole pipeline
PREREGISTRATION.md           hypotheses and decision rules, committed before any run
LAB_NOTEBOOK.md              dated record of every run: cost, result, and decisions made
```

## License

MIT, see [LICENSE](LICENSE).
