# Pole-Dynamics Experiments

Runnable, laptop-scale implementations of **every** `\begin{vizexp}...\end{vizexp}`
block from *"Neural Network Training as a Dynamical System"* — 39 experiments
in total, one Python file each, all using plain **NumPy / SciPy /
scikit-learn / Matplotlib** (no GPU, no internet download, no deep-learning
framework required).

## Why "laptop-scale"?

The paper's `vizexp` blocks are specified at production scale (ResNet-50 on
ImageNet, GPT-2, ViT-B/L, StyleGAN2, a real DCGAN, pretrained
BERT/distilGPT-2 attention weights, ...). Reproducing those exactly needs a
GPU cluster and multi-gigabyte downloads. This repo follows the same
convention the paper's own **Appendix A ("Numerical Verification", Scripts
1–38)** already uses for every one of its *proofs*: small, explicit, plain
NumPy/SciPy models and synthetic or `sklearn`-bundled data. Every experiment
here:

1. States which theorem/proposition/corollary it verifies and which
   `\label{viz:...}` it corresponds to, in its module docstring.
2. Uses a structurally faithful small proxy in place of any large pretrained
   model or production dataset (a `TinyMLP`, a small tanh-RNN, a synthetic
   attention matrix, `sklearn.datasets.load_diabetes`/`load_digits`, ...),
   since every claim in the paper is stated for architectures of *arbitrary*
   size, not asymptotically.
3. Prints the same kind of "measured vs. predicted" comparison the paper's
   own Scripts print, and where the paper's appendix already ran the
   identical small-scale check, quotes the paper's own numbers alongside
   ours for direct comparison.

Nothing here is mocked or hard-coded to "look right" — every number printed
is computed from the model at runtime (closed-form where the paper gives a
closed form, or by finite-difference Jacobians/Hessians/Hessian-vector
products and Lanczos otherwise), so any of it can be re-derived by hand
against the paper's equations.

## Project layout

```
pole_dynamics/        shared library
  core.py                TinyMLP, finite-difference Jacobian/Hessian/HVP,
                          Lanczos top-k eigenvalues, pole formula z=1-eta*lambda
  plotting.py             shared matplotlib helpers (complex-plane pole plots, savefig)
experiments/           one file per vizexp, exp01..exp39
outputs/                figures land here when you run an experiment
run_all.py              run everything (or a subset) in one command
tests/                  smoke tests for the shared library
```

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# run everything (prints each experiment's verification, saves all figures to outputs/)
python run_all.py

# run just a few, by number
python run_all.py 05 13 28

# or run one experiment directly
python experiments/exp09_ce_pole_count.py
```

Every experiment finishes in well under a minute on a laptop; the whole
suite (39 experiments) runs in a few minutes.

## Map: `\label{viz:...}` -> file -> what it verifies

| # | Paper label | File | Theorem/Prop. verified |
|---|---|---|---|
| 01 | `viz:one-step` | `exp01_one_step_identity.py` | Thm 2.2 exact one-step identity; realised curvature kappa_k |
| 02 | `viz:lyapunov` | `exp02_lyapunov_functions.py` | Thm 2.10 (loss is Lyapunov, no convexity) & Thm 2.12 (geometric rate) |
| 03 | `viz:manifold` | `exp03_normal_attraction_manifold.py` | Thm 3.14 normal attraction to the interpolating manifold |
| 04 | `viz:svd-poles` | `exp04_svd_pole_formula.py` | Thm 3.9 closed-form pole formula z_i=1-eta*sigma_i^2/N |
| 05 | `viz:eos-slow` | `exp05_eos_self_stabilization.py` | Prop. 8.2 exact 2-coordinate EoS self-stabilisation (matches Script 15) |
| 06 | `viz:alternating` | `exp06_alternating_vs_simultaneous_gan.py` | Prop. 5.3 det(A_alt)=1 exactly vs. simultaneous instability |
| 07 | `viz:root-locus` | `exp07_root_locus.py` | Cor. 4.2 root locus, **real** diabetes dataset (no scale-down needed) |
| 08 | `viz:dirac-gan-viz` | `exp08_dirac_gan_spiral.py` | Prop. 12.5 consensus optimisation moves eigenvalues off imaginary axis |
| 09 | `viz:ce-pole-count` | `exp09_ce_pole_count.py` | Cor. 6.9 CE loses N poles vs. MSE to the logit-shift null direction |
| 10 | `viz:multitask` | `exp10_multitask_null_space.py` | Thm 6.11 ker(H)=intersection of ker(G_k) (matches Script 28) |
| 11 | `viz:loss-poles` | `exp11_loss_functions_poles.py` | Lemma 6.5 / Table 1: Lambda for MSE/Huber/log-cosh/BCE/softmax-CE |
| 12 | `viz:vae-collapse` | `exp12_vae_posterior_collapse.py` | Prop. 6.13 posterior collapse = stable fixed point, not saturation (matches Script 33) |
| 13 | `viz:momentum-poles` | `exp13_momentum_poles.py` | Prop. 4.7 heavy-ball real-to-complex pole transition at beta* |
| 14 | `viz:momentum-bode` | `exp14_momentum_bode_resonance.py` | Prop. 8.5 closed-form Bode resonance frequency (matches Script 10) |
| 15 | `viz:adam-poles` | `exp15_adam_vs_sgd_poles.py` | Prop. 9.4 Cauchy-Schwarz: kappa_Adam <= kappa_GD always |
| 16 | `viz:gen-gap` | `exp16_generalization_gap.py` | Prop. 10.3 N^-1/2 trajectory-gap scaling |
| 17 | `viz:gain-margin` | `exp17_gain_margin_eos.py` | Thm 8.3 gain margin -> 0 dB at Edge of Stability |
| 18 | `viz:waterbed` | `exp18_sensitivity_waterbed.py` | Prop. 8.6 Bode sensitivity integral = 0 exactly |
| 19 | `viz:sensitivity-bode` | `exp19_sensitivity_bode.py` | Prop. 8.4 \|\|S\|\|_inf = 2/(2-eta*lambda), resonance as eta*lambda->2 |
| 20 | `viz:depth-poles` | `exp20_depth_poles_resnet_vs_plain.py` | Prop. 11.2 residual depth-propagation floor (1-c)^L vs. c^L->0 |
| 21 | `viz:non-normal` | `exp21_non_normal_transient_growth.py` | Remark 11.4: pseudospectral transient growth despite rho(M)<1 |
| 22 | `viz:real-kernel-bode` | `exp22_conv_kernel_bode.py` | Prop. 11.6/Cor. 11.7 circulant DFT diagonalisation, residual +1 Bode shift |
| 23 | `viz:layer-gradients` | `exp23_layer_gradient_norms.py` | Prop. 11.10 no early residual layer is arbitrarily starved |
| 24 | `viz:gain-scheduling` | `exp24_gain_scheduling.py` | Prop. 12.1 dead-beat step eta_k=1/kappa_k prevents divergence (matches Script 6) |
| 25 | `viz:wd-null-poles` | `exp25_weight_decay_null_poles.py` | Prop. 13.1(3) weight decay = rigid pole-histogram shift by -eta*lambda |
| 26 | `viz:consensus-dcgan` | `exp26_consensus_optimization.py` | Prop. 12.5(3) closed form, **exact** match to Script 7's numbers |
| 27 | `viz:sgd-variance` | `exp27_sgd_stationary_variance.py` | Thm 14.2 SGD stationary Lyapunov covariance (matches Script 23[B]) |
| 28 | `viz:dino-viz` | `exp28_dino_self_distillation.py` | Thm 15.4/15.5 DINO fixed-pointwise manifold + block-triangular spectrum (matches Script 24) |
| 29 | `viz:replicator` | `exp29_replicator_equation.py` | Thm 16.4 natural policy gradient == replicator equation, **exactly** (matches Script 29) |
| 30 | `viz:rps` | `exp30_rps_selfplay.py` | Prop. 16.6 bilinear self-play game Jacobian is exactly antisymmetric |
| 31 | `viz:atlas` | `exp31_pole_atlas.py` | Section 17.2: aggregated unit-disc atlas of every named phenomenon |
| 32 | `viz:weight-sharing` | `exp32_cnn_weight_sharing.py` | Prop. 18.2 weight-sharing curvature aggregation, boost <= P^2 (matches Script 27[D]) |
| 33 | `viz:rnn-ptb` | `exp33_rnn_time_axis_poles.py` | Thm 18.4 RNN time-axis vanishing/exploding gradient bound |
| 34 | `viz:lstm-copy` | `exp34_lstm_forget_gate_copy_task.py` | Cor. 18.8 LSTM forget-gate retention vs. vanilla RNN attenuation |
| 35 | `viz:gated-jacobians` | `exp35_gated_jacobians_lstm_gru_highway.py` | Prop. 18.6/Rem. 18.9/Prop. 18.10 LSTM exact channel vs. GRU/Highway dense coupling (matches Scripts 27[A,B], 30[A]) |
| 36 | `viz:attention-pole` | `exp36_attention_stochastic_pole.py` | Prop. 18.11/Lemma 18.12 softmax attention's exact unit eigenvalue + Dobrushin bound (matches Script 35) |
| 37 | `viz:grok` | `exp37_grokking_time_law.py` | Appendix Prop. A.1 grokking time-to-threshold t_eps*lambda=const (matches Script 31[A]) |
| 38 | `viz:double-descent` | `exp38_double_descent.py` | Appendix Cor. A.2 generalisation bound blow-up where mu dips (interpolation threshold) |
| 39 | `viz:forgetting` | `exp39_catastrophic_forgetting_ewc.py` | Appendix Prop. A.3 EWC monotonically suppresses drift energy (matches Scripts 31[B], 38) |

Every file is self-contained: it can be read top-to-bottom as a standalone
script, or imported and its `run()` function called from `run_all.py`.

## What's an honest proxy vs. what's the real thing

A few experiments substitute something structurally equivalent for what the
paper's vizexp names, because the named artifact needs an internet
connection or a GPU:

- **#7 (root locus)** needs *no* substitution — `sklearn.datasets.load_diabetes`
  is exactly the dataset the paper's own vizexp text specifies.
- **#2, #16** use `sklearn.datasets.load_digits` (bundled, 8x8 images) in
  place of full MNIST — same closed-form beta_L/mu machinery.
- **#22, #36** use synthetic kernels / synthetic attention matrices in place
  of a pretrained ResNet-18/distilGPT-2's real weights — the propositions
  being tested (circulant diagonalisation; A·1=1) hold for *any* kernel or
  *any* softmax attention matrix, so the mechanism is identical; only the
  specific numeric kernel differs.
- **#17, #20, #23, #24, #33** replace ResNet-50/ImageNet-scale or
  Penn-Treebank-scale training with a `TinyMLP`/small tanh-RNN of the same
  architecture class trained on synthetic data.
- **#37 (grokking)** implements and verifies the paper's own proposed
  *mechanism* (null-space contraction under weight decay) exactly, but does
  **not** reproduce the full modular-arithmetic Transformer benchmark — the
  paper's own text lists testing this mechanism against the real grokking
  benchmark as future work (its Appendix "Research To-Do List", item E1).

Every substitution is called out explicitly in that experiment's module
docstring.

## Tests

```bash
python -m pytest tests/
```

A small smoke-test suite that checks the shared `pole_dynamics` primitives
(Jacobian, Hessian, HVP, Lanczos, pole formula) against closed-form linear
and quadratic test cases with known answers.

## Citing

If you use this code, please cite the paper it accompanies ("Neural Network
Training as a Dynamical System") alongside this repository.
