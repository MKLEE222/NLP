# Operational Consequence Gap: model-free research checks

This directory only contains public mathematical toy fixtures and a preregistered research gate. It contains neither the 2026-09-23 manuscript nor private neural experiment material.

**Observed motivation (not proof of operational failure):** full local first-order rank in one RoBERTa/MRPC LoRA chart coincided with a failed, separately evaluated finite-amplitude prediction diagnostic. These are distinct tests. We do **not** claim a verified ambiguity pair in the original neural carrier.

For legal finite-probe transcript T(s), future task consequence H(s), measurement tolerance epsilon and domain D, consider

Gamma = sup ||H(s)-H(s')|| over s,s' in D with ||T(s)-T(s')|| <= 2 epsilon.

Worst-case consequence reconstruction error given observations with noise <= epsilon lies between Gamma/2 and Gamma. This is a standard two-point argument, not a novel mathematical technique.

## Acceptance gate (not yet run on real models)
1. Freeze legal actions and measurement budgets, separate oracle parameter differentiation from deployment-available information.
2. Construct real sequential edit trajectories with hidden editable state, independently held-out future edits and consequences.
3. Compare direct finite-amplitude interpolation, no-probe prediction, random legal probes, gradient/rank selection, singular-value and logdet experimental design under matched scalar observations, forward calls, gradients and resets.
4. Measure false-safe rate, consequence prediction loss, regret, costs and uncertainty by independent edit trajectory.
5. If no naturally occurring, cost-matched difference survives untouched confirmation, drop the operational innovation claim.

Run the public fixtures with `python -m unittest discover -s operational_gap/tests -v`. Passing fixtures does not confirm the real-world hypothesis.
