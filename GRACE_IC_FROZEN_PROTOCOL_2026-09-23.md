# GRACE Identification-Complexity Ceiling Test — Frozen Development Protocol

Date: 2026-09-23  
Status: **FROZEN BEFORE DEVELOPMENT RESPONSE GEOMETRY / HELD-OUT NOT AUTHORIZED**

## 1. Purpose

This is the single final empirical ceiling carrier for the writable-model identification-complexity program.

The sole decision question is whether a real published persistent-write system exhibits an identification frontier that is structurally different from the two frozen LoRA carriers, both of which collapsed to first-order single-probe closure.

No additional carrier is authorized after this test merely because the outcome is negative or simple.

## 2. Published carrier

Use the official implementation of:

- Thomas Hartvigsen et al., **Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors**, NeurIPS 2023;
- official repository: `Thartvigsen/GRACE`;
- frozen repository commit: `f674183f17a995d109e10ee6140d4c3e6d016115`;
- official SCOTUS model: `tomh/scotus-bert`;
- tokenizer: `bert-base-cased`;
- official SCOTUS data: `tomh/grace-scotus`;
- official edited layer: `bert.encoder.layer[10].output.dense`;
- official GRACE editor values: `edit_lr=1.0`, `n_iter=100`, `eps=1.0`, Euclidean routing, cold value initialization, `replace_prompt`, coverage epsilon expansion.

The exact resolved Hugging Face model and dataset revisions are recorded before response geometry begins and must be reused in any later held-out execution.

## 3. Scientific specialization

This experiment does **not** claim that a smooth Taylor chart exists for discrete key insertion itself.

GRACE first constructs a real persistent codebook using its official sequential edit operation. After this current state is constructed, the measured writable substate is the fixed-topology codebook **value tensor**. This is a natural continuous persistent state because GRACE itself trains these values during editing.

Local admissible probes are normalized negative-loss-gradient rays in the same GRACE value parameters, with keys, radii, routing rule, base model, and codebook topology fixed.

The valid claim is therefore limited to:

> identification complexity of the fixed-topology GRACE value state under natural local value-training write directions.

No claim is made about the differential geometry of adding or deleting discrete keys.

## 4. Why this carrier is theorem-discriminating

Unlike LoRA, GRACE stores edits in a routed multi-slot memory. A local value-training write can affect the slot selected by routing while leaving other slots unchanged. Therefore a single write probe need not identify the full current response state.

The predeclared ceiling signal is any of:

1. `q_1(V_star) > 1` — nontrivial first-order probe breadth;
2. `r_2 > r_1` or `r_3 > r_2` — delayed higher-order visibility;
3. `q_2 < q_1` or `q_3 < q_2` — breadth-for-depth substitution.

If none occurs and the target is recovered by one first-order probe, record `DEV_COLLAPSE_11` and stop this carrier without rescue.

## 5. Deterministic data roles

Use the official SCOTUS **test** split only for this experiment. Sort all examples by SHA256 of:

`GRACE-IC-20260923|label|raw_text`

without text normalization.

Take non-overlapping pools in sorted order:

- anchor candidates: 128;
- probe candidates: 256;
- development consequence candidates: 256;
- held-out candidates: 256.

The held-out pool may be hashed and reserved but may not be evaluated, routed, summarized, plotted, or used for numerical choices during development.

## 6. Construct the current persistent state

Start from the public `tomh/scotus-bert` model and attach official GRACE at the official layer.

Scan the anchor-candidate pool in frozen hash order. Only examples misclassified by the current model are edited, matching the published GRACE use case. Apply the **official GRACE edit** with `100` iterations per edit.

Stop when the codebook contains four keys or after 12 successful edits.

Development stop condition:

- if fewer than four keys exist after 12 successful edits, record `ANCHOR_TOPOLOGY_DEGENERATE`;
- do not change epsilon, editor iterations, model, layer, or candidate ordering.

Once four keys exist, freeze the first four key/value/radius slots. No later topology search is allowed.

## 7. Four-dimensional local state chart

For each frozen slot, use the anchor example that created that slot.

At the final current state, require the creator example still routes to the same slot. Compute the negative classification-loss gradient with respect to the four-slot GRACE value tensor, retain only that slot's row, and normalize it.

These four disjoint-support unit directions are transformed by the fixed normalized `4 x 4` Hadamard matrix. The resulting four directions define the state chart.

The chart must be orthonormal to maximum absolute Gram error `<= 1e-5`; otherwise stop as a technical/state-chart degeneracy.

This fixed mixing prevents the chart axes themselves from simply being named by probe slot.

## 8. Four natural local write probes

Scan the disjoint probe-candidate pool in frozen hash order.

For each frozen memory slot, select the first **separate** example that:

- routes inside that slot's frozen GRACE radius;
- yields a nonzero value gradient.

Its normalized negative loss gradient on the chosen slot is that slot's local write probe.

Exactly one probe is selected for each of the four slots. Selection uses routing/support only, never response-rank or frontier outcomes.

If all four slots cannot be covered, stop as `PROBE_COVERAGE_DEGENERATE`. Do not enlarge the pool or change GRACE epsilon.

## 9. Development consequence map

From the development candidate pool, select the first 16 examples in frozen order that route inside one of the four frozen slots. Selection uses routing only.

If fewer than 16 supported examples exist, stop as `DEV_SUPPORT_DEGENERATE`.

For example `i` with `C` logits and gold class `y_i`, define the smooth signed margin

`m_i(s) = z_{i,y_i} - mean_{c != y_i} z_{i,c}`.

The consequence map is the 16-dimensional vector of these margins.

Accuracy is descriptive only.

## 10. Probe chart and development-only amplitude calibration

For normalized probe direction `d_p`, define

`phi_p(u;s) = s + u * eta * d_p`,

on the fixed GRACE value tensor only.

Use normalized coordinates

`u in {-1,-1/2,-1/4,0,1/4,1/2,1}`.

Development-only candidate amplitudes:

`eta in {1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1}`.

For every eta and every probe fit the zero-intercept cubic response. Eta passes only if all four probes satisfy:

- `signal_RMS >= 1e-4`;
- `curve_NRMSE <= 0.05`;
- `max_abs_response <= 0.5`.

Select the largest passing eta. If none passes, stop as `NO_LEGAL_LOW_ORDER_WINDOW`.

Freeze the state finite-difference amplitude as `epsilon = eta`.

## 11. Response jets and state differential

Fit orders 1, 2, and 3 from the same seven-point grid.

For each state-chart direction `e_j`, evaluate exactly

`s_j+ = s0 + epsilon e_j`,

`s_j- = s0 - epsilon e_j`.

For each probe subset `P` and response order `k`, form `C_(k,P)(s)` by stacking coefficient blocks, then estimate

`Xi[:,j] = (C(s_j+) - C(s_j-))/2`

in normalized chart coordinates.

All probes are evaluated from exact resets of the intended value state. Keys, radii, and base-model parameters never change in this geometry phase.

## 12. Rank and target equality

The declared state-chart dimension is four.

Primary numerical rank threshold:

`tau = max(1e-8, 1e-4 * sigma_1)`.

Mandatory diagnostic relative thresholds:

- `1e-3`;
- `1e-5`.

Define the development target

`V_star = V_s0(3,P_all)`

with all four probes.

For every one of the `2^4=16` probe subsets and `k in {1,2,3}`, declare equality to `V_star` only if:

1. primary ranks are equal; and
2. operator norm of the difference of row-space projectors is `<= 0.05`.

Define

`q_k(V_star) = min |P|`

among target-equal subsets, or infinity if none exists.

## 13. Frozen development dispositions

Development output is one of:

- `NONTRIVIAL_DEV_FRONTIER` if any of:
  - `q_1 > 1`;
  - delayed higher-order visibility;
  - breadth-for-depth substitution;
- `DEV_COLLAPSE_11` if one first-order probe already recovers the order-3 all-probe target and no higher-order gain occurs;
- one of the explicit technical/degeneracy stops above.

A nontrivial development result authorizes only a later freeze of the exact model/data revisions, anchor state, probe identities, eta, target rules, and one held-out execution. It is **not** confirmatory evidence by itself.

A collapse result ends this final carrier. No alternative model, layer, epsilon, memory size, consequence family, or dataset is opened to search for a positive result.

## 14. Held-out prohibition

This development workflow must report `heldout_evaluated=false`.

The reserved held-out pool is not routed or evaluated. There is no held-out workflow in the repository at protocol freeze time.

## 15. Claim boundary

This experiment can support a contrast between:

- globally coupled LoRA carriers that empirically collapsed to first-order single-probe closure; and
- a routed persistent-memory carrier whose local write-response state may require multiple probes.

It cannot support claims that GRACE is generally superior, that all memory editors have nontrivial breadth, or that discrete key insertion has a smooth Taylor geometry.
