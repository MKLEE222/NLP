# Native reproducibility witness: an accepted exception is not success

**Real historical source issue:** https://github.com/openai/openai-agents-python/issues/3280

**Real fixing PR:** https://github.com/openai/openai-agents-python/pull/3281

This experiment checks just one branch of the published bug: an output guardrail raises an ordinary exception during a streamed run. Before the fix, the exception was logged but a candidate answer was finalized. After the fix, the exception should propagate, preventing a success terminal state. It runs the SDK's own public `FakeModel` so there is no LLM API, provider key, financial write or simulated result disguised as production.

Expected pre-fix source SHA: `683b6e79e5b4a2e3fc3cac80b556101ce778beae`.
Expected merged fix SHA: `4a3d33e44fcd5051ff1b08f136ca61aef7824e63`.

This is a *native integration reproduction* on an actual SDK codebase, not a new algorithm. The normal guardrail case is an internal negative control. Per source issue, the second failure path (cancelled sibling guardrail not awaited) is deliberately **out of scope**, so do not report that it was tested. Do not extrapolate this one historical issue to other frameworks.

Run `python boundary_native/issue3280_stream_guardrail.py --upstream-root upstream --upstream-commit <SHA> --phase before|after --output results.json` after installing the pinned SDK source in a Python 3.11 environment and placing its public test package at `upstream/tests/`.

**Novelty is OPEN, not claimed.** This validates that exception/terminal state is a real test target. The existence of a historical bug and a passing reproducer alone cannot justify a paper. See the mechanism design and strongest prior work before developing a custom component.
