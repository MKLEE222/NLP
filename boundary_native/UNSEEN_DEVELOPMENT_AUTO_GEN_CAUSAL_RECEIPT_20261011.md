# Development result: AutoGen stream-cancellation boundary (2026-10-11)

The 16-issue register is frozen before issue-body reads: 4 development issues and 12 confirmation issues. This receipt concerns the *preregistered development* issue microsoft/autogen#7956, which was opened after split commitment. The 12 sealed confirmation issue bodies and patches have **not** been opened.

## Actual native source and measured result

Upstream source commit `027ecf0a379bcc1d09956d46d12d44a3ad9cee14`, source-file SHA256 `c0e578fc34be40f46f8e36a91b53fb27cf5ff454f4efb3f4ea5dc1e32d0048c1`.

[Source plan and real native run](https://github.com/MKLEE222/NLP/actions/runs/38116679755):

- Source AST scanner selected the nested `_execute_tool_calls` producer, its `asyncio.gather` await (line 1200), and the normal-path end-of-stream signal (line 1214), paired with the outer awaiting stream consumer.
- In a local, no-network ReplayChatCompletionClient + function-tool fixture, **normal completion** reached ToolCallExecutionEvent and Response and returned.
- When the same tool was cancelled after confirmed entry, **consumer did not terminate within the predeclared 1.5-second bound**. The runner recorded FAIL because normal control was valid. Hard timeout prevented a background/unbounded hang.
- The first CI attempt failed due to invalid Python-identifier agent name. It is preserved as an *invalid fixture* and excluded from scientific outcomes. Changing the fixture name to a valid identifier produced the actual native result above.

## Causal intervention (test-only)

[Guarded source intervention and native rerun](https://github.com/MKLEE222/NLP/actions/runs/38116775373).

In the ephemeral source checkout only, the lone producer's `None` stream-termination signal was placed in the `finally` block enclosing the abortable gather. The script first verified the full original source SHA and AST source lines. It then produced patched source SHA256 `918208ace598ece8bcb9b0f1414377a0faa529c00389edb9c082e2ec5710f600`.

- On modified source, the scanner reported zero missing-sentinel candidates, as expected.
- The **same native control** completed normally; the cancellation case ended within the same 1.5-second bound with `CancelledError`, instead of timing out.
- This supports a concrete missing-terminator causal mechanism. It does **not** demonstrate complete cancellation semantics, sibling cleanup, absence of leaked background tasks, or correctness of this test-only patch as a production fix.

## Research rights, nearest neighbors and outstanding limits

- The issue was already publicly reported. This is the first new development bug family added since the three original hand-read patterns, **not blind discovery**. Its source motif and adapter were coded after reading the development issue body.
- The generator's AST call-graph overapproximates dynamic reachability; here a real native run confirmed the selected execution.
- Official AutoGen testing utilities already support deterministic model replay. Source-guided fault generation competes with mature static flow analysis, concurrency/fault scheduling and property testing. No superiority data are available yet.
- Initial unit/registry regression suite: 27 passed. After cold-start negative-control hardening, **29 passed**, including a distinct producer-vs-consumer queue mismatch and keyword-argument queue binding. Latest [native source-guided Autogen run](https://github.com/MKLEE222/NLP/actions/runs/38117010172) still detects the same actual bug. The original cross-SDK [seven-job source-guided v2 regression](https://github.com/MKLEE222/NLP/actions/runs/38117012932) also remained green. The 12 precommitted confirmation cases remain untouched and must not be used to retune scanner patterns.
- Scientifically meaningful next steps are: dynamic source-to-adapter generalization without writing issue-specific fixtures, blind confirmed issue-level evaluation, and matched-budget differential/metamorphic/property testing baselines. Avoid asserting novelty based on one known development bug plus a test-only causal patch.

Relevant files: `unseen_issue_registry_v1.json`, `source_guided_synth.py`, `execute_autogen_cancel_plan.py`, `autogen_sentinel_causal_control.py`, and the two Actions workflows.

## 2026-10-11 cold-start soundness correction

An early AST scanner could infer a producer/consumer relation from *unrelated* stream queues: seeing `put_nowait(None)` and `await queue.get()` somewhere in one method is not sufficient. The strengthened compiler requires an actual binding from the task's producer argument to the reader's awaited queue, including positional and keyword argument passing. On the pinned real AutoGen source it records `stream_queue -> stream`; a mismatch becomes **NO CANDIDATE**, not FAIL. This is a useful validity improvement, but not a formal interprocedural proof.

Earlier intermediate GitHub Actions failures due to development-test fixture mismatch have been preserved, not counted as model failures; current 29-test and native suites pass. The causal intervention was also strengthened to refuse patching unless this exact queue binding is present.
