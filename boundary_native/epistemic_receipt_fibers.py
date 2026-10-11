"""Parallel C: identity-anchored possible-world receipt fibers for Agent side effects.

This is NOT a neural encoding, nor a claim to invent epistemic model checking.
It asks whether Agent runtime testing needs a *set of still-possible histories*
rather than conflating tool logs, user permission, commit acknowledgments and
persisted tool results.

All states are deterministic abstractions on one logical call ID. Receipt
source identity is role-checked; real authorization/remote idempotency API
is NEVER invoked here. An SDK trace can establish a logged call, but cannot
by itself prove external commit or legal retry. UNKNOWN remains first-class.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from itertools import product
from typing import Sequence


# Distinct public evidence rights; source_kind is a reviewer-attested
# adapter capability, not a cryptographic token. For real external systems,
# separate authenticated adapter and caller rights would be necessary.
RIGHTS={
    "session_journal":frozenset({"call_recorded","result_recorded"}),
    "local_effect_counter":frozenset({"effect_committed"}),
    "idempotency_receipt":frozenset({"effect_committed"}),
    "approval_authority":frozenset({"approval_valid"}),
}

FACTS=frozenset({"approval_valid","call_recorded","result_recorded","effect_committed"})


@dataclass(frozen=True)
class Receipt:
    receipt_id:str
    logical_call_id:str
    source_kind:str
    fact:str
    value:bool
    claimed_authorized:bool=True


@dataclass(frozen=True)
class World:
    approval_valid:bool
    call_recorded:bool
    result_recorded:bool
    effect_committed:bool


@dataclass(frozen=True)
class PossibleWorldFiber:
    status:str
    logical_call_id:str
    world_count:int
    world_keys:tuple[tuple[bool,...],...]
    accepted_receipt_count:int
    duplicate_receipt_count:int
    unknown_reason:str
    can_replay_without_external_duplication:str
    mandatory_next_evidence:tuple[str,...]
    semantic_scope:str

    def json(self):
        return asdict(self)


def all_worlds():
    for app,call,result,effect in product((False,True),repeat=4):
        if result and not call:
            continue  # a durable result record has an associated durable call
        yield World(app,call,result,effect)


def _world_key(w):
    return (w.approval_valid,w.call_recorded,w.result_recorded,w.effect_committed)


def _incomplete(identity,reason,seen=0,dups=0):
    return PossibleWorldFiber(
        "UNKNOWN",identity,0,(),seen,dups,reason,"UNKNOWN",(),
        "test-only identity-specific histories; no remote exactly-once proof",
    )


def infer_worlds(
    logical_call_id:str,
    receipts:Sequence[Receipt],
    *,
    authority_mode:str="read_only_reconciliation",
) -> PossibleWorldFiber:
    if not isinstance(logical_call_id,str) or not logical_call_id:
        raise ValueError("logical call ID required")
    if authority_mode!="read_only_reconciliation":
        return _incomplete(logical_call_id,"unsupported_mutating_authority")
    unique={}
    duplicates=0
    for item in receipts:
        if not isinstance(item,Receipt):
            raise ValueError("malformed provenance receipt")
        if item.logical_call_id!=logical_call_id:
            # A receipt about another invocation is not evidence here,
            # even if function names and result contents are equal.
            continue
        if (not item.receipt_id or item.source_kind not in RIGHTS
            or item.fact not in FACTS
            or type(item.value) is not bool):
            return _incomplete(logical_call_id,"invalid_receipt_shape",len(unique),duplicates)
        if not item.claimed_authorized:
            return _incomplete(logical_call_id,"receipt_not_authorized_to_attest",len(unique),duplicates)
        if item.fact not in RIGHTS[item.source_kind]:
            return _incomplete(logical_call_id,"source_cannot_attest_requested_fact",len(unique),duplicates)
        if item.receipt_id in unique:
            if unique[item.receipt_id]!=item:
                return _incomplete(logical_call_id,"conflicting_duplicate_receipt",len(unique),duplicates)
            duplicates+=1
            continue
        unique[item.receipt_id]=item
    worlds=[
        w for w in all_worlds()
        if all(getattr(w,item.fact)==item.value for item in unique.values())
    ]
    if not worlds:
        return _incomplete(logical_call_id,"inconsistent_authoritative_receipts",len(unique),duplicates)
    safe={not w.effect_committed and w.approval_valid for w in worlds}
    if safe=={True}:
        replay="CONDITIONALLY_SAFE_ONLY_IF_EXTERNAL_REPLAY_IS_PERMITTED"
    elif safe=={False}:
        replay="UNSAFE_OR_UNAUTHORIZED"
    else:
        replay="UNKNOWN"
    uncertain=[
        fact for fact in ("approval_valid","effect_committed","result_recorded")
        if len({getattr(w,fact) for w in worlds})>1
    ]
    return PossibleWorldFiber(
        "SET_VALUED_EVIDENCE",logical_call_id,len(worlds),
        tuple(sorted(_world_key(w) for w in worlds)),
        len(unique),duplicates,"",replay,tuple(uncertain),
        "read-only source-scoped possible-world filter, NOT authorization or exactly-once execution",
    )


def source_scoped_observation_value(receipts:Sequence[Receipt],fact:str,call_id:str):
    """Independent exact-identity fact projection, no fallthrough to logs."""
    value=set()
    for item in receipts:
        if (item.logical_call_id==call_id and item.fact==fact
            and item.source_kind in RIGHTS and
            fact in RIGHTS[item.source_kind] and item.claimed_authorized):
            value.add(item.value)
    return next(iter(value)) if len(value)==1 else None


def naive_log_only_retry(receipts:Sequence[Receipt],call_id:str):
    """Deliberately UNSAFE control: missing durable result interpreted as retry."""
    has_result=any(
        item.logical_call_id==call_id and item.fact=="result_recorded"
        and item.value is True for item in receipts
    )
    return "SUGGEST_RETRY" if not has_result else "NO_RETRY"


def next_read_only_receipt_query(fiber:PossibleWorldFiber,
                                 permitted_sources:Sequence[str]):
    """Classical set partition heuristic, not a novel info-gain theorem.

    Return an admissible read-only query that can reduce possible worlds.
    Never returns a write or retry instruction.
    """
    if fiber.status!="SET_VALUED_EVIDENCE":
        return {"status":"ABSTAIN","reason":fiber.unknown_reason}
    choice=[]
    for fact in fiber.mandatory_next_evidence:
        kind={
            "approval_valid":"approval_authority",
            "effect_committed":"idempotency_receipt",
            "result_recorded":"session_journal",
        }[fact]
        if kind in permitted_sources:
            choice.append((0 if fact=="effect_committed" else 1,fact,kind))
    if not choice:
        return {"status":"ABSTAIN","reason":"no_authorized_observation"}
    _,fact,kind=min(choice)
    return {"status":"REQUEST_READ_ONLY_RECEIPT","source_kind":kind,
            "fact":fact,"logical_call_id":fiber.logical_call_id,
            "claim":"information request only, never auto-retry"}
