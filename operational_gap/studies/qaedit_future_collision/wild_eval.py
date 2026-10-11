"""Public, local implementation of the ACL-2025 WILD-style free-generation metric.

Source ancestry (not our novelty):
WanliYoung/Revisit-Editing-Evaluation, easyeditor/evaluate/evaluate_utils.py,
functions normalize_answer/exact_match_score/test_prediction_acc_real.
Important adaptation: Qwen2.5-Instruct receives a chat-formatted prompt, unlike
the original repository's default question-only context. We therefore call
this WILD-aligned, NOT an exact authors' reproduction.
No LLM-as-a-Judge/API calls are made.
"""
import re
import string


def normalize_answer(s):
    text = str(s).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match_score(prediction, target):
    return int(normalize_answer(prediction) == normalize_answer(target))


def transcript_key(generated):
    """Unchanged observation contract: only case and whitespace normalization."""
    return " ".join(generated.strip().casefold().split())


def locality_preserved(before, after):
    """Binary free-generation stability: this does not use locality gold labels."""
    return int(transcript_key(before) == transcript_key(after))


def free_generate_wild_aligned(model, tokenizer, question, max_new_tokens=50):
    """Natural stopping is defined externally, not adjusted to model outcomes."""
    import torch
    model.eval()
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True, tokenize=True)
    tensor = torch.tensor([ids], dtype=torch.long)
    kwargs = {
        "input_ids": tensor, "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "stop_strings": [".", "\n", tokenizer.eos_token],
        "tokenizer": tokenizer,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True
    }
    with torch.inference_mode():
        generated = model.generate(**kwargs)
    content = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)
    for suffix in (".", "\n"):
        if content.endswith(suffix):
            content = content[:-len(suffix)]
    return content
