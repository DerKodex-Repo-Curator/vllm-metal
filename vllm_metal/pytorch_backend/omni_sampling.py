# SPDX-License-Identifier: Apache-2.0
"""MRv2 sampling operations required by Qwen3-TTS on MPS.

The upstream sampler still owns request state, filtering order and writeback.
These bindings cover its non-speculative MPS path in vLLM 0.31.
"""

import torch

from vllm_metal.pytorch_backend.runtime import cpu_mirror


def apply_temperature(logits, mapping, temperature):
    values = temperature[mapping.long(), None]
    logits.div_(torch.where(values == 0, 1, values))


def bincount(mapping, all_ids, prompt_len, prefill_len, prompt_mask, counts, max_len):
    slots = cpu_mirror(mapping).tolist()
    prompts = cpu_mirror(prompt_len)
    prefills = cpu_mirror(prefill_len)
    ids = all_ids[slots, :max_len].cpu().long()
    bits = 1 << torch.arange(32, dtype=torch.int64)
    for row, slot in enumerate(slots):
        prompt_end, end = int(prompts[slot]), int(prefills[slot])
        mask = torch.zeros(prompt_mask.shape[1] * 32, dtype=torch.int64)
        mask[ids[row, :prompt_end]] = 1
        prompt_mask[slot].copy_((mask.view(-1, 32) * bits).sum(-1).to(torch.int32))
        counts[slot].copy_(
            torch.bincount(ids[row, prompt_end:end], minlength=counts.shape[1])
        )


def apply_penalties(
    logits,
    mapping,
    token_ids,
    local_pos,
    repetition,
    frequency,
    presence,
    prompt_mask,
    counts,
):
    slots = mapping.long()
    vocab = torch.arange(logits.shape[-1], device=logits.device)
    seen_prompt = (
        (prompt_mask[slots[:, None], vocab[None, :] // 32] >> (vocab % 32)) & 1
    ).bool()
    output_counts = counts[slots]
    seen_output = output_counts > 0
    penalty = repetition[slots, None]
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    logits.copy_(torch.where(seen_prompt | seen_output, penalized, logits))


def apply_min_tokens(
    logits,
    mapping,
    pos,
    n_allowed,
    allowed,
    n_bias,
    bias_ids,
    bias,
    min_lens,
    n_stop,
    restore,
    stops,
    check_all_masked_rows=False,
):
    slots = cpu_mirror(mapping).tolist()
    positions = pos.cpu().tolist()
    stop_counts = cpu_mirror(n_stop)
    minimums = cpu_mirror(min_lens)
    restores = cpu_mirror(restore)
    for row, slot in enumerate(slots):
        count = int(stop_counts[slot])
        if count and positions[row] + 1 < int(minimums[slot]):
            indices = stops[slot, :count].long()
            saved = logits[row, indices].clone()
            logits[row, indices] = -torch.inf
            if (
                check_all_masked_rows
                and bool(restores[slot])
                and bool(torch.isneginf(logits[row]).all())
            ):
                logits[row, indices] = saved


def gumbel_sample(
    logits,
    mapping,
    temperature,
    seed,
    pos,
    apply_temperature,
    is_drafting,
    logits_cache=None,
    logits_cache_col=None,
    use_fp64=False,
):
    if is_drafting or logits_cache is not None or use_fp64:
        raise NotImplementedError("MPS Omni sampling requires ordinary fp32 sampling")
    slots = cpu_mirror(mapping).tolist()
    temperatures = cpu_mirror(temperature)
    seeds = cpu_mirror(seed)
    positions = pos.cpu().tolist()
    samples = []
    for row, slot in enumerate(slots):
        value = float(temperatures[slot])
        if value == 0:
            samples.append(logits[row].argmax().reshape(1))
            continue
        scores = logits[row].float()
        if apply_temperature:
            scores = scores / value
        # Include position so a seed is independent of scheduler row order and
        # slot reuse. CUDA uses a different stateless RNG; draws need not match.
        draw_seed = (int(seeds[slot]) ^ (int(positions[row]) * 0x9E3779B97F4A7C15)) & (
            (1 << 64) - 1
        )
        generator = torch.Generator(device=logits.device).manual_seed(draw_seed)
        samples.append(torch.multinomial(scores.softmax(-1), 1, generator=generator))
    return torch.cat(samples)


def install():
    from vllm.v1.worker.gpu.sample import logit_bias, penalties, sampler, states

    states.apply_temperature = apply_temperature
    penalties.bincount = bincount
    penalties.apply_penalties = apply_penalties
    logit_bias.apply_logit_bias = apply_min_tokens
    sampler.gumbel_sample = gumbel_sample


def validate_request(params):
    from vllm.exceptions import VLLMValidationError

    unsupported = [
        name
        for name in (
            "logprobs",
            "logprob_token_ids",
            "prompt_logprobs",
            "structured_outputs",
            "thinking_token_budget",
        )
        if getattr(params, name, None) is not None
    ]
    unsupported.extend(
        name
        for name, enabled in (
            ("min_p", params.min_p != 0),
            ("frequency_penalty", params.frequency_penalty != 0),
            ("presence_penalty", params.presence_penalty != 0),
            ("logit_bias", bool(params.logit_bias)),
            ("allowed_token_ids", bool(params.allowed_token_ids)),
            ("bad_words", bool(params.bad_words)),
        )
        if enabled
    )
    if unsupported:
        raise VLLMValidationError(f"MPS Omni does not support {', '.join(unsupported)}")
