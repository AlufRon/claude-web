# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch


def multinomial(
    input: torch.Tensor, num_samples: int, replacement=False, *, generator=None
):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (torch.Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        torch.Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """
    input_ = input.reshape(-1, input.shape[-1])
    # We should probably be able to remove this once the following PR has landed:
    # https://github.com/pytorch/pytorch/pull/134818/files
    # In the meantime, we specialize the case no-replacement, nsamples=1 so as not
    # to have a synchronization point.
    if replacement or num_samples != 1:
        output_ = torch.multinomial(
            input_,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
    else:
        q = torch.empty_like(input_).exponential_(1, generator=generator)
        q = input_ / q
        output_ = q.argmax(dim=-1, keepdim=True)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    k = min(k, probs.shape[-1])
    probs, indices = torch.topk(probs, k, dim=-1)
    next_token = multinomial(probs, num_samples=1)
    next_token = indices.gather(-1, next_token)
    return next_token


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (torch.Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        torch.Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sample_token(
    logits: torch.Tensor,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    repetition_penalty: float = 1.0,
    recent_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Given logits of shape [*, Card], returns a LongTensor of shape [*].

    Args:
        logits: Logits tensor of shape [*, vocab_size]
        use_sampling: Whether to use sampling or greedy decoding
        temp: Temperature for sampling
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        repetition_penalty: Penalty for tokens that appear in recent_tokens (default 1.0 = disabled)
        recent_tokens: Recent token history of shape [*, window_size] for repetition penalty

    Returns:
        Sampled tokens of shape [*]
    """
    # Apply repetition penalty BEFORE temperature scaling
    if repetition_penalty != 1.0 and recent_tokens is not None:
        # recent_tokens shape: [*, window_size]
        # logits shape: [*, vocab_size]
        batch_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]

        # Flatten for processing
        logits_flat = logits.reshape(-1, vocab_size)  # [batch, vocab_size]
        tokens_flat = recent_tokens.reshape(-1, recent_tokens.shape[-1])  # [batch, window]

        # Apply penalty to each batch element
        for i in range(logits_flat.shape[0]):
            for token_id in tokens_flat[i]:
                token_id_item = token_id.item()
                if 0 <= token_id_item < vocab_size:  # Valid token
                    # Standard repetition penalty:
                    # if score < 0, multiply by penalty (make more negative)
                    # if score >= 0, divide by penalty (make smaller)
                    if logits_flat[i, token_id_item] < 0:
                        logits_flat[i, token_id_item] *= repetition_penalty
                    else:
                        logits_flat[i, token_id_item] /= repetition_penalty

        logits = logits_flat.reshape(*batch_shape, vocab_size)

    # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
    if use_sampling and temp > 0.0:
        probs = torch.softmax(logits / temp, dim=-1)
        if top_p > 0.0:
            next_token = sample_top_p(probs, p=top_p)
        elif top_k > 0:
            next_token = sample_top_k(probs, k=top_k)
        else:
            next_token = multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    assert next_token.shape[-1] == 1
    return next_token[..., 0]


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = "cpu"
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda:0"

    ps = torch.tensor([5.0, 2.0, 12.0, 6.0, 8.0, 1.0, 0.0, 4.0], device=device)
    cnts = torch.zeros(ps.shape, dtype=torch.long, device=device)
    total_samples = 1000
    for _ in range(total_samples):
        vs = multinomial(ps, num_samples=1, replacement=False)
        cnts[vs] += 1
    diff = cnts / cnts.sum() - ps / ps.sum()
    max_diff = diff.abs().max().cpu().item()
    print(ps / ps.sum())
    print(cnts / cnts.sum())
    assert max_diff < 1.5e-2
