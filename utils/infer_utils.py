import torch


def decode_gaussian_blurred_probs(probs, vmin, vmax, deviation, threshold):
    num_bins = probs.shape[-1]
    interval = (vmax - vmin) / (num_bins - 1)
    width = int(3 * deviation / interval)  # 3 * sigma
    idx = torch.arange(num_bins, device=probs.device)[None, None, :]  # [1, 1, N]
    idx_values = idx * interval + vmin
    center = torch.argmax(probs, dim=-1, keepdim=True)  # [B, T, 1]
    start = torch.clip(center - width, min=0)  # [B, T, 1]
    end = torch.clip(center + width + 1, max=num_bins)  # [B, T, 1]
    idx_masks = (idx >= start) & (idx < end)  # [B, T, N]
    weights = probs * idx_masks  # [B, T, N]
    product_sum = torch.sum(weights * idx_values, dim=2)  # [B, T]
    weight_sum = torch.sum(weights, dim=2)  # [B, T]
    values = product_sum / (weight_sum + (weight_sum == 0))  # avoid dividing by zero, [B, T]
    rest = probs.max(dim=-1)[0] < threshold  # [B, T]
    return values, rest


def decode_bounds_to_sequence(bounds):
    bounds_step = bounds.cumsum(dim=1).round().long()
    bounds_inc = torch.diff(
        bounds_step, dim=1, prepend=torch.full(
            (bounds.shape[0], 1), fill_value=-1,
            dtype=bounds_step.dtype, device=bounds_step.device
        )
    ) > 0
    frame2seq = bounds_inc.long().cumsum(dim=1)
    return frame2seq
