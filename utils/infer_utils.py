import torch
import torch.nn.functional as F


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


def decode_bounds_to_alignment(bounds):
    bounds_step = bounds.cumsum(dim=1).round().long()
    bounds_inc = torch.diff(
        bounds_step, dim=1, prepend=torch.full(
            (bounds.shape[0], 1), fill_value=-1,
            dtype=bounds_step.dtype, device=bounds_step.device
        )
    ) > 0
    frame2item = bounds_inc.long().cumsum(dim=1)
    return frame2item


def decode_note_sequence(frame2item, values, masks, threshold=0.5):
    """

    :param frame2item: [1, 1, 1, 1, 2, 2, 3, 3, 3]
    :param values:
    :param masks:
    :param threshold: minimum ratio of unmasked frames required to be regarded as an unmasked item
    :return: item_values, item_dur, item_masks
    """
    b = frame2item.shape[0]
    space = frame2item.max() + 1

    item_dur = frame2item.new_zeros(b, space).scatter_add(
        1, frame2item, torch.ones_like(frame2item)
    )[:, 1:]
    item_unmasked_dur = frame2item.new_zeros(b, space).scatter_add(
        1, frame2item, masks.long()
    )[:, 1:]
    item_masks = item_unmasked_dur / item_dur >= threshold

    values_quant = values.round().long()
    histogram = frame2item.new_zeros(b, space * 128).scatter_add(
        1, frame2item * 128 + values_quant, torch.ones_like(frame2item) * masks
    ).unflatten(1, [space, 128])[:, 1:, :]
    item_values_center = histogram.argmax(dim=2).to(dtype=values.dtype)
    values_center = torch.gather(F.pad(item_values_center, [1, 0]), 1, frame2item)
    values_near_center = masks & (values >= values_center - 0.5) & (values <= values_center + 0.5)
    item_valid_dur = frame2item.new_zeros(b, space).scatter_add(
        1, frame2item, values_near_center.long()
    )[:, 1:]
    item_values = values.new_zeros(b, space).scatter_add(
        1, frame2item, values * values_near_center
    )[:, 1:] / (item_valid_dur + (item_valid_dur == 0))

    return item_values, item_dur, item_masks


# if __name__ == '__main__':
#     frame2item = torch.LongTensor([
#         [1, 1, 1, 1, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0],
#         [1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 4, 0, 0, 0]
#     ])
#     values = torch.FloatTensor([
#         [60, 61, 60.5, 63, 57, 57, 50, 55, 54, 0, 0, 0, 0, 0],
#         [50, 51, 50.5, 53, 47, 47, 40, 45, 44, 38, 38, 0, 0, 0]
#     ])
#     masks = frame2item > 0
#     decode_note_sequence(frame2item, values, masks)
