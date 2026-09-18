"""Single-modal ECG data: Dataset and time-shift augmentation."""
import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """ECG classification dataset. ecg: (N, leads, L); labels: (N,) or None."""

    def __init__(self, ecg: np.ndarray, labels: np.ndarray | None = None):
        self.ecg = torch.from_numpy(np.asarray(ecg, dtype=np.float32))
        self.labels = None if labels is None else torch.from_numpy(np.asarray(labels, dtype=np.float32))

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, i):
        x = self.ecg[i]
        if self.labels is None:
            return x
        return x, self.labels[i]


def time_shift_batch(x: torch.Tensor, max_shift_ratio: float = 0.2) -> torch.Tensor:
    """SimCLR augmentation: apply a random circular time shift to each ECG in the batch."""
    b, c, l = x.shape
    max_shift = max(1, int(l * max_shift_ratio))
    shifts = torch.randint(-max_shift, max_shift + 1, (b,), device=x.device)
    out = torch.empty_like(x)
    for i in range(b):
        out[i] = torch.roll(x[i], shifts=int(shifts[i]), dims=-1)
    return out
