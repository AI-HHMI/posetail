"""Precision-weighted triplet ranking loss for the track-quality scorer.

Ported from miss-alignment (bioRxiv 2026.04.29.721716,
`miss_alignment/models/models.py::TripletMarginRankingLoss`). Adapted to operate on
per-point scores: each tracked point contributes one (good, bad, anchor) triplet, so the
loss batch is N = b * k triplets. Convention: **lower score = cleaner** track.

The class mirrors `losses.TotalLoss`'s logging interface (a defaultdict(list) history with
`collapse_history`/`reset_history`) so train_scorer can log to wandb the same way.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TripletScorerLoss(nn.Module):
    """Precision-weighted triplet margin ranking loss over per-point score triplets.

    Inputs are stacked per triplet as `[N, 3]` with the three columns ordered
    (good, bad, anchor) and labels in {+1 (clean), -1 (corrupt)}: good is always +1, bad
    always -1, anchor inherits its source's label. The two same-label entries form the
    "close" pair (their scores should match -> invariance); the odd one is "distant" and
    is pushed to the correct side by the margin. Identical algebra to the reference port.
    """

    def __init__(self, margin=0.5, precision_reg_weight=0.01, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.precision_reg_weight = precision_reg_weight
        self.reduction = reduction
        self.loss_history = defaultdict(list)

    def forward(self, scores, log_precisions, labels):
        """scores/log_precisions/labels: each `[N, 3]` (columns = good, bad, anchor).

        Returns the scalar total loss (triplet + precision_reg) and appends per-call
        scalars to `loss_history` for logging.
        """
        precisions = log_precisions.exp()

        # The majority sign per row identifies the same-label ("close") pair; the odd one
        # out is "distant". With good=+1, bad=-1, anchor=±1 each row sums to ±1.
        example_type = rearrange(labels.sum(dim=-1), 'n -> n 1')
        close_mask = labels == example_type
        distant_mask = labels != example_type

        close_scores = rearrange(scores[close_mask], '(n p) -> n p', p=2)
        distant_scores = rearrange(scores[distant_mask], 'n -> n 1')
        close_prec = rearrange(precisions[close_mask], '(n p) -> n p', p=2)
        distant_prec = rearrange(precisions[distant_mask], 'n -> n 1')

        # close pair should agree; distant should sit `margin` past it on the correct side
        # (example_type sign selects the direction: cleaner = lower score).
        dist_pos = torch.abs(close_scores[..., 0] - close_scores[..., 1])
        dist_neg = torch.min((close_scores - distant_scores) * example_type, dim=-1).values
        triplet_losses = F.relu(dist_pos + dist_neg + self.margin)

        # downweight triplets where any member is uncertain (geometric mean of precisions)
        all_prec = torch.cat([close_prec, distant_prec], dim=-1)
        triplet_precision = all_prec.prod(dim=-1).pow(1.0 / 3.0)
        weighted = triplet_losses * triplet_precision

        if self.reduction == 'sum':
            triplet_loss = weighted.sum()
        else:
            triplet_loss = weighted.mean()

        # keep precision from collapsing toward zero
        precision_reg = -log_precisions.mean() * self.precision_reg_weight
        total = triplet_loss + precision_reg

        # ---- logging metrics (good=col 0, bad=col 1 by construction) ----
        good_s, bad_s = scores[:, 0], scores[:, 1]
        with torch.no_grad():
            triplet_acc = (good_s < bad_s).float().mean()
            score_gap = (bad_s - good_s).mean()                  # >0 means good is cleaner
            self.loss_history['scorer_loss'].append(float(total))
            self.loss_history['triplet_loss'].append(float(triplet_loss))
            self.loss_history['precision_reg'].append(float(precision_reg))
            self.loss_history['triplet_acc'].append(float(triplet_acc))
            self.loss_history['score_gap'].append(float(score_gap))
            self.loss_history['score_good'].append(float(good_s.mean()))
            self.loss_history['score_bad'].append(float(bad_s.mean()))
            self.loss_history['mean_precision'].append(float(precisions.mean()))

        return total

    def collapse_history(self, prefix=''):
        summary = {}
        for name, losses in self.loss_history.items():
            if len(losses):
                summary[f'{prefix}{name}'] = float(np.nanmean(losses))
        return summary

    def reset_history(self):
        self.loss_history = defaultdict(list)
