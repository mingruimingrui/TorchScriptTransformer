import torch


class SmoothCrossEntropyLoss(torch.nn.Module):

    def __init__(self, smoothing=0.0, reduction='mean'):
        super().__init__()
        assert reduction in {'none', 'mean', 'sum'}
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, labels):
        lprobs = logits.float().log_softmax(-1)
        if labels.dim() == lprobs.dim() - 1:
            labels = labels.unsqueeze(-1)
        nll_loss = -lprobs.gather(index=labels, dim=-1).squeeze(-1)
        smooth_loss = -lprobs.mean(keepdim=False, dim=-1)
        loss = nll_loss * self.confidence + \
            smooth_loss * self.smoothing
        if self.reduction == 'mean':
            nll_loss = nll_loss.mean()
            loss = loss.mean()
        elif self.reduction == 'sum':
            nll_loss = nll_loss.sum()
            loss = loss.sum()
        return loss, nll_loss
