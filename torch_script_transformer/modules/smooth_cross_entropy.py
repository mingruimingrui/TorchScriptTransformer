import torch


class SmoothCrossEntropyLoss(torch.nn.Module):

    def __init__(self, num_labels, smoothing=0.0, reduction='mean'):
        super().__init__()
        assert reduction in {'none', 'mean', 'sum'}
        self.num_labels = num_labels
        self.confidence = 1 - smoothing
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, labels):
        lprobs = logits.float().log_softmax(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(lprobs)
            true_dist.fill_(self.smoothing / (self.num_labels - 1))
            true_dist.scatter_(1, labels.data.unsqueeze(1), self.confidence)
        smooth_loss = -(true_dist * lprobs).sum(-1)
        if self.reduction == 'mean':
            smooth_loss = smooth_loss.mean()
        elif self.reduction == 'sum':
            smooth_loss = smooth_loss.sum()
        return smooth_loss
