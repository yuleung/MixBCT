import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch import distributed

class CompatibleLearningLoss(nn.Module):
    def __init__(self, num_features, num_classes, queue_size, temp=1.0):
        super(CompatibleLearningLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.temp = temp

        assert (
            distributed.is_initialized()
        ), "must initialize distributed before create this"
        self.rank = distributed.get_rank()
        self.world_size = distributed.get_world_size()


        self.register_buffer("feat_queue", torch.zeros(self.queue_size, self.num_features))
        self.register_buffer("logit_queue", torch.zeros(self.queue_size, self.num_classes))
        self.register_buffer("labels", -1 * torch.ones(self.queue_size, dtype=int))
        self.header = 0

    def forward(self, old_embeds, old_logits, new_embeds, new_logits, labels):
        N = new_embeds.shape[0]
        bz = N // self.world_size
        # update queue
        new_embeds = F.normalize(new_embeds)
        for embed, logit, label in zip(old_embeds, old_logits, labels):
            self.feat_queue[self.header] = embed
            self.logit_queue[self.header] = logit
            self.labels[self.header] = label
            self.header = (self.header + 1) % self.queue_size

        new_embeds = new_embeds[self.rank * bz:(self.rank + 1) * bz]
        old_embeds = old_embeds[self.rank * bz:(self.rank + 1) * bz]
        new_logits = new_logits[self.rank * bz:(self.rank + 1) * bz]
        #old_logits = old_logits[self.rank * bz:(self.rank + 1) * bz]
        labels = labels[self.rank * bz:(self.rank + 1) * bz]

        outputs = new_embeds.mm(self.feat_queue.t())  # BxN
        outputs /= self.temp

        outputs_max, _ = torch.max(outputs, dim=1, keepdim=True)
        outputs = outputs - outputs_max.detach()

        old_outputs = old_embeds.mm(self.feat_queue.t())
        weight = 0.5 * (old_outputs + 1)
        mask = self.labels == (labels.reshape(-1, 1))  # B*N  #find positive?

        exp_outputs = torch.exp(outputs)


        log_prob = outputs - torch.log(exp_outputs.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob * weight).sum(1) / mask.sum(1)
        l1_loss = -mean_log_prob_pos.mean()

        logits = new_logits.mm(self.logit_queue.t())
        logits /= self.temp
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob * weight).sum(1) / mask.sum(1)
        l2_loss = -mean_log_prob_pos.mean()

        return l1_loss, l2_loss