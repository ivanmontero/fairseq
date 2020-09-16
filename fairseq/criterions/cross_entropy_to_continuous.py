# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F
import torch

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('cross_entropy_to_continuous')
class CrossEntropyToContinuousCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.prev_pred = None   
        self.prev_pred_diff_prev = 0

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--continuous-delay', default=4000., type=int,
                            help='delay until switching to continuous')
        parser.add_argument('--continuous-warmup', default=4000., type=float, metavar='D',
                            help='linear warmup period to continuous')
        parser.add_argument('--continuous-max', default=1.0, type=float, metavar='D',
                            help='delay until switching to continuous')
        # fmt: on


    # def get_continuous_prob(self, update_num):

    # def __init__(self, task, sentence_avg, label_smoothing):
    #     super().__init__(task)
    #     self.sentence_avg = sentence_avg
    #     self.eps = label_smoothing

    # @staticmethod
    # def add_args(parser):
    #     """Add criterion-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
    #                         help='epsilon for label smoothing, 0 means no label smoothing')
    #     # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])  # features_only=True
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        if model.num_updates % 1000 == 0:
            curr_pred = model.decoder.output_projection.weight.detach().clone()
            prev_pred_diff = 0 if self.prev_pred is None else 1 - F.cosine_similarity(curr_pred, self.prev_pred, dim=-1).mean()
            self.prev_pred = curr_pred
            self.prev_pred_diff_prev = prev_pred_diff
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'prev_pred_diff': self.prev_pred_diff_prev
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        pre_logits = net_output[0]  # (batch, tgt_len, vocab)

        similarity = pre_logits.reshape(-1, pre_logits.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        # filter out padding
        idx = (target != self.padding_idx)
        similarity = similarity[idx]
        target = target[idx]

        # Figure out signs
        sign = torch.ones_like(similarity).float()
        sign.scatter_(-1, target.unsqueeze(1), -1.0)
        
        # Compute cosine similarity
        loss = (1.0 + sign * similarity)

        if reduce:
            loss = loss.sum()
        
        return loss, loss

        # 1 - cs(x, y) if y = 1 else 1 + cs(x, y)

        # lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        # loss = F.nll_loss(
        #     lprobs,
        #     target,
        #     ignore_index=self.padding_idx,
        #     reduction='sum' if reduce else 'none',
        # )
        # return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        sum_prev_pred_diff = sum(log.get('prev_pred_diff', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('prev_pred_diff', sum_prev_pred_diff / len(logging_outputs), 0)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            # metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        # else:
            # metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
