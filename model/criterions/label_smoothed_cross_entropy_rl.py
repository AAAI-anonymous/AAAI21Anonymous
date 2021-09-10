# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from typing import Dict, List, Optional
from torch import Tensor
from .rougescore import *

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def get_rouge2f(references, hypotheses):
    references = [[char for char in "".join(ref)] for ref in references]
    hypotheses = [[char for char in "".join(hyp)] for hyp in hypotheses]
    #print('references', references)
    #print('hypotheses', hypotheses)
    # compute ROUGE-2 F1-SCORE
    rouge2f_score = rouge_2_corpus(references, hypotheses)
    #print('rg2', rouge2f_score)
    return rouge2f_score


@register_criterion(
    "label_smoothed_cross_entropy_rl", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyRLCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        
        #print('img', sample["net_input"]['img_vec_tokens'])
        #print('kb_repeat', sample["net_input"]['kb_repeat_tokens'].shape)
        #print('kb_repeat', sample["net_input"]['kb_repeat_tokens'])
        #print('kb_distinct', sample["net_input"]['kb_distinct_tokens'].shape)
        #print('kb_distinct', sample["net_input"]['kb_distinct_tokens'])

        example_losses, sample_ys = self.decode_sample(model, sample)
        base_ys = self.decode_baseline(model, sample)
        #print('sample["target"]', sample["target"])
        #print(eeee)
        tgt_sents = []
        #help(model.encoder.dictionary)
        for tgt_ids in sample["target"]:
            sentence = model.encoder.dictionary.string(tgt_ids)
            sentence = sentence.replace(" ", "").replace("##", " ").strip()
            sentence = sentence.replace("[PAD]", "")
            sentence = sentence.replace("[SEP]", "")
            tgt_sents.append(sentence)
            
        kb_repeat_list = []
        for kb_ids in sample["net_input"]["kb_repeat_tokens"]:
            sentence = model.encoder.dictionary.string(kb_ids)
            sentence = sentence.replace(" ", "").replace("##", " ").strip()
            sentence = sentence.replace("[PAD]", "")
            kb_repeat_list.append([val.strip() for val in sentence.strip().split('[SEP]') if val.strip()])
                
        #print('kb_repeat_list', kb_repeat_list)
        kb_distinct_list = []
        for kb_ids in sample["net_input"]["kb_distinct_tokens"]:
            sentence = model.encoder.dictionary.string(kb_ids)
            sentence = sentence.replace(" ", "").replace("##", " ").strip()
            sentence = sentence.replace("[PAD]", "")
            kb_distinct_list.append([val.strip() for val in sentence.strip().split('[SEP]') if val.strip()])

        #print('kb_distinct_list', kb_distinct_list)
        #print('tgt_sents', tgt_sents)
        reward_rouge_tgt = []
        bs = len(sample_ys)
        #print('example_losses', example_losses.shape)
        for idx in range(bs):
            rouge_sample = get_rouge2f([[tgt_sents[idx]]], [sample_ys[idx]])
            rouge_base = get_rouge2f([[tgt_sents[idx]]], [base_ys[idx]])

            num_repeat_sample = 0.
            num_repeat_base = 0.
            for val in kb_repeat_list[idx]:
                if val and val in sample_ys[idx]:
                    num_repeat_sample += len(val)
                if val and val in base_ys[idx]:
                    num_repeat_base += len(val)

            num_distinct_sample = 0.
            num_distinct_base = 0.
            for val in kb_distinct_list[idx]:
                if val and val in sample_ys[idx]:
                    num_distinct_sample += len(val)
                if val and val in base_ys[idx]:
                    num_distinct_base += len(val)

            rd_kb_sample = (num_repeat_sample * 2.0  + num_distinct_sample) / len(sample_ys[idx])
            rd_kb_base = (num_repeat_base * 2.0  + num_distinct_base) / len(base_ys[idx])

            #print(kb_repeat_list[idx])
            #print(kb_distinct_list[idx])
            #print('sample_ys[idx]', sample_ys[idx])
            #print('base_ys[idx]', base_ys[idx])
            #print('tgt', tgt_sents[idx])
            #print('rouge_sample_tgt', rouge_sample)
            #print('rouge_base_tgt', rouge_base)
            #print('rd_kb', rd_kb)
            reward_rouge_tgt.append(rouge_sample - rouge_base + rd_kb_sample - rd_kb_base)
        #print(eee)
        reward_rouge = torch.tensor(reward_rouge_tgt, dtype=torch.float)
        reward_rouge = reward_rouge.cuda()
        #print('reward_rouge_tgt', reward_rouge.shape)
        loss_rouge = (example_losses * reward_rouge).sum()

        #print('loss', loss)
        #print(eee)
        ###########################
        
        net_output = model(**sample["net_input"])
        loss_mle, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        loss = loss_rouge * 0.9984 + 0.0016 * loss_mle
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        
        #print(eee)
        return loss, sample_size, logging_output

    def decode_sample(self, model, sample):
        
        #print('model', model.__dict__.keys())
        #print('model_en', model.encoder.__dict__.keys())
        #print('model_de', model.decoder.__dict__.keys())
        """
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                #for i in range(model.models_size)
                for i in range(6)
            ],
        )
        """
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
        elif "source" in net_input:
            src_tokens = net_input["source"]

        sample_ys = []  # 解码的采样集合 bs * seq_len
        sample_ids = []  # 解码的采样集合 bs * seq_len
        scores = []  # loss

        max_len = sample["target"].size(1)
        temperature = 0.1
        bsz, src_len = src_tokens.size()[:2]
        tokens = (
            torch.zeros(bsz, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(model.encoder.padding_idx)
        )  # +2 for eos and pad
        #bos_token = model.bos  # to do
        #tokens[:, 0] = model.eos if bos_token is None else bos_token
        #help(model.encoder.dictionary)
        #print('dictionary', model.encoder.dictionary)
        #print('dictionary', model.encoder.dictionary.eos_word)
        #print('dictionary', model.encoder.dictionary.bos_word)
        #print('dictionary', model.encoder.dictionary.pad_word)
        tokens[:, 0] = model.encoder.dictionary.index(model.encoder.dictionary.bos_word)

        #encoder_outs = model.forward_encoder(net_input)
        encoder_outs = model.encoder.forward_torchscript(net_input)

        #print(max_len)
        for step in range(max_len + 1):  # one extra step for EOS marker
            if step > 0:
                y_ids = torch.clone(sample_ids[-1].view(-1))
                #y_ids = sample_ids[-1].view(-1)
                #y_ids[y_ids >= len(self.vocab.tgt)] = self.vocab.tgt.word2id['<unk>']
                #y_tm1_embed = self.tgt_embed(y_ids)
                y_ids = torch.unsqueeze(y_ids, dim=-1)
                #print('tokens[:, step: step + 1]', tokens[:, step: step + 1])
                #print('y_ids', y_ids)
                #print('tokens[:, step: step + 1]', tokens[:, step: step + 1].shape)
                #print('y_ids', y_ids.shape)
                tokens_clone = tokens.clone()
                tokens_clone[:, step: step + 1] = y_ids
                tokens = tokens_clone

            lprobs, avg_attn_scores = model.decoder(
                tokens[:, : step + 1],
                encoder_outs,
                #incremental_states,
                img_vec_tokens=sample["net_input"]['img_vec_tokens']
            )
            lprobs = lprobs[:, -1, :]

            #print('lprobs', lprobs.shape)
            #print('lprobs', lprobs)
            probs = torch.softmax(lprobs, dim=-1)
            scores.append(probs)
            #print('probs', probs.shape)
            probs_sqz = torch.squeeze(probs)
            #print('probs', probs_sqz.shape)
            cur_y = torch.multinomial(probs_sqz, 1).view(-1)
            #print('cur_y', cur_y.shape)
            sample_ids.append(cur_y.view(-1,1))

        sample_ids = torch.stack(sample_ids)
        scores = torch.stack(scores)
        score_mask = torch.ones(scores.size(0), scores.size(1), dtype=torch.float).cuda()
        #print('sample_ids', sample_ids.shape)
        sample_ids_trans = sample_ids.transpose(0, 1).data.tolist()
        #sample_ys = model.encoder.dictionary.string(sample_ids_trans)
        for y in sample_ids_trans:
            cur_words = []
            for y_id in y:
                #y_id = y_id[0]
                y_word = model.encoder.dictionary.string(y_id)
                #print('y_word', y_word)
                if y_word != model.encoder.dictionary.eos_word:
                    cur_words.append(y_word)
                else:
                    break
            if cur_words == []:
                cur_words = ['xx']
            sentence = ''.join(cur_words)
            sentence = sentence.replace(" ", "").replace("##", " ").strip()
            sentence = sentence.replace("[SEP]", "")
            sample_ys.append(sentence)

        for idx, y in enumerate(sample_ys):
            score_mask[len(y)+1:, idx] = 0
        #print('sample_ys', sample_ys)
        scores = torch.gather(scores, index=sample_ids, dim=-1).squeeze(-1)
        scores = torch.log(scores) * score_mask
        #scores = scores.sum(dim=0) / torch.sum(score_mask, 0)  # (batch_size)
        #print('scores', scores)
        return -scores, sample_ys

    def decode_baseline(self, model, sample):
        
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                #for i in range(model.models_size)
                for i in range(6)
            ],
        )
        net_input = sample["net_input"]

        if "src_tokens" in net_input:
            src_tokens = net_input["src_tokens"]
        elif "source" in net_input:
            src_tokens = net_input["source"]

        base_ys = []  # 解码的采样集合 bs * seq_len
        base_ids = []  # 解码的采样集合 bs * seq_len
        temperature = 0.1
        bsz, src_len = src_tokens.size()[:2]
        max_len = sample["target"].size(1)

        tokens = (
            torch.zeros(bsz, max_len + 2)
            .to(src_tokens)
            .long()
            .fill_(model.encoder.padding_idx)
        )  # +2 for eos and pad
        #bos_token = model.bos  # to do
        #tokens[:, 0] = model.eos if bos_token is None else bos_token
        tokens[:, 0] = model.encoder.dictionary.index(model.encoder.dictionary.bos_word)

        #encoder_outs = model.forward_encoder(net_input)
        encoder_outs = model.encoder.forward_torchscript(net_input)

        for step in range(max_len + 1):  # one extra step for EOS marker
            if step > 0:
                y_ids = torch.clone(base_ids[-1].view(-1))
                #y_ids = sample_ids[-1].view(-1)
                #y_ids[y_ids >= len(self.vocab.tgt)] = self.vocab.tgt.word2id['<unk>']
                #y_tm1_embed = self.tgt_embed(y_ids)
                y_ids = torch.unsqueeze(y_ids, dim=-1)

                tokens[:, step: step + 1] = y_ids

            lprobs, avg_attn_scores = model.decoder(
                tokens[:, : step + 1],
                encoder_outs,
                #incremental_states,
                img_vec_tokens=sample["net_input"]['img_vec_tokens']
            )
            lprobs = lprobs[:, -1, :]
            probs = torch.softmax(lprobs, dim=-1)
            probs_sqz = torch.squeeze(probs)
            _, cur_y = torch.max(probs_sqz, 1)
            base_ids.append(cur_y.view(-1,1))

        base_ids = torch.stack(base_ids)
        base_ids_trans = base_ids.transpose(0, 1).data.tolist()
        for y in base_ids_trans:
            cur_words = []
            for y_id in y:
                #y_id = y_id[0]
                y_word = model.encoder.dictionary.string(y_id)
                #print('y_word', y_word)
                if y_word != model.encoder.dictionary.eos_word:
                    cur_words.append(y_word)
                else:
                    break
            if cur_words == []:
                cur_words = ['xx']
            sentence = ''.join(cur_words)
            sentence = sentence.replace(" ", "").replace("##", " ").strip()
            sentence = sentence.replace("[SEP]", "")
            base_ys.append(sentence)
        #print('base_ys', base_ys)
        return base_ys

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
