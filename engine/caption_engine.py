# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------

import itertools
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.functional import F
from torch.nn import NLLLoss
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from datasets.caption import metrics
from engine.iterative_refinement_generator import IterativeRefinementGenerator
from engine.utils import NestedTensor
from natlib.utils import new_arange
from utils import noising

def build_optimizers(model, config, mode='xe'):
    model = getattr(model, 'module', model)

    no_decay = ['bias', 'gamma', 'beta']

    model_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' not in n and any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': 0.0
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' not in n and not any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': config.optimizer.weight_decay
        },
    ]

    backbone_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' in n and any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': 0.0
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if p.requires_grad and 'detector' in n and not any(nd in n for nd in no_decay)
            ],
            'weight_decay_rate': config.optimizer.weight_decay
        },
    ]

    optimizers = {
        'model':
            torch.optim.Adam(
                model_parameters,
                lr=getattr(config.optimizer, f'{mode}_lr', config.optimizer.sc_lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
            ),
        'backbone':
            torch.optim.Adam(
                backbone_parameters,
                lr=getattr(config.optimizer, f'{mode}_backbone_lr', config.optimizer.sc_backbone_lr),
                betas=(config.optimizer.beta_1, config.optimizer.beta_2),
            ),
        'mode':
            mode
    }
    return optimizers

def gather_result(value):
    if isinstance(value, torch.Tensor):
        torch.distributed.all_reduce(value, async_op=False)  # compute the sum
        value.mul_(1.0 / torch.distributed.get_world_size())  # compute the avg
    return value

def save_checkpoint(
    model,
    optimizers,
    epoch,
    scores,
    best_ciders,
    config=None,
    filename='checkpoint_last.pth',
    scheduler=None,
):
    torch.save(
        {
            "state_dict": model.module.state_dict(),
            "optim_model": optimizers['model'].state_dict(),
            "optim_backbone": optimizers['backbone'].state_dict(),
            "scores": scores,
            "best_ciders": best_ciders,
            "epoch": epoch,
            "exp_name": "" if config is None else config.exp.name,
            "scheduler": [] if scheduler is None else scheduler.state_dict(),
        }, filename)

def log_epoch(config, writer, epoch, train_res, split, scores, which='ft_xe'):
    """For better logging and viewing the log file.
    Run the command in terminal: 
    >>> column -t -s, result.csv
    """
    head = 'exp, backbone, imsize, resize, raug, epoch, split, cider, B1, B4, R, M, B2, B3, t-loss, t-reward, b-reward, which, v-loss'

    if epoch == 0 and not os.path.exists('result.csv'):
        with open('result.csv', 'w') as f:
            f.write(head + '\n')

    with open('result.csv', 'a') as f:
        text = f'{config.exp.name.split("/")[-1]}, '
        backbone = 'B-'
        backbone += 'VG' if os.path.exists(config.model.detector.checkpoint) else 'IM'
        text += f'{backbone}, '
        text += f'{config.dataset.transform_cfg.size[0]}_{config.dataset.transform_cfg.size[1]}, '
        text += f'{config.dataset.transform_cfg.resize_name}, {config.dataset.transform_cfg.randaug}, '
        text += f'{epoch}, {split:<5}, '
        text += f'{scores["CIDEr"]*100:3.2f}, {scores["BLEU"][0]*100:3.2f}, '
        text += f'{scores["BLEU"][3]*100:3.2f}, {scores["ROUGE"]*100:3.2f}, '
        text += f'{train_res["loss"]:2.2f}, {train_res["reward"]:2.2f}, {train_res["reward_baseline"]:2.2f}, '
        text += f'{which}, {train_res["val_loss"]:1.2f}'
        f.write(text + '\n')
        print(text)

    writer.add_scalar(f'{split}_cider', scores['CIDEr'], epoch)
    writer.add_scalar(f'{split}_bleu1', scores['BLEU'][0], epoch)
    writer.add_scalar(f'{split}_bleu4', scores['BLEU'][3], epoch)
    writer.add_scalar(f'{split}_rouge', scores['ROUGE'], epoch)

    writer.add_scalar(f'train_loss', train_res['loss'], epoch)
    writer.add_scalar(f'train_reward', train_res['reward'], epoch)
    writer.add_scalar(f'train_reward_baseline', train_res['reward_baseline'], epoch)

def evaluate_metrics_levt(
    model,
    optimizers,
    dataloader,
    text_field,
    epoch=0,
    split='test',
    config=None,
    train_res=None,
    writer=None,
    best_cider=None,
    which='ft_xe',
    scheduler=None,
    log_and_save=True,
):
    generator=IterativeRefinementGenerator(
        unk=0,
        pad=1,
        bos = 2,
        eos = 3,
        vocab_size=text_field.vocab,
        max_iter=2,
        retain_history=True
    )
    model.eval()
    gen, gts = {}, {}
    unk_idx=0
    pad_idx=1
    bos_idx=2
    eos_idx=3
    config_max_len=54

    counter = 0
    times = []
    decode_times=[]
    batch_size=dataloader.batch_size
    total_steps=[]
    with tqdm(desc=f'Epoch {epoch} - evaluation on {split}', unit='it', total=len(dataloader)) as pbar:
        results = []
        for it, batch in enumerate(iter(dataloader)):
            counter += 1
            with torch.no_grad():
                if 'refs' in batch:
                    refs = [c[:config_max_len] for c in batch['refs']]
                    max_len = max([len(c) for c in batch['refs']])

                    padded = []
                    for c in refs:
                        ref = [bos_idx] + c + [eos_idx] + [pad_idx] * (max_len - len(c))
                        padded.append(ref)

                    padded = [torch.Tensor(ref).long() for ref in padded]
                    padded = pad_sequence(padded, batch_first=True)
                    batch['refs'] = padded
                
                start_it = time.time()
                if not model.module.cached_features:
                    batch['samples'] = model.module.detector(batch['samples'])
                if 'gri_feat' in batch['samples']:
                    gri_feat, _ = model.module.grid_net(batch['samples']['gri_feat'], attention_mask=batch['samples']['gri_mask'])
                    batch['samples']['gri_feat'] = gri_feat[:, -1]
                mid_it = time.time()
                if config.model.use_ref:
                    out,step_count=generator.generate(
                        [model],
                        batch['samples'],
                    )
                else:
                    out,step_count=generator.generate(
                        [model],
                        batch['samples'],
                    )
            end_it = time.time()
            torch.cuda.synchronize()
            times.append(end_it - start_it)
            decode_times.append(end_it-mid_it)
            total_steps.append(step_count)
            if 'samples' in batch and not isinstance(batch['samples'], dict):
                bs = batch['samples'].tensors.shape[0]
            else:
                bs = batch['samples']['reg_feat'].shape[0]
            if it % 100 == 0:
                print(f"Number of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times)/counter:0.5f}s, Decode time per 1 batch: {sum(decode_times)/counter:0.5f}s, decode iter per 1 batch: {sum(total_steps)/counter:0.5f}")
            out=[t[1:] for t in out]
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(batch['captions'], caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen[f'{it}_{i}'] = [gen_i]
                gts[f'{it}_{i}'] = gts_i
                res = {'image_id': batch['image_id'][i], 'caption': gen_i}
                results.append(res)
            pbar.update()

    avg_time = sum(times) / counter
    avg_decode_time=sum(decode_times)/counter
    avg_step_count=sum(total_steps)/counter
    print(f"Epoch: {epoch} iters: {counter}\n"+
        f"Total time per 1 batch: {avg_time:0.5f}s,batch size:{batch_size},time per 1 item: {avg_time/batch_size}"+
        f"decode time per 1 item: {avg_decode_time/batch_size},decode iter per 1 item: {avg_step_count/batch_size}")
    gts = metrics.PTBTokenizer.tokenize(gts)
    gen = metrics.PTBTokenizer.tokenize(gen)
    scores, _ = metrics.compute_scores(gts, gen)
    print(f'Epoch {epoch}: {split} scores: ' + str(scores) + '\n')

    if log_and_save:
        with open('result.txt', 'a') as f:
            f.write(f'Epoch {epoch}: {split} scores: ' + str(scores) + '\n')
        log_epoch(config, writer, epoch, train_res, split=split, scores=scores, which=which)

        if scores['CIDEr'] >= best_cider:
            best_ciders = (scores['CIDEr'], 0) if split == 'valid' else (0, scores['CIDEr'])
            save_checkpoint(
                model,
                optimizers=optimizers,
                epoch=epoch,
                scores=scores,
                best_ciders=best_ciders,
                config=config,
                filename=f'checkpoint_best_{split}.pth',
                scheduler=scheduler,
            )
            best_cider = scores['CIDEr']
        return best_cider
    else:
        return scores

def inference_coco_test(
    model,
    dataloader,
    text_field,
    epoch=0,
    split='test',
    config=None,
):
    model.eval()
    gen, gts = {}, {}

    counter = 0
    times = []
    with tqdm(desc=f'Epoch {epoch} - evaluation on {split}', unit='it', total=len(dataloader)) as pbar:

        results = []
        for it, batch in enumerate(iter(dataloader)):
            counter += 1
            start_it = time.time()
            with torch.no_grad():
                out, _ = model(
                    batch['samples'],
                    seq=None,
                    use_beam_search=True,
                    max_len=config.model.beam_len,
                    eos_idx=config.model.eos_idx,
                    beam_size=config.model.beam_size,
                    out_size=1,
                    return_probs=False,
                )
            torch.cuda.synchronize()
            end_it = time.time()
            times.append(end_it - start_it)

            if 'samples' in batch:
                bs = batch['samples'].tensors.shape[0]
            elif 'vis_feat' in batch:
                bs = batch['vis_feat'].shape[0]
            if it % 100 == 0:
                print(
                    f"Number of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times)/counter:0.5f}s"
                )

            caps_gen = text_field.decode(out, join_words=False)
            for i, gen_i in enumerate(caps_gen):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                res = {'image_id': batch['image_id'][i], 'caption': gen_i}
                results.append(res)
            pbar.update()

    with open(f'result_{split}.json', 'w') as f:
        json.dump(results, f)

def inference_coco_test_levt(
    model,
    dataloader,
    text_field,
    epoch=0,
    split='test',
    config=None,
):
    model.eval()
    gen, gts = {}, {}

    counter = 0
    times = []
    with tqdm(desc=f'Epoch {epoch} - evaluation on {split}', unit='it', total=len(dataloader)) as pbar:

        results = []
        for it, batch in enumerate(iter(dataloader)):
            counter += 1
            start_it = time.time()
            batch['samples'] = model.module.detector(batch['samples'])
            gri_feat, _ = model.module.grid_net(batch['samples']['gri_feat'], attention_mask=batch['samples']['gri_mask'])
            batch['samples']['gri_feat'] = gri_feat[:, -1]
            with torch.no_grad():
                out, _ = model.module.generate(
                        batch['samples'],
                    )
            torch.cuda.synchronize()
            end_it = time.time()
            times.append(end_it - start_it)

            if 'samples' in batch:
                bs = batch['samples']['gri_feat'].shape[0]
            elif 'vis_feat' in batch:
                bs = batch['vis_feat'].shape[0]
            if it % 100 == 0:
                print(
                    f"Number of iterations: {counter}, batch_size={bs}, Total time per 1 batch: {sum(times)/counter:0.5f}s"
                )
            out=[t[1:] for t in out]
            caps_gen = text_field.decode(out, join_words=False)
            for i, gen_i in enumerate(caps_gen):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                res = {'image_id': batch['image_id'][i], 'caption': gen_i}
                results.append(res)
            pbar.update()

    with open(f'result_{split}.json', 'w') as f:
        json.dump(results, f)

def evaluate_loss(model, dataloader, loss_fn, text_field, epoch, writer):
    model.eval()

    running_loss = .0
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                out = model(batch['samples'], batch['captions'])

                captions_gt = batch['captions'][:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

                loss = gather_result(loss)
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    if dist.get_rank() == 0:
        writer.add_scalar('val_loss', val_loss, epoch)
    return val_loss


class TgtDict(object):
    def __init__(self,unk=0,pad=1,bos=2,eos=3):
        self._unk = unk
        self._pad = pad
        self._bos = bos
        self._eos = eos
    def unk(self):
        return self._unk
    def pad(self):
        return self._pad
    def bos(self):
        return self._bos
    def eos(self):
        return self._eos

tgt_dict = TgtDict()


def inject_noise(target_tokens,noise,text_field=None):
    unk=0
    pad=1
    bos = 2
    eos = 3
    def delete_all(target_tokens):
        target_tokens = target_tokens.clone()
        target_tokens.fill_(pad)
        target_tokens[:, 0] = bos
        target_tokens[:, 1] = eos
        return target_tokens
        
    def _random_delete(target_tokens,delete_rate=1):
        max_len = target_tokens.size(1)
        target_mask = target_tokens.eq(pad)
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(
            target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
        )
        target_score.masked_fill_(target_mask, 1)
        target_score, target_rank = target_score.sort(1)
        target_length = target_mask.size(1) - target_mask.float().sum(
            1, keepdim=True
        )
        target_cutoff = (
            2
            + (
                (target_length - 2)
                * (target_score.new_zeros(target_score.size(0), 1).uniform_()*delete_rate).clamp(0, 1)
            ).long()
        )
        target_cutoff = target_score.sort(1)[1] >= target_cutoff

        prev_target_tokens = (
            target_tokens.gather(1, target_rank)
            .masked_fill_(target_cutoff, pad)
            .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
        )
        prev_target_tokens = prev_target_tokens[
            :, : prev_target_tokens.ne(pad).sum(1).max()
        ]

        return prev_target_tokens

    def _random_shuffle(target_tokens, p, max_shuffle_distance):
        word_shuffle = noising.WordShuffle(tgt_dict,bpe_cont_marker=None,bpe_end_marker=None)
        target_mask = target_tokens.eq(tgt_dict.pad())
        target_length = target_mask.size(1) - target_mask.long().sum(1)
        prev_target_tokens, _ = word_shuffle.noising(
            target_tokens.t().cpu(), target_length.cpu(), max_shuffle_distance)
        prev_target_tokens = prev_target_tokens.to(target_tokens.device).t()
        masks = (target_tokens.clone().sum(dim=1, keepdim=True).float()
            .uniform_(0, 1) < p)
        prev_target_tokens = masks * prev_target_tokens + (~masks) * target_tokens
        return prev_target_tokens
    def _random_mask(target_tokens):
        target_masks = (
            target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
        )
        target_score = target_tokens.clone().float().uniform_()
        target_score.masked_fill_(~target_masks, 2.0)
        target_length = target_masks.sum(1).float()
        target_length = target_length * target_length.clone().uniform_()
        target_length = target_length + 1  # make sure to mask at least one token.

        _, target_rank = target_score.sort(1)
        target_cutoff = new_arange(target_rank) < target_length[:, None].long()
        prev_target_tokens = target_tokens.masked_fill(
            target_cutoff.scatter(1, target_rank, target_cutoff), unk
        ) 
        return prev_target_tokens

    
    def _full_mask(target_tokens):
        target_mask = (
            target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
        )
        return target_tokens.masked_fill(~target_mask, unk)
    if noise == "random_delete":
        return _random_delete(target_tokens)
    elif noise == 'random_delete_shuffle':
        return _random_shuffle(_random_delete(target_tokens), 0.5, 3)
    elif noise == "sc_random_delete_shuffle":
        return _random_shuffle(_random_delete(target_tokens,1.5), 0.5, 3)
    elif noise == "random_mask":
        return _random_mask(target_tokens)
    elif noise == "full_mask":
        return _full_mask(target_tokens)
    elif noise == "no_noise":
        return target_tokens
    else:
        raise NotImplementedError



def gaussian(x, cen, wid):
    return (1/(np.sqrt(2*np.pi)*wid)) * np.exp(-(x - cen)**2 / (2 * wid**2))
#def a empty class
class DualLoss:
    def __init__(self,label_smoothing=0.0):
        self.label_smoothing=label_smoothing
        max_len=54
        self.cached_unify = np.zeros((max_len, max_len))
        np.fill_diagonal(self.cached_unify, gaussian(0,0,0.9))
        distance=5
        for i in range(1, max_len):
            j=i
            for k in range(1,distance+1):
                if j+k<self.cached_unify.shape[1]:
                    self.cached_unify[i,j+k]=gaussian(k, 0, 0.9)
                if j-k>0:
                    self.cached_unify[i,j-k]=gaussian(k, 0, 0.9)
        cached_unify_sum=self.cached_unify.sum(axis=1)
        self.cached_unify=self.cached_unify/cached_unify_sum
        
        
    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0,loss_fn=None,loss_weight=None
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """
        def mean_ds(x: torch.Tensor, dim=None) -> torch.Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            if loss_weight is not None:
                loss_weight=loss_weight.unsqueeze(1).expand_as(targets)[masks]
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = loss_fn(logits, targets.to(logits.device))

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device),reduction='batchmean')*2
            nll_loss=losses
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}
    def __call__(self, outputs):
        pass
def evaluate_loss_levt(model, dataloader,loss_obj, text_field, epoch, writer,config):
    model.eval()
    running_loss = .0
    mode=config.optimizer.xe_policy
    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                if config.model.use_ref:
                    if random.random()<0.5:
                        batch['prev_output_tokens'] = inject_noise(batch['refs'],"random_delete_shuffle")
                        if batch['prev_output_tokens'].shape[1]>batch['captions'].shape[1]:
                            batch['prev_output_tokens']=inject_noise(batch['captions'], "random_delete_shuffle")
                    else:
                        batch['prev_output_tokens']=inject_noise(batch['captions'], "random_delete_shuffle")
                else:
                    batch['prev_output_tokens'] = inject_noise(batch['captions'], "random_delete_shuffle")
                outputs = model(batch['samples'], batch['captions'],prev_output_tokens=batch['prev_output_tokens'],gt_raw=batch['gt_tokenized_captions'],mode=mode)
                loss_fns={
                    "mask_ins":NLLLoss(),
                    "word_ins":NLLLoss(ignore_index=1),
                    "word_del":NLLLoss(),
                    "ori_word_del":NLLLoss(),
                    'word_reposition':NLLLoss(),
                    'unify':NLLLoss(),
                }
                losses, nll_loss = [], []
                loss=0
                for obj in outputs:
                    if outputs[obj].get("loss", None) is None:
                        _losses = loss_obj._compute_loss(
                            outputs[obj].get("out"),
                            outputs[obj].get("tgt"),
                            outputs[obj].get("mask", None),
                            outputs[obj].get("ls", 0.0),
                            name=obj + "-loss",
                            factor=outputs[obj].get("factor", 1.0),
                            loss_fn=loss_fns[obj]
                        )
                    else:
                        _losses = loss_obj._custom_loss(
                            outputs[obj].get("loss"),
                            name=obj + "-loss",
                            factor=outputs[obj].get("factor", 1.0),
                        )
                    losses += [_losses]
                    loss+=_losses['nll_loss']
                    if outputs[obj].get("nll_loss", False):
                        nll_loss += [_losses.get("nll_loss", 0.0)]
                loss = gather_result(loss)
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    if dist.get_rank() == 0:
        writer.add_scalar('val_loss', val_loss, epoch)
    return val_loss
def train_xe_levt(
    model,
    dataloaders,
    optimizers,
    text_field,
    epoch,
    rank=0,
    config=None,
    scheduler=None,
    writer=None,
):
    model.train()

    loss_obj = DualLoss()
    if scheduler is not None:
        scheduler.step()
    running_loss = .0
    with tqdm(desc=f'Epoch {epoch} - train', unit='it', total=len(dataloaders['train'])) as pbar:
        for it, batch in enumerate(dataloaders['train']):
            mode=config.optimizer.xe_policy
            batch['ref_tokens']=None
            if config.model.use_ref:
                if random.random()<0.5:
                    batch['prev_output_tokens']=batch['refs']
                    batch['ref_tokens']=batch['refs']
                    mode="ref"
                    outputs = model(batch['samples'], batch['captions'],prev_output_tokens=batch['prev_output_tokens'],
                                    gt_raw=batch['gt_tokenized_captions'],mode=mode,
                                    ref_tokens=batch['ref_tokens'])

                else:
                    batch['prev_output_tokens']=inject_noise(batch['distillation'], "random_delete_shuffle")
                    batch['ref_tokens']=batch['refs']
                    outputs = model(batch['samples'], batch['distillation'],prev_output_tokens=batch['prev_output_tokens'],
                                    gt_raw=batch['gt_tokenized_captions'],mode=mode,
                                    ref_tokens=batch['ref_tokens'])
            else:
                if random.random()<config.exp.distillation_ratio:
                    batch['prev_output_tokens'] = inject_noise(batch['distillation'],"random_delete_shuffle")
                    outputs = model(batch['samples'], batch['distillation'],prev_output_tokens=batch['prev_output_tokens'],
                                    gt_raw=batch['gt_tokenized_captions'],mode=mode,
                                    ref_tokens=batch['ref_tokens'])
                else:
                    batch['prev_output_tokens'] = inject_noise(batch['captions'],"random_delete_shuffle")
                    outputs = model(batch['samples'], batch['captions'],prev_output_tokens=batch['prev_output_tokens'],
                                    gt_raw=batch['gt_tokenized_captions'],mode=mode,
                                    ref_tokens=batch['ref_tokens'])
            
            optimizers['model'].zero_grad()
            optimizers['backbone'].zero_grad()
            loss=0
            loss_fns={
                "mask_ins":NLLLoss(),
                "word_ins":NLLLoss(ignore_index=1),
                "word_del":NLLLoss(),
                'ori_word_del':NLLLoss(),
                'word_reposition':NLLLoss(),
                'word_reposition_reward':NLLLoss(),
                'unify':NLLLoss(),
            }
            for obj in outputs:
                if outputs[obj].get("loss", None) is None:
                    if (not obj.endswith("reward")):
                        _losses = loss_obj._compute_loss(
                            outputs[obj].get("out"),
                            outputs[obj].get("tgt"),
                            outputs[obj].get("mask", None),
                            outputs[obj].get("ls", 0.0),
                            name=obj + "-loss",
                            factor=outputs[obj].get("factor", 1.0),
                            loss_fn=loss_fns[obj],
                        )
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
                loss+=_losses['nll_loss']
            
            loss.backward()
            optimizers['model'].step()
            optimizers['backbone'].step()

            loss = gather_result(loss)
            running_loss += loss.item()

            pbar.set_postfix(loss=running_loss / (it + 1),mode=mode)
            pbar.update()

            if scheduler is not None:
                lr = scheduler.step()
                assert optimizers['model'].param_groups[0]['lr'] == lr, "LR scheduler doesn't work properly."

            if rank == 0:
                writer.add_scalar(
                    'backbone_lr',
                    optimizers['backbone'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train']) + it,
                )
                writer.add_scalar(
                    'model_lr',
                    optimizers['model'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train']) + it,
                )
                lr = optimizers['model'].param_groups[0]['lr']

    val_loss = evaluate_loss_levt(model, dataloaders['valid'], loss_obj, text_field, epoch, writer,config)

    if rank == 0:
        save_checkpoint(
            model=model,
            optimizers=optimizers,
            epoch=epoch,
            scores=[],
            best_ciders=(0, 0),
            config=config,
            filename='checkpoint_last.pth',
            scheduler=scheduler,
        )
    torch.distributed.barrier()

    return {
        'loss': running_loss / len(dataloaders['train']),
        'reward': 0,
        'reward_baseline': 0,
        'val_loss': val_loss,
    }
def train_sc_levt(
    model,
    dataloaders,
    optimizers,
    text_field,
    epoch,
    rank=0,
    config=None,
    scheduler=None,
    writer=None,
):
    model.train()
    loss_obj = DualLoss()
    if scheduler is not None:
        scheduler.step()
    running_loss = .0
    noise_choices = ["random_delete_shuffle", "no_noise"]
    with tqdm(desc=f'Epoch {epoch} - train', unit='it', total=len(dataloaders['train'])) as pbar:
        for it, batch in enumerate(dataloaders['train']):
            if it==1500:
                break
            mode=""
            sc_or_xe='sc'
            batch['ref_tokens']=None
            if config.model.use_ref:
                raise NotImplementedError
            else:
                if random.random()<0.9:
                    mode=config.optimizer.sc_policy
                    sc_or_xe='sc'
                    noise_mode="sc_random_delete_shuffle"
                else:#trick: do some xe training in sc stage
                    mode=config.optimizer.xe_policy
                    sc_or_xe='xe'
                    noise_mode="random_delete_shuffle"
            if random.random()<config.exp.distillation_ratio:
                batch['prev_output_tokens'] = inject_noise(batch['distillation'],noise_mode)
                outputs = model(batch['samples'], batch['distillation'],prev_output_tokens=batch['prev_output_tokens'],
                                gt_raw=batch['gt_tokenized_captions'],mode=mode,
                                ref_tokens=batch['ref_tokens'])
            else:
                batch['prev_output_tokens'] = inject_noise(batch['captions'],noise_mode)
                outputs = model(batch['samples'], batch['captions'],prev_output_tokens=batch['prev_output_tokens'],
                                gt_raw=batch['gt_tokenized_captions'],mode=mode,
                                ref_tokens=batch['ref_tokens'])
            optimizers['model'].zero_grad()
            optimizers['backbone'].zero_grad()

            if sc_or_xe=='sc':
                losses, nll_loss = [], []
                loss=0
                for obj in outputs:
                    if obj.endswith("reward"):
                        reward=outputs[obj].get("reward")
                        del_loss=reward
                        _losses={
                            "nll_loss":del_loss
                        }
                        loss+=_losses['nll_loss']
                loss.backward()
                optimizers['model'].step()
                optimizers['backbone'].step()

                loss = gather_result(loss)
                running_loss += outputs['log']
                pbar.set_postfix(loss_now= outputs['log'],loss=running_loss / (it + 1),mode=mode)
                with open('result.jsonl', 'a') as f:
                    log_data={'step':epoch * len(dataloaders['train']) + it,'reward':outputs['log_jsonl']}
                    f.write(json.dumps(log_data)+'\n')
            elif sc_or_xe=='xe':
                losses, nll_loss = [], []
                loss=0
                loss_fns={
                    "mask_ins":NLLLoss(),
                    "word_ins":NLLLoss(ignore_index=1),
                    "word_del":NLLLoss(),
                    'ori_word_del':NLLLoss(),
                    'word_reposition':NLLLoss(),
                    'word_reposition_reward':NLLLoss(),
                    'unify':NLLLoss(),
                }
                loss_weight=None
                for obj in outputs:
                    if outputs[obj].get("loss", None) is None:
                        if (not obj.endswith("reward")):
                            _losses = loss_obj._compute_loss(
                                outputs[obj].get("out"),
                                outputs[obj].get("tgt"),
                                outputs[obj].get("mask", None),
                                outputs[obj].get("ls", 0.0),
                                name=obj + "-loss",
                                factor=outputs[obj].get("factor", 1.0),
                                loss_fn=loss_fns[obj],
                            )
                        else:
                            raise NotImplementedError
                            reward=outputs[obj].get("reward")
                            del_loss=reward
                            _losses={
                                "nll_loss":del_loss
                            }
                    else:
                        raise NotImplementedError
                    loss+=_losses['nll_loss']*0.1
                
                loss.backward()
                optimizers['model'].step()
                optimizers['backbone'].step()

                loss = gather_result(loss)
                running_loss += running_loss/(it+1)

                pbar.set_postfix(loss=running_loss / (it + 1),mode=mode)
            pbar.update()

            if rank == 0:
                writer.add_scalar(
                    'backbone_lr',
                    optimizers['backbone'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train']) + it,
                )
                writer.add_scalar(
                    'model_lr',
                    optimizers['model'].param_groups[0]['lr'],
                    epoch * len(dataloaders['train']) + it,
                )
                if 'log' in outputs:
                    writer.add_scalar(
                        'my_reward',
                        outputs['log'],
                        epoch * len(dataloaders['train']) + it,
                    )

    val_loss = evaluate_loss_levt(model, dataloaders['valid'], loss_obj, text_field, epoch, writer,config)

    if rank == 0:
        save_checkpoint(
            model=model,
            optimizers=optimizers,
            epoch=epoch,
            scores=[],
            best_ciders=(0, 0),
            config=config,
            filename='checkpoint_last.pth',
            scheduler=scheduler,
        )
    torch.distributed.barrier()

    return {
        'loss': running_loss / len(dataloaders['train']),
        'reward': 0,
        'reward_baseline': 0,
        'val_loss': val_loss,
    }
