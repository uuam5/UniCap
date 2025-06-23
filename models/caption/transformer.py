import os
import random
from collections import namedtuple
import itertools
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from zmq import device

import cider.pyciderevalcap.ciderD.ciderD as ciderD
from datasets.caption.field import TextField
from engine.iterative_refinement_generator import DecoderOut
from engine.utils import NestedTensor
from models.caption.base import BaseCaptioner
from models.caption.levenshtein_utils import (_apply_del_words,
                                              _apply_ins_masks,
                                              _apply_ins_words,
                                              _apply_reposition_words, _fill,
                                              _get_advanced_ins_targets,
                                              _get_advanced_reposition_targets,
                                              _get_del_targets,
                                              _get_ins_targets, _skip,
                                              _skip_encoder_out,
                                              _apply_unify)
from natlib.utils import new_arange
import json

class UnifyTransformer(BaseCaptioner):
    def __init__(self,
                 grid_net,
                 cap_generator,
                 bos_idx=2,
                 detector=None,
                 use_gri_feat=True,
                 use_reg_feat=False,
                 cached_features=False,
                 sampling_for_deletion=False,
                 config=None,
                 reg_net=None,
                 ):
        super(UnifyTransformer, self).__init__()
        self.bos_idx = bos_idx
        self.eos_idx = 3
        self.pad_idx=1
        self.unk_idx=0
        self.eos=self.eos_idx
        self.bos=self.bos_idx
        self.pad=self.pad_idx
        self.unk=self.unk_idx
        self.max_iter = 2
        self.max_ratio = 2
        self.eos_penalty =0
        self.adaptive =True
        self.retain_history=False
        self.retain_dropout = False
        self.reranking = False
        self.decoding_format=None
        
        self.grid_net = grid_net
        self.use_reg_feat = use_reg_feat
        self.use_gri_feat = use_gri_feat
        self.cached_features = cached_features
        self.config = config
        
        if self.use_gri_feat:
            self.register_state('gri_feat', None)
            self.register_state('gri_mask', None)

        if self.use_reg_feat:
            self.register_state('reg_feat', None)
            self.register_state('reg_mask', None)
        self.cap_generator = cap_generator
        self.init_weights()
        self.detector = detector
        

        self.sampling_for_deletion = sampling_for_deletion
        self.text_field = TextField(vocab_path=config.dataset.vocab_path)
        if config.model.use_bert:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        if True or config.model.use_cider :
            self.cider_scorer=ciderD.CiderD(df=os.path.join(config.dataset.ann_root,"coco-train-idxs.p"))
            print(f"created cider scorer with {os.path.join(config.dataset.ann_root,'coco-train-idxs.p')}")
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_bs_device(self, samples):
        if isinstance(samples, dict):
            key = 'gri_feat' if 'gri_feat' in samples else 'reg_feat'
            batch_size = samples[key].shape[0]
            device = samples[key].device
        elif isinstance(samples, NestedTensor):
            batch_size = samples.tensors.shape[0]
            device = samples.tensors.device
        return batch_size, device

    def init_state(self, batch_size, device):
        return [torch.zeros((batch_size, 0), dtype=torch.long, device=device), None, None]

    def encoder_out_select(self, encoder_out,mask_select):
        if 'gri_feat' in encoder_out:
            return {
                'gri_feat': encoder_out['gri_feat'][mask_select],
                'gri_mask': encoder_out['gri_mask'][mask_select],
                'reg_feat': encoder_out['reg_feat'][mask_select],
                'reg_mask': encoder_out['reg_mask'][mask_select],
            }
        else:
            return {
                'reg_feat': encoder_out['reg_feat'][mask_select],
                'reg_mask': encoder_out['reg_mask'][mask_select],
            }
    def forward_decoder(
        self,
        decoder_out,
        images,
        eos_penalty=0.0,
        max_ratio=None,
        is_random=False,
        **kwargs
    ):

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history
        total_reposition_ops, total_deletion_ops, total_insertion_ops=0,0,0
        bsz = output_tokens.size(0)
        if max_ratio is None:
            max_lens = output_tokens.new(bsz).fill_(20).long()
        else:
            max_lens = output_tokens.new(bsz).fill_(20).long()
        unify_score = self.cap_generator.forward_unify(
            normalize=True,
            input=output_tokens,
            vis_inputs=images,
        )
        unify_score, unify_pred = unify_score.max(-1)
        _tokens, _scores,_ = _apply_unify(#9ms
            output_tokens,
            output_scores,
            unify_pred=unify_pred,
            unify_score=unify_score,
            bos_idx=self.bos_idx,
            eos_idx=self.eos_idx,
            pad_idx=self.pad_idx,
            unk_idx=self.unk_idx,
        )
        output_tokens=_tokens
        output_scores=_scores
        can_ins_word = output_tokens.eq(self.unk_idx).sum(1) > 0
        if can_ins_word.sum() != 0:
            prev_score=output_scores[can_ins_word].clone()
            unk_mask=output_tokens[can_ins_word].eq(self.unk_idx)
            unk_or_eos_mask=unk_mask
            
            word_ins_score = self.cap_generator.forward_mask_pred(#14ms
                normalize=True,
                input=output_tokens[can_ins_word],
                vis_inputs=self.encoder_out_select(images,can_ins_word),
            )


            if is_random:
                prob=torch.exp(word_ins_score).view(-1, word_ins_score.size(-1))
                prob[:,self.eos_idx]=0
                prob[:,self.bos_idx]=0
                prob[:,self.pad_idx]=0
                word_ins_pred = torch.multinomial(
                    prob, 1
                ).view(word_ins_score.size(0), -1,1)
                word_ins_score=torch.gather(word_ins_score,-1,word_ins_pred).squeeze(-1)
            else:
                prob=torch.exp(word_ins_score.clone().detach())
                prob[:,:,self.eos_idx]=0
                prob[:,:,self.bos_idx]=0
                prob[:,:,self.pad_idx]=0
                _, word_ins_pred = prob.max(-1)
                word_ins_score=torch.gather(word_ins_score,-1,word_ins_pred.unsqueeze(-1)).squeeze(-1)


            _tokens, _scores = _apply_ins_words(#6ms
                output_tokens[can_ins_word],
                output_scores[can_ins_word],
                word_ins_pred,
                word_ins_score,
                self.unk_idx,
            )
            _tokens=_tokens.masked_scatter(~unk_or_eos_mask,output_tokens[can_ins_word][~unk_or_eos_mask])
            _scores=_scores.masked_scatter(~unk_or_eos_mask,output_scores[can_ins_word][~unk_or_eos_mask])
            if not output_tokens.eq(self.eos_idx).sum()-output_tokens.shape[0]==0:
                print("error!!!!!!!!!!!!!!!!!!!!!")

            output_tokens = _fill(output_tokens, can_ins_word, _tokens, self.pad_idx)
            output_scores = _fill(output_scores, can_ins_word, _scores, 0)
            attn=None
            if history is not None:
                history.append(output_tokens.clone())

        cut_off = output_tokens.ne(self.pad_idx).sum(1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        attn = None if attn is None else attn[:, :cut_off, :]
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history,
        )
    def get_unify_targets(self,word_predictions, tgt_tokens, pad_idx,eos_idx):
        def align_words(s1_words, s2_words):
            s2_word_positions = {}
            for i, word in enumerate(s2_words):
                if word in s2_word_positions:
                    s2_word_positions[word].append(i)
                else:
                    s2_word_positions[word]=[i]
                if word==eos_idx:
                    break
            alignment = []
            for word in s1_words:
                if word in s2_word_positions:
                    if len(s2_word_positions[word]) == 0:
                        alignment.append(0)
                        s2_word_positions.pop(word)
                    else:
                        alignment.append(s2_word_positions[word].pop(0))
                else:
                    alignment.append(0)
            return alignment
        batch_size = word_predictions.size(0)
        target = torch.zeros_like(word_predictions)
        for i in range(batch_size):
            s1_words = word_predictions[i].tolist()
            s2_words = tgt_tokens[i].tolist()
            alignment = align_words(s1_words, s2_words)
            target[i] = torch.tensor(alignment)
        return target

    def forward(self,
                images,
                tgt_tokens,
                prev_output_tokens=None,
                gt_raw=None,
                mode="",
                ref_tokens=None,
                **kwargs):
        if not self.cached_features:
            vis_inputs = self.detector(images)
        else:
            vis_inputs = images
        if self.config.model.use_gri_feat:
            gri_feat, _ = self.grid_net(vis_inputs['gri_feat'], attention_mask=vis_inputs['gri_mask'])
            vis_inputs['gri_feat'] = gri_feat[:, -1]
        if mode=="ref":
            raise NotImplementedError
        else:

            if mode=='sc':
                masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_advanced_ins_targets(
                    prev_output_tokens, tgt_tokens, self.pad_idx, self.unk_idx
                )
                word_ins_out = self.cap_generator.forward_mask_pred(
                    normalize=False,
                    input=masked_tgt_tokens,
                    vis_inputs=vis_inputs
                )# shape : [batch_size, seq_len, vocab_size]
                word_probs=F.softmax(word_ins_out, dim=-1).max(-1)[0] # shape:[batch_size, seq_len]
                word_threshold_mask=word_probs.gt(0.7)# shape:[batch_size, seq_len]

                
                sample_num=5
                sample_predictions = torch.multinomial(
                    F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), sample_num,replacement=True
                ).T.view(sample_num,word_ins_out.size(0),word_ins_out.size(1)).permute(1,0,2).contiguous().view(-1,word_ins_out.size(1))


                word_prediction_log_prob=F.log_softmax(word_ins_out, dim=-1)
                temp_masked_tgt_masks=masked_tgt_masks.repeat_interleave(sample_num,0)
                temp_tgt_tokens=tgt_tokens.repeat_interleave(sample_num,0)

                sample_predictions=sample_predictions.masked_scatter_(
                    ~temp_masked_tgt_masks, temp_tgt_tokens[~temp_masked_tgt_masks]
                )
                sample_text=self.text_field.decode(sample_predictions[:,1:])

                gt_raw = list(itertools.chain(*([c] * sample_num for c in gt_raw)))  # [c,]
                gt_cider_input={i:gt_raw[i] for i in range(len(gt_raw))}

                sample_cider_input=[{"image_id":i,"caption":[sample_text[i]]} for i in range(len(sample_text))]
                sample_cider_score=self.cider_scorer.compute_score(gt_cider_input,sample_cider_input)[1]

                sample_cider_score=torch.from_numpy(sample_cider_score).float().to(word_ins_out.device)
                sample_cider_score_mean=sample_cider_score.reshape(-1,sample_num).mean(-1,keepdim=True)
                reward=sample_cider_score.reshape(-1,sample_num)-sample_cider_score_mean#
                word_prediction_log_prob=word_prediction_log_prob.repeat_interleave(sample_num,0)
                word_prediction_log_prob=word_prediction_log_prob.gather(-1,sample_predictions.unsqueeze(-1)).squeeze(-1)
                word_threshold_mask=word_threshold_mask.repeat_interleave(sample_num,0)
                word_prediction_log_prob[word_threshold_mask]=0
                reward = -word_prediction_log_prob.mean(dim=-1) * reward.view(-1)
                reward=reward.mean()
                res={}
                res['word_ins_reward']={
                    "reward":reward,
                }
                res['log']=sample_cider_score.mean().item()
                res['log_jsonl']={
                    'prediction_mean_reward':0,
                    'sample_mean_reward':sample_cider_score_mean.mean().item(),
                    }
                return res   




            #NOT SC, USING PATHES:
            if mode=='12random':
                rate1=0.5
                rate2=0.5
            elif mode=='2only':
                rate1=1
                rate2=0
            elif mode=='1only':
                rate1=0
                rate2=0
            elif mode=='normal':#mode=123
                rate1=0.5
                rate2=0.6
            elif mode=='13random':
                rate1=0
                rate2=0.5
            else:
                raise NotImplementedError
            
            random_value=random.random()
            if random_value<=rate1:#PATH2
                masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_advanced_ins_targets(
                    prev_output_tokens, tgt_tokens, self.pad_idx, self.unk_idx
                )
                mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
                mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad_idx)
                word_ins_out = self.cap_generator.forward_mask_pred(
                    normalize=False,
                    input=masked_tgt_tokens,
                    vis_inputs=vis_inputs
                )
                if self.sampling_for_deletion:
                    word_predictions = torch.multinomial(
                        F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1
                    ).view(word_ins_out.size(0), -1)
                else:
                    word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

                word_predictions.masked_scatter_(
                    ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
                )
                
                
                unify_masks=word_predictions.ne(self.pad_idx)
                unify_targets=self.get_unify_targets(word_predictions, tgt_tokens, self.pad_idx,self.eos_idx)

                unify_output=self.cap_generator.forward_unify(
                    normalize=False,
                    input=word_predictions,
                    vis_inputs=vis_inputs
                )
                res={
                    "word_ins": {
                        "out": word_ins_out,
                        "tgt": tgt_tokens,
                        "mask": masked_tgt_masks,
                        "ls":0,
                        "nll_loss": True,
                    },
                    "unify": {
                        "out": unify_output,
                        "tgt": unify_targets,
                        "mask": unify_masks,
                    },
                }
            elif random_value<=rate2:#PATH3
                unify_targets=self.get_unify_targets(tgt_tokens, tgt_tokens, self.pad_idx,self.eos_idx)
                unify_output=self.cap_generator.forward_unify(
                    normalize=False,
                    input=tgt_tokens,
                    vis_inputs=vis_inputs
                )
                unify_masks=tgt_tokens.ne(self.pad_idx)

                masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_advanced_ins_targets(
                    prev_output_tokens, tgt_tokens, self.pad_idx, self.unk_idx
                )
                mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
                mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad_idx)
                word_ins_out = self.cap_generator.forward_mask_pred(
                    normalize=False,
                    input=masked_tgt_tokens,
                    vis_inputs=vis_inputs
                )
                res={
                    "word_ins": {
                        "out": word_ins_out,
                        "tgt": tgt_tokens,
                        "mask": masked_tgt_masks,
                        "ls":0,
                        "nll_loss": True,
                    },
                    "unify": {
                        "out": unify_output,
                        "tgt": unify_targets,
                        "mask": unify_masks,
                    },
                }
            else:#PATH1
                unify_targets=self.get_unify_targets(prev_output_tokens, tgt_tokens, self.pad_idx,self.eos_idx)

                unify_output=self.cap_generator.forward_unify(
                    normalize=False,
                    input=prev_output_tokens,
                    vis_inputs=vis_inputs
                )
                unify_masks=prev_output_tokens.ne(self.pad_idx)

                unify_pred=F.log_softmax(unify_output, dim=-1).max(-1)[1]
                
                res={
                    "unify": {
                        "out": unify_output,
                        "tgt": unify_targets,
                        "mask": unify_masks,
                    },
                }
        return res
        
    def initialize_output_tokens(self, encoder_out,refs):
        bs,device=self.get_bs_device(encoder_out)
        if refs is not None:
            initial_output_tokens=refs.to(device)
        else:
            initial_output_tokens = torch.zeros((bs, 2), dtype=torch.long, device=device)
            initial_output_tokens[:, 0] = self.bos_idx
            initial_output_tokens[:, 1] = self.eos_idx


        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out['reg_feat'])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=[],
        )
    
    def generate_sc(self, sample,refs=None, prefix_tokens=None, constraints=None,return_inter=False,random_steps=0):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the IterativeRefinementGenerator is not supported"
            )
        encoder_out = sample
        inter=[]
        
        prev_decoder_out = self.initialize_output_tokens(encoder_out,refs)
        bsz=prev_decoder_out.output_tokens.size(0)

        sent_idxs = torch.arange(bsz).to(prev_decoder_out.output_tokens.device)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()

        if self.retain_history:
            prev_decoder_out = prev_decoder_out._replace(history=[prev_output_tokens])

        finalized = [[] for _ in range(bsz)]

        def is_a_loop(x, y, s, a):
            b, l_x, l_y = x.size(0), x.size(1), y.size(1)
            if l_x > l_y:
                y = torch.cat([y, x.new_zeros(b, l_x - l_y).fill_(self.pad)], 1)
                s = torch.cat([s, s.new_zeros(b, l_x - l_y)], 1)
                if a is not None:
                    a = torch.cat([a, a.new_zeros(b, l_x - l_y, a.size(2))], 1)
            elif l_x < l_y:
                x = torch.cat([x, y.new_zeros(b, l_y - l_x).fill_(self.pad)], 1)
            return (x == y).all(1), y, s, a
        def finalized_hypos(step, prev_out_token, prev_out_score, prev_out_attn):
            cutoff = prev_out_token.ne(self.pad)
            tokens = prev_out_token[cutoff]
            if prev_out_score is None:
                scores, score = None, None
            else:
                scores = prev_out_score[cutoff]
                score = scores.mean()

            if prev_out_attn is None:
                hypo_attn, alignment = None, None
            else:
                hypo_attn = prev_out_attn[cutoff]
                alignment = hypo_attn.max(dim=1)[1]
            return {
                "steps": step,
                "tokens": tokens,
                "positional_scores": scores,
                "score": score,
                "hypo_attn": hypo_attn,
                "alignment": alignment,
            }


        step_count=0
        for step in range(self.max_iter + 1):
            decoder_options = {
                "eos_penalty": self.eos_penalty,
                "max_ratio": self.max_ratio,
                "decoding_format": self.decoding_format,
            }
            prev_decoder_out = prev_decoder_out._replace(
                step=step,
                max_step=self.max_iter + 1,
            )


            if step<random_steps:
                decoder_out = self.forward_decoder(
                    prev_decoder_out, encoder_out,is_random=True, **decoder_options
                )
            else:
                decoder_out = self.forward_decoder(
                    prev_decoder_out, encoder_out,is_random=False, **decoder_options
                )

            inter.append(decoder_out.output_tokens)
            step_count+=1

            if self.adaptive:
                terminated, out_tokens, out_scores, out_attn = is_a_loop(
                    prev_output_tokens,
                    decoder_out.output_tokens,
                    decoder_out.output_scores,
                    decoder_out.attn,
                )
                decoder_out = decoder_out._replace(
                    output_tokens=out_tokens,
                    output_scores=out_scores,
                    attn=out_attn,
                )
                lenght_terminated=decoder_out.output_tokens.size(1)>22
                terminated=terminated|lenght_terminated
            else:
                terminated = decoder_out.output_tokens.new_zeros(
                    decoder_out.output_tokens.size(0)
                ).bool()
            if step == self.max_iter:  # reach last iteration, terminate
                terminated.fill_(1)
            finalized_idxs = sent_idxs[terminated]
            finalized_tokens = decoder_out.output_tokens[terminated]
            finalized_scores = decoder_out.output_scores[terminated]
            finalized_attn = (
                None
                if (decoder_out.attn is None or decoder_out.attn.size(0) == 0)
                else decoder_out.attn[terminated]
            )

            if self.retain_history:
                finalized_history_tokens = [h[terminated] for h in decoder_out.history]

            for i in range(finalized_idxs.size(0)):
                finalized[finalized_idxs[i]] = [
                    finalized_hypos(
                        step,
                        finalized_tokens[i],
                        finalized_scores[i],
                        None if finalized_attn is None else finalized_attn[i],
                    )
                ]

                if self.retain_history:
                    finalized[finalized_idxs[i]][0]["history"] = []
                    for j in range(len(finalized_history_tokens)):
                        finalized[finalized_idxs[i]][0]["history"].append(
                            finalized_hypos(
                                step, finalized_history_tokens[j][i], None, None
                            )
                        )
            if terminated.sum() == terminated.size(0):
                break
            not_terminated = ~terminated
            prev_decoder_out = decoder_out._replace(
                output_tokens=decoder_out.output_tokens[not_terminated],
                output_scores=decoder_out.output_scores[not_terminated],
                attn=decoder_out.attn[not_terminated]
                if (decoder_out.attn is not None and decoder_out.attn.size(0) > 0)
                else None,
                history=[h[not_terminated] for h in decoder_out.history]
                if decoder_out.history is not None
                else None,
            )
            encoder_out=self.encoder_out_select(encoder_out,not_terminated)
            sent_idxs = sent_idxs[not_terminated]
            prev_output_tokens = prev_decoder_out.output_tokens.clone()
        if return_inter:
            return [out[0]['tokens'][1:] for out in finalized], [out[0]['positional_scores'][1:-1] for out in finalized],inter
        else:
            return [out[0]['tokens'][1:] for out in finalized], [out[0]['positional_scores'][1:-1] for out in finalized]
    
    
    
    
    def select(self, t, candidate_logprob, beam_size, **kwargs):
        raise NotImplementedError
    def _expand_state(self, selected_beam, cur_beam_size, batch_size, beam_size):
        raise NotImplementedError
    def step(self, timestep, prev_output, samples, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError
    def iter(self, timestep, samples, outputs, return_probs, batch_size, beam_size=5, eos_idx=3, **kwargs):
        raise NotImplementedError
