
import re
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import rearrange, repeat
from models.common.attention import MultiHeadAttention,MaskHalfAttention
from models.common.pos_embed import sinusoid_encoding_table, PositionWiseFeedForward
from models.caption.containers import Module, ModuleList

import torch
import torch.nn as nn
import torch.nn.functional as F



class GeneratorLayer(Module):

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=.1,
        attn_dropout=0.0,
        self_att_module=None,
        self_att_module_kwargs=None,
        **kwargs,
    ):

        super().__init__()
        self.self_att = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attn_dropout=attn_dropout,
            can_be_stateful=True,
            attention_module=self_att_module,
            attention_module_kwargs=self_att_module_kwargs,
        )
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)


class ParallelAttentionLayer(GeneratorLayer):
    def __init__(
        self,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=.1,
        attn_dropout=0.0,
        self_att_module=None,
        enc_att_module=None,
        self_att_module_kwargs=None,
        enc_att_module_kwargs=None,
        activation='sigmoid',
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            self_att_module=self_att_module,
            self_att_module_kwargs=self_att_module_kwargs,
            **kwargs,
        )
        self.vis_att1 = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attn_dropout=attn_dropout,
            can_be_stateful=False,
            attention_module=enc_att_module,
            attention_module_kwargs=enc_att_module_kwargs,
        )
        self.vis_att2 = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attn_dropout=attn_dropout,
            can_be_stateful=False,
            attention_module=enc_att_module,
            attention_module_kwargs=enc_att_module_kwargs,
        )

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.activation = activation

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):
        self_att = self.self_att(x, x, x, mask_x)
        self_att = self_att * mask_pad

        enc_att1 = self.vis_att1(self_att, y1, y1, mask_y1) * mask_pad
        enc_att2 = self.vis_att2(self_att, y2, y2, mask_y2) * mask_pad
        alpha1 = self.fc_alpha1(torch.cat([self_att, enc_att1], -1))
        alpha2 = self.fc_alpha1(torch.cat([self_att, enc_att2], -1))

        if self.activation == 'sigmoid':
            alpha1 = torch.sigmoid(alpha1)
            alpha2 = torch.sigmoid(alpha2)

        if self.activation == 'softmax':
            alpha = rearrange([alpha1, alpha2], 'n1 b n d -> n1 b n d')
            alpha = torch.softmax(alpha, dim=0)
            alpha1 = alpha[0]
            alpha2 = alpha[1]

        if self.activation == 'identity':
            alpha1 = torch.ones_like(alpha1)
            alpha2 = torch.ones_like(alpha2)

        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2) / np.sqrt(2)
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class ConcatAttentionLayer(GeneratorLayer):

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=.1,
        attn_dropout=0.0,
        self_att_module=None,
        enc_att_module=None,
        self_att_module_kwargs=None,
        enc_att_module_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            self_att_module=self_att_module,
            self_att_module_kwargs=self_att_module_kwargs,
            **kwargs,
        )
        self.vis_att = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attn_dropout=attn_dropout,
            can_be_stateful=False,
            attention_module=enc_att_module,
            attention_module_kwargs=enc_att_module_kwargs,
        )

    def forward(self, x, y, mask_pad, mask_x, mask_y):
        out = self.self_att(x, x, x, mask_x) * mask_pad
        out = self.vis_att(out, y, y, mask_y) * mask_pad
        out = self.pwff(out) * mask_pad
        return out


class SequentialAttentionLayer(GeneratorLayer):

    def __init__(
        self,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=.1,
        attn_dropout=0.0,
        self_att_module=None,
        enc_att_module=None,
        self_att_module_kwargs=None,
        enc_att_module_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            attn_dropout=attn_dropout,
            self_att_module=self_att_module,
            self_att_module_kwargs=self_att_module_kwargs,
            **kwargs,
        )
        self.vis_att1 = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attn_dropout=attn_dropout,
            can_be_stateful=False,
            attention_module=enc_att_module,
            attention_module_kwargs=enc_att_module_kwargs,
        )

        self.vis_att2 = MultiHeadAttention(
            d_model,
            n_heads,
            dropout,
            attn_dropout=attn_dropout,
            can_be_stateful=False,
            attention_module=enc_att_module,
            attention_module_kwargs=enc_att_module_kwargs,
        )

        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2):
        out = self.self_att(x, x, x, mask_x) * mask_pad
        out = self.vis_att1(out, y1, y1, mask_y1) * mask_pad
        out = self.vis_att2(out, y2, y2, mask_y2) * mask_pad
        ff = self.pwff(out)
        ff = ff * mask_pad
        return ff


class CaptionGenerator(Module):
    GENERATOR_LAYER = {
        'concat': ConcatAttentionLayer,
        'parallel': ParallelAttentionLayer,
        'sequential': SequentialAttentionLayer,
    }

    def __init__(
        self,
        vocab_size,
        max_len,
        n_layers,
        pad_idx,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=.1,
        attn_dropout=0.0,
        self_att_module=None,
        enc_att_module=None,
        self_att_module_kwargs=None,
        enc_att_module_kwargs=None,
        decoder_name='parallel',
        cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)

        self.cfg = cfg
        self.decoder_name = decoder_name
        generator_layer = self.GENERATOR_LAYER[self.decoder_name]

        self.layers = ModuleList([
            generator_layer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                attn_dropout=attn_dropout,
                self_att_module=self_att_module,
                enc_att_module=enc_att_module,
                self_att_module_kwargs=self_att_module_kwargs,
                enc_att_module_kwargs=enc_att_module_kwargs,
                **kwargs,
            ) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.N = n_layers

        self.register_state('running_mask_x', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def get_seq_inputs(self, input):
        b_s, seq_len = input.shape[:2]
        mask_pad = (input != self.pad_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)
        mask_x = mask_x.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_x = mask_x + (input == self.pad_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_x = mask_x.gt(0)  # (b_s, 1, seq_len, seq_len)
        if self._is_stateful:
            self.running_mask_x = torch.cat([self.running_mask_x, mask_x], -1)
            mask_x = self.running_mask_x

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_pad.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
        x = self.word_emb(input) + self.pos_emb(seq)

        return {
            'mask_pad': mask_pad,
            'mask_x': mask_x,
            'seq': seq,
            'x': x,
        }

    def forward(self, input, vis_inputs,return_attention=False):
        seq_inputs = self.get_seq_inputs(input)
        mask_pad = seq_inputs['mask_pad']
        mask_x = seq_inputs['mask_x']
        x = seq_inputs['x']

        if self.decoder_name == 'concat':
            y, mask_y = [], []
            if 'gri' in self.cfg.vis_inputs:
                y.append(vis_inputs['gri_feat'])
                mask_y.append(vis_inputs['gri_mask'])
            if 'reg' in self.cfg.vis_inputs:
                y.append(vis_inputs['reg_feat'])
                mask_y.append(vis_inputs['reg_mask'])

            y = torch.cat(y, dim=1)
            mask_y = torch.cat(mask_y, dim=3)
            for i, l in enumerate(self.layers):
                x = l(x, y, mask_pad, mask_x, mask_y)

        if self.decoder_name == 'sequential':
            if self.cfg.sequential_order == 'gri_reg':
                y1 = vis_inputs['gri_feat']
                y2 = vis_inputs['reg_feat']
                mask_y1 = vis_inputs['gri_mask']
                mask_y2 = vis_inputs['reg_mask']
            else:
                y1 = vis_inputs['reg_feat']
                y2 = vis_inputs['gri_feat']
                mask_y1 = vis_inputs['reg_mask']
                mask_y2 = vis_inputs['gri_mask']

            for i, l in enumerate(self.layers):
                x = l(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)

        if self.decoder_name == 'parallel':
            y1 = vis_inputs['gri_feat']
            y2 = vis_inputs['reg_feat']
            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']
            for i, l in enumerate(self.layers):
                x = l(x, y1, y2, mask_pad, mask_x, mask_y1, mask_y2)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

class UnifyDecoder(Module):
    GENERATOR_LAYER = {
        'concat': ConcatAttentionLayer,
        'parallel': ParallelAttentionLayer,
        'sequential': SequentialAttentionLayer,
    }
    def __init__(
        self,
        vocab_size,
        max_len,
        n_layers,
        pad_idx,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        dropout=.1,
        attn_dropout=0.0,
        self_att_module=None,
        enc_att_module=None,
        self_att_module_kwargs=None,
        enc_att_module_kwargs=None,
        decoder_name='sequential',
        cfg=None,#下边的是levt所需的参数
        all_cfg=None,
        text_field=None,
        **kwargs,
    ):
        super().__init__()
        print(f"d_model={d_model}")
        self.d_model = d_model
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.unify_pos_emb=nn.Embedding(max_len+1, d_model)
        self.reposition_word_embedding=None
        if all_cfg.model.reposition_pos_embedding == 'learnable':
            self.reposition_pos_embedding=nn.Embedding(max_len + 1, d_model)
            self.reposition_word_embedding= nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        elif all_cfg.model.reposition_pos_embedding == 'fixed':
            self.reposition_pos_embedding=nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        else:
            self.reposition_pos_embedding=None
        self.cfg = cfg
        self.decoder_name = decoder_name
        no_share_discriminator=all_cfg.model.no_share_discriminator
        no_share_maskinserter=all_cfg.model.no_share_maskinserter
        no_share_unifier=all_cfg.model.no_share_unifier
        generator_layer = self.GENERATOR_LAYER[self.decoder_name]
        self.layers = ModuleList([
            generator_layer(
                d_model,
                n_heads,
                d_ff,
                dropout,
                attn_dropout=attn_dropout,
                self_att_module=self_att_module,
                enc_att_module=enc_att_module,
                self_att_module_kwargs=self_att_module_kwargs,
                enc_att_module_kwargs=enc_att_module_kwargs,
                **kwargs,
            ) for _ in range(n_layers)
        ])
        self.unifyer=None
        if no_share_unifier:
            self.unifyer=ModuleList([
                generator_layer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    attn_dropout=attn_dropout,
                    self_att_module=self_att_module,
                    enc_att_module=enc_att_module,
                    self_att_module_kwargs=self_att_module_kwargs,
                    enc_att_module_kwargs=enc_att_module_kwargs,
                    **kwargs,
                ) for _ in range(n_layers)
            ])


        self.repositioner=None
        

        self.mask_inserter=None
        self.mask_inserter_word_embedding=None

        self.fc = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.N = n_layers
        self.fc_unify=nn.Linear(d_model, self.max_len, bias=False)#unify的输出
        self.fc_mask=nn.Linear(d_model, vocab_size, bias=False)#mask的输出
        self.fc_unify_fusion=nn.Linear(d_model*2, d_model, bias=False)#unifyconcat两个
        self.text_field=text_field
    def get_seq_inputs(self, input,word_embedding=None):
        b_s, seq_len = input.shape[:2]
        mask_pad = (input != self.pad_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_x = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1)


        
        mask_x_test=mask_pad.squeeze(-1).unsqueeze(1).repeat(1,mask_pad.shape[1],1).unsqueeze(1).gt(0)
        if self._is_stateful:
            self.running_mask_x = torch.cat([self.running_mask_x, mask_x], -1)
            mask_x = self.running_mask_x

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        pos_emb_len=self.max_len+1
        full_seq=torch.arange(1, pos_emb_len).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        x_full_seq=self.unify_pos_emb(full_seq)+self.pos_emb(full_seq)

        seq = seq.masked_fill(mask_pad.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq
        if word_embedding==None:
            word_embedding=self.word_emb
        x = word_embedding(input) + self.pos_emb(seq)

        return {
            'mask_pad': mask_pad,
            'mask_x': (~mask_x_test).to(torch.uint8),
            'seq': seq,
            'x': x,
            'x_full_seq':x_full_seq
        }

    def forward_feature(
        self,
        input,
        vis_inputs,
        layers=None,
        word_embedding=None,
        is_unify=False
    ):
        seq_inputs = self.get_seq_inputs(input,word_embedding)
        x=seq_inputs['x']
        mask_pad=seq_inputs['mask_pad']
        mask_x=seq_inputs['mask_x']

        if self.decoder_name == 'sequential':
            if self.cfg.sequential_order == 'gri_reg':
                y1 = vis_inputs['gri_feat']
                y2 = vis_inputs['reg_feat']
                mask_y1 = vis_inputs['gri_mask']
                mask_y2 = vis_inputs['reg_mask']
            else:
                y1 = vis_inputs['reg_feat']
                y2 = vis_inputs['gri_feat'] 
                mask_y1 = vis_inputs['reg_mask']
                mask_y2 = vis_inputs['gri_mask']
            if layers is None:
                for i, l in enumerate(self.layers):
                    x = l(x, y1, y2, mask_pad, seq_inputs['mask_x'], mask_y1, mask_y2)
            else:
                for i, l in enumerate(layers):
                    x = l(x, y1, y2, mask_pad, seq_inputs['mask_x'], mask_y1, mask_y2)
        elif self.decoder_name == 'parallel':
            if is_unify:
                y1 = vis_inputs['gri_feat_unify']
                y2 = vis_inputs['reg_feat_unify']
            else:
                y1 = vis_inputs['gri_feat']
                y2 = vis_inputs['reg_feat']
            mask_y1 = vis_inputs['gri_mask']
            mask_y2 = vis_inputs['reg_mask']
            if layers is None:
                for i, l in enumerate(self.layers):
                    x = l(x, y1, y2, mask_pad, seq_inputs['mask_x'], mask_y1, mask_y2)
            else:
                for i, l in enumerate(layers):
                    x = l(x, y1, y2, mask_pad, seq_inputs['mask_x'], mask_y1, mask_y2)
        elif self.decoder_name == 'concat':
            y2 = vis_inputs['reg_feat'] #(?,?,2048)
            mask_y2 = vis_inputs['reg_mask']#(?,?,1)
            y2=F.relu(self.butd_proj(y2))
            y2=self.butd_dropout(y2)
            y2=self.butd_layernorm(y2)
            if layers is None:
                for i, l in enumerate(self.layers):
                    x = l(x, y2, mask_pad, seq_inputs['mask_x'], mask_y2)
            else:
                for i, l in enumerate(layers):
                    x = l(x, y2, mask_pad, seq_inputs['mask_x'], mask_y2)
        else:
            raise NotImplementedError
        return x
    def forward_unify(
        self,
        input,
        vis_inputs,
        normalize=False):
        assert self.repositioner==None
        features=self.forward_feature(input,vis_inputs,self.unifyer,self.reposition_word_embedding,is_unify=False)
        seq_inputs = self.get_seq_inputs(input,self.reposition_word_embedding)
        x=seq_inputs['x']#[b,seq_len,emb]
        seq_len=x.shape[1]
        x_full_seq=seq_inputs['x_full_seq']
        features=self.fc_unify_fusion(torch.cat([features,x],dim=-1))
        decoder_out = torch.bmm(features, x_full_seq.transpose(1, 2))#等于是 seqxemb * emb*max_len->seq*max_len
        if normalize:
            return F.log_softmax(decoder_out, dim=-1)
        else:
            return decoder_out
        
    def forward_mask_pred(
        self,
        input,
        vis_inputs,
        normalize=False):
        features=self.forward_feature(input,vis_inputs,self.layers)
        if normalize:
            return F.log_softmax(self.fc_mask(features), dim=-1)
        else:
            return self.fc_mask(features)
