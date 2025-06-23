# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch

from natlib.utils import new_arange


def load_libnat():
    try:
        from natlib import libnat_cuda

        return libnat_cuda, True

    except ImportError as e:
        print(str(e) + "... fall back to CPU version")

        try:
            from natlib import libnat

            return libnat, False

        except ImportError as e:
            import sys

            sys.stderr.write(
                "ERROR: missing libnat_cuda. run `python setup.py build_ext --inplace`\n"
            )
            raise e
def _get_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx):
    libnat, use_cuda = load_libnat()

    def _get_ins_targets_cuda(in_tokens, out_tokens, padding_idx, unk_idx):
        in_masks = in_tokens.ne(padding_idx)
        out_masks = out_tokens.ne(padding_idx)
        mask_ins_targets, masked_tgt_masks = libnat.generate_insertion_labels(
            out_tokens.int(),
            libnat.levenshtein_distance(
                in_tokens.int(),
                out_tokens.int(),
                in_masks.sum(1).int(),
                out_masks.sum(1).int(),
            ),
        )
        masked_tgt_masks = masked_tgt_masks.bool() & out_masks
        mask_ins_targets = mask_ins_targets.type_as(in_tokens)[
            :, 1 : in_masks.size(1)
        ].masked_fill_(~in_masks[:, 1:], 0)
        masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
        return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets

    def _get_ins_targets_cpu(in_tokens, out_tokens, padding_idx, unk_idx):
        in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

        full_labels = libnat.suggested_ed2_path(
            in_tokens_list, out_tokens_list, padding_idx
        )
        mask_inputs = [
            [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
        ]
        masked_tgt_masks = []
        for mask_input in mask_inputs:
            mask_label = []
            for beam_size in mask_input[1:-1]:  # HACK 1:-1
                mask_label += [0] + [1 for _ in range(beam_size)]
            masked_tgt_masks.append(
                mask_label + [0 for _ in range(out_seq_len - len(mask_label))]
            )
        mask_ins_targets = [
            mask_input[1:-1]
            + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
            for mask_input in mask_inputs
        ]
        masked_tgt_masks = torch.tensor(
            masked_tgt_masks, device=out_tokens.device
        ).bool()
        mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
        masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
        return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets

    if use_cuda:
        return _get_ins_targets_cuda(in_tokens, out_tokens, padding_idx, unk_idx)
    return _get_ins_targets_cpu(in_tokens, out_tokens, padding_idx, unk_idx)


def _get_del_targets(in_tokens, out_tokens, padding_idx):
    libnat, use_cuda = load_libnat()

    def _get_del_targets_cuda(in_tokens, out_tokens, padding_idx):
        in_masks = in_tokens.ne(padding_idx)
        out_masks = out_tokens.ne(padding_idx)

        word_del_targets = libnat.generate_deletion_labels(
            in_tokens.int(),
            libnat.levenshtein_distance(
                in_tokens.int(),
                out_tokens.int(),
                in_masks.sum(1).int(),
                out_masks.sum(1).int(),
            ),
        )
        word_del_targets = word_del_targets.type_as(in_tokens).masked_fill_(
            ~in_masks, 0
        )
        return word_del_targets

    def _get_del_targets_cpu(in_tokens, out_tokens, padding_idx):
        out_seq_len = out_tokens.size(1)
        with torch.cuda.device_of(in_tokens):
            in_tokens_list = [
                [t for t in s if t != padding_idx]
                for i, s in enumerate(in_tokens.tolist())
            ]
            out_tokens_list = [
                [t for t in s if t != padding_idx]
                for i, s in enumerate(out_tokens.tolist())
            ]

        full_labels = libnat.suggested_ed2_path(
            in_tokens_list, out_tokens_list, padding_idx
        )
        word_del_targets = [b[-1] for b in full_labels]
        word_del_targets = [
            labels + [0 for _ in range(out_seq_len - len(labels))]
            for labels in word_del_targets
        ]
        word_del_targets = torch.tensor(word_del_targets, device=out_tokens.device)
        return word_del_targets

    if use_cuda:
        return _get_del_targets_cuda(in_tokens, out_tokens, padding_idx)
    return _get_del_targets_cpu(in_tokens, out_tokens, padding_idx)


def _get_advanced_ins_targets(in_tokens, out_tokens, padding_idx, unk_idx):
    libnat, use_cuda = load_libnat()

    def _get_ins_targets_cuda(in_tokens, out_tokens, padding_idx, unk_idx):
        in_masks = in_tokens.ne(padding_idx)
        out_masks = out_tokens.ne(padding_idx)
        mask_ins_targets, masked_tgt_masks = libnat.generate_insertion_labels(
            out_tokens.int().contiguous(),
            libnat.advanced_levenshtein_distance(
                in_tokens.int().contiguous(), out_tokens.int().contiguous(),
                in_masks.sum(1).int(), out_masks.sum(1).int()
            )
        )
        masked_tgt_masks = masked_tgt_masks.bool() & out_masks
        mask_ins_targets = mask_ins_targets.type_as(
            in_tokens)[:, 1:in_masks.size(1)].masked_fill_(~in_masks[:, 1:], 0)
        masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
        return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets

    def _get_ins_targets_cpu(in_tokens, out_tokens, padding_idx, unk_idx):
        in_seq_len, out_seq_len = in_tokens.size(1), out_tokens.size(1)

        in_tokens_list = [
            [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
        ]
        out_tokens_list = [
            [t for t in s if t != padding_idx]
            for i, s in enumerate(out_tokens.tolist())
        ]

        full_labels = libnat.suggested_ed3_path(
            in_tokens_list, out_tokens_list, padding_idx
        )
        mask_inputs = [
            [len(c) if c[0] != padding_idx else 0 for c in a[:-1]] for a in full_labels
        ]
        word_del_targets = [b[-1] for b in full_labels]
        masked_tgt_masks = []
        for mask_input, del_labels in zip(mask_inputs, word_del_targets):
            mask_label = []
            for beam_size, del_label in zip(mask_input[1:-1], del_labels):  # HACK 1:-1
                if del_label > 0:
                    mask_label += [0]
                mask_label += [1 for _ in range(beam_size)]
            masked_tgt_masks.append(
                mask_label + [0 for _ in range(out_seq_len - len(mask_label))]
            )
        mask_ins_targets = [
            mask_input[1:-1] + [0 for _ in range(in_seq_len - 1 - len(mask_input[1:-1]))]
            for mask_input in mask_inputs
        ]
        masked_tgt_masks = torch.tensor(
            masked_tgt_masks, device=out_tokens.device
        ).bool()
        mask_ins_targets = torch.tensor(mask_ins_targets, device=in_tokens.device)
        masked_tgt_tokens = out_tokens.masked_fill(masked_tgt_masks, unk_idx)
        return masked_tgt_masks, masked_tgt_tokens, mask_ins_targets

    if use_cuda:
        return _get_ins_targets_cuda(in_tokens, out_tokens, padding_idx, unk_idx)
    return _get_ins_targets_cpu(in_tokens, out_tokens, padding_idx, unk_idx)


def _get_advanced_reposition_targets(in_tokens, out_tokens, padding_idx):
    libnat, use_cuda = load_libnat()

    def _get_reposition_targets_cuda(in_tokens, out_tokens, padding_idx):
        in_masks = in_tokens.ne(padding_idx)
        out_masks = out_tokens.ne(padding_idx)

        word_reposition_targets = libnat.generate_reposition_labels(
            in_tokens.int().contiguous(),
            libnat.advanced_levenshtein_distance(
                in_tokens.int().contiguous(), out_tokens.int().contiguous(),
                in_masks.sum(1).int(), out_masks.sum(1).int()
            )
        )
        word_reposition_targets = word_reposition_targets.type_as(in_tokens).masked_fill_(~in_masks, 0)
        return word_reposition_targets

    def _get_reposition_targets_cpu(in_tokens, out_tokens, padding_idx):
        out_seq_len = in_tokens.size(1)
        with torch.cuda.device_of(in_tokens):
            in_tokens_list = [
                [t for t in s if t != padding_idx] for i, s in enumerate(in_tokens.tolist())
            ]
            out_tokens_list = [
                [t for t in s if t != padding_idx]
                for i, s in enumerate(out_tokens.tolist())
            ]

        full_labels = libnat.suggested_ed3_path(
            in_tokens_list, out_tokens_list, padding_idx
        )
        word_reposition_targets = [b[-1] for b in full_labels]
        word_reposition_targets = [
            labels + [0 for _ in range(out_seq_len - len(labels))]
            for labels in word_reposition_targets
        ]
        word_reposition_targets = torch.tensor(word_reposition_targets, device=in_tokens.device)
        return word_reposition_targets

    if use_cuda:
        return _get_reposition_targets_cuda(in_tokens, out_tokens, padding_idx)
    return _get_reposition_targets_cpu(in_tokens, out_tokens, padding_idx)
def _apply_unify(
    in_tokens, in_scores, unify_pred,unify_score, pad_idx, bos_idx, eos_idx ,unk_idx):
    unify_pred[:,0]=0
    _unify_pred=unify_pred.clone()
    pred_mask=in_tokens.ne(pad_idx)
    _unify_pred[~pred_mask]=-1
    eos_pos=in_tokens.eq(eos_idx).nonzero()[:,1]
    assert in_tokens.eq(eos_idx).sum()==in_tokens.size(0)
    eos_unify_pred=_unify_pred.gather(1,eos_pos.unsqueeze(-1)).squeeze()
    max_unify_pred=_unify_pred.max(-1)[0]
    
    eos_error=(eos_unify_pred<max_unify_pred)|(eos_unify_pred==0)

    error_rows=eos_error.nonzero().squeeze(1)


    error_eos_pos=eos_pos[eos_error]
    _unify_pred[error_rows,error_eos_pos]=max_unify_pred[error_rows]+1


    max_size=100
    true_max_size=50
    out_tokens = in_tokens.new_zeros(in_tokens.size(0),max_size).fill_(unk_idx)
    out_scores = in_scores.new_zeros(in_tokens.size(0),max_size).fill_(0)
    _unify_pred[_unify_pred.eq(0)]=1000
    _unify_pred[~pred_mask]=1000
    _unify_pred[:,0]=0
    unify_pred_sort, unify_pred_sort_idx = _unify_pred.sort(-1)
    difference_matrix = unify_pred_sort[:, 1:] - unify_pred_sort[:, :-1]
    difference_matrix = (difference_matrix == 0).long()
    difference_matrix = torch.cumsum(difference_matrix, dim=-1)
    unify_pred_sort[:,1:]+=difference_matrix

    unify_pred_sort[unify_pred_sort>=1000]=0
    out_tokens.scatter_(1,unify_pred_sort,in_tokens.gather(1,unify_pred_sort_idx))
    out_scores.scatter_(1,unify_pred_sort,in_scores.gather(1,unify_pred_sort_idx))
    out_tokens[:,0]=bos_idx
    out_attn = None
    eos_pos_after=out_tokens.eq(eos_idx).nonzero()[:,1]
    max_eos_pos=eos_pos_after.max()

    out_tokens=out_tokens[:,:max_eos_pos+1]
    out_scores=out_scores[:,:max_eos_pos+1]
    invaild_mask=(eos_pos_after>=true_max_size)
    if invaild_mask.any():
        out_tokens[invaild_mask]=out_tokens[invaild_mask].index_fill_(1, torch.tensor(true_max_size-1).to(out_tokens.device), eos_idx)
        out_scores[invaild_mask]=out_scores[invaild_mask].index_fill_(1, torch.tensor(true_max_size-1).to(out_tokens.device), 0)
        out_tokens=out_tokens[:,:true_max_size]
        out_scores=out_scores[:,:true_max_size]

    eos_pos=out_tokens.eq(eos_idx).nonzero()[:,1]
    for i in range(eos_pos.size(0)):
        out_tokens[i,eos_pos[i]+1:]=pad_idx
        out_scores[i,eos_pos[i]+1:]=0
        
    
    return out_tokens, out_scores, out_attn
def _apply_ins_masks(
    in_tokens, in_scores, mask_ins_pred, padding_idx, unk_idx, eos_idx
):

    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(1)
    in_tokens.masked_fill_(~in_masks, eos_idx)
    mask_ins_pred.masked_fill_(~in_masks[:, 1:], 0)

    out_lengths = in_lengths + mask_ins_pred.sum(1)
    out_max_len = out_lengths.max()
    out_masks = new_arange(out_lengths, out_max_len)[None, :] < out_lengths[:, None]

    reordering = (mask_ins_pred + in_masks[:, 1:].long()).cumsum(1)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), out_max_len)
        .fill_(padding_idx)
        .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, 0] = in_tokens[:, 0]
    out_tokens.scatter_(1, reordering, in_tokens[:, 1:])

    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, 0] = in_scores[:, 0]
        out_scores.scatter_(1, reordering, in_scores[:, 1:])

    return out_tokens, out_scores


def _apply_ins_words(in_tokens, in_scores, word_ins_pred, word_ins_scores, unk_idx):
    word_ins_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(word_ins_masks, word_ins_pred[word_ins_masks])

    if in_scores is not None:
        out_scores = in_scores.masked_scatter(
            word_ins_masks, word_ins_scores[word_ins_masks]
        )
    else:
        out_scores = None

    return out_tokens, out_scores


def _apply_del_words(
    in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx
):
    in_masks = in_tokens.ne(padding_idx)
    bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)

    max_len = in_tokens.size(1)
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)

    reordering = new_arange(in_tokens).masked_fill_(word_del_pred, max_len).sort(1)[1]

    out_tokens = in_tokens.masked_fill(word_del_pred, padding_idx).gather(1, reordering)

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0).gather(1, reordering)

    out_attn = None
    if in_attn is not None:
        _mask = word_del_pred[:, :, None].expand_as(in_attn)
        _reordering = reordering[:, :, None].expand_as(in_attn)
        out_attn = in_attn.masked_fill(_mask, 0.0).gather(1, _reordering)

    return out_tokens, out_scores, out_attn
def _apply_reposition_words(
    in_tokens, in_scores, in_attn, word_reposition_pred, padding_idx, bos_idx, eos_idx
):
    in_masks = in_tokens.ne(padding_idx)
    word_reposition_pred.masked_fill_(~in_masks, 0)
    word_reposition_pred = word_reposition_pred.clamp(min=0, max=in_tokens.size(1) - 1)
    out_tokens = (in_tokens
        .gather(1, word_reposition_pred)
        .masked_fill(word_reposition_pred.eq(0), padding_idx)
        .masked_fill(in_tokens.eq(bos_idx), bos_idx)
        .masked_fill(in_tokens.eq(eos_idx), eos_idx)
    )

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.gather(1, word_reposition_pred)

    out_attn = None
    if in_attn is not None:
        _reposition = word_reposition_pred[:, :, None].expand_as(in_attn)
        out_attn = in_attn.gather(1, _reposition)
    out_tokens, out_scores, out_attn = _apply_del_words(
        out_tokens,
        out_scores,
        out_attn,
        out_tokens.eq(padding_idx),
        padding_idx,
        bos_idx,
        eos_idx
    )
    return out_tokens, out_scores, out_attn

def _apply_edit_words(
    in_tokens, in_scores, in_attn, unify_pred, padding_idx, bos_idx, eos_idx
):
    max_size=54
    out_tokens = in_tokens.new_zeros(in_tokens.size(0),max_size).fill_(padding_idx)

    unify_pred[unify_pred.eq(0)]=-1
    unify_pred_sort, unify_pred_sort_idx = inunify_pred_tokens.sort(-1)

    difference_matrix = unify_pred_sort[:, 1:] - unify_pred_sort[:, :-1]
    difference_matrix = torch.cumsum(difference_matrix, dim=-1)

    unify_pred_sort[:,1:]+=difference_matrix
    
    out_tokens[unify_pred_sort]=in_tokens[unify_pred_sort_idx]

    out_scores = None
    out_atten = None
    return out_tokens, out_scores, out_attn




def _skip(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [_skip(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: _skip(v, mask) for k, v in x.items()}

    raise NotImplementedError


def _skip_encoder_out(encoder, encoder_out, mask):
    if not mask.any():
        return encoder_out
    else:
        return encoder.reorder_encoder_out(
            encoder_out, mask.nonzero(as_tuple=False).squeeze()
        )


def _fill(x, mask, y, padding_idx):
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None:
        return y
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))
    n_selected = mask.sum()
    assert n_selected == y.size(0)

    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        dims = [x.size(0), y.size(1) - x.size(1)]
        if x.dim() == 3:
            dims.append(x.size(2))
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 1)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = padding_idx
        if x.dim() == 2:
            x[mask, : y.size(1)] = y
        else:
            x[mask, : y.size(1), :] = y
    else:
        x[mask] = y
    return x
