# ------------------------------------------------------------------------
# Modified from Meshed Memory Transformer
# https://github.com/aimagelab/meshed-memory-transformer
# ------------------------------------------------------------------------
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO as pyCOCO
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler
from tqdm import tqdm

from engine.utils import nested_tensor_from_tensor_list

from .example import Example
from .field import ImageField, TextField
from .transforms import *
from .transforms import get_transform
from .transforms.utils import MinMaxResize

OVERFIT_SIZE = 64


class DictionaryCollator:

    def __init__(self, img_field, device='cpu'):
        self.img_field = img_field
        self.device = device

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        image_ids = [item[2] for item in batch]
        outputs = {}

        if len(batch[0])>3:
            refs_mask= [ not (len(item[3])==0) for item in batch]
            refs = [item[3] for item in batch]
            outputs['refs'] = refs
            outputs['refs_mask'] = refs_mask
        if len(batch[0])>4:
            gt_tokenized_captions = [item[4] for item in batch]
            outputs['gt_tokenized_captions'] = gt_tokenized_captions
        if len(batch[0])>5:
            distillation = [item[5] for item in batch]
            outputs['distillation'] = distillation
        if self.img_field.use_hdf5_feat:
            samples = {}
            if self.img_field.use_gri_feat:
                samples['gri_feat'] = torch.stack([im['gri_feat'] for im in imgs]).to(self.device)
                samples['gri_mask'] = torch.stack([im['gri_mask'] for im in imgs]).to(self.device)
            if self.img_field.use_reg_feat:
                samples['reg_feat'] = torch.stack([im['reg_feat'] for im in imgs]).to(self.device)
                samples['reg_mask'] = torch.stack([im['reg_mask'] for im in imgs]).to(self.device)
            outputs['samples'] = samples
        else:
            outputs['samples'] = nested_tensor_from_tensor_list(imgs).to(self.device)

        outputs['captions'] = captions
        outputs['image_id'] = image_ids

        return outputs


class PairedCollator(DictionaryCollator):

    def __init__(self, img_field, device='cpu', max_len=54, pad_idx=1, bos_idx=2, eos_idx=3):
        super().__init__(img_field, device)
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __call__(self, batch):
        b = super().__call__(batch)

        # truncate
        captions = [c[:self.max_len] for c in b['captions']]
        max_len = max([len(c) for c in b['captions']])


        padded = []
        for c in captions:
            caption = [self.bos_idx] + c + [self.eos_idx] + [self.pad_idx] * (max_len - len(c))
            padded.append(caption)

        padded = [torch.Tensor(caption).long() for caption in padded]
        padded = pad_sequence(padded, batch_first=True).to(self.device)
        
        b['captions'] = padded

        if 'refs' in b:
            # ref:same as captions
            refs = [c[:self.max_len] for c in b['refs']]
            max_len = max([len(c) for c in b['refs']])

            padded = []
            for c in refs:
                ref = [self.bos_idx] + c + [self.eos_idx] + [self.pad_idx] * (max_len - len(c))
                padded.append(ref)

            padded = [torch.Tensor(ref).long() for ref in padded]
            padded = pad_sequence(padded, batch_first=True).to(self.device)

            b['refs'] = padded
        #处理distillation
        if 'distillation' in b:
            distillation = [c[:self.max_len] for c in b['distillation']]
            max_len = max([len(c) for c in b['distillation']])
            padded = []
            for c in distillation:
                distillation = [self.bos_idx] + c + [self.eos_idx] + [self.pad_idx] * (max_len - len(c))
                padded.append(distillation)
            
            padded = [torch.Tensor(distillation).long() for distillation in padded]
            padded = pad_sequence(padded, batch_first=True).to(self.device)
            b['distillation'] = padded
        return b

import time


class CPairedDataset:

    def __init__(self, examples, image_field, overfit=False):
        self.examples = examples
        self.image_field = image_field
        self.overfit = overfit
        self.img2captions={}
        self.img2image_id={}
        for example in examples:
            if example.image not in self.img2captions:
                self.img2captions[example.image] = []
            self.img2captions[example.image].append(example.tokenized_text)
            self.img2image_id[example.image] = example.image_id
    def __getitem__(self, idx):
        # start_time=time.time()
        example = self.examples[idx]
        img_path, caption = example.image, example.tokens
        
        #random select a ref from ref_tokens_list
        if len(example.ref_tokens_list)>0:
            ref = random.choice(example.ref_tokens_list)
        else:
            ref=[]
        distillation=example.distillation_tokens
        # ref=getattr(example,'ref_tokens',"")
        image_id = example.image_id
        # id_end_time=time.time()
        # print(f"get id time: {id_end_time-start_time}")
        img = self.image_field.preprocess(img_path)
        # load_feature_time=time.time()
        # print(f"load feature time: {load_feature_time-id_end_time}")
        
        return img, caption, image_id,ref,self.img2captions[example.image],distillation

    def __len__(self):
        if self.overfit:
            return OVERFIT_SIZE
        return len(self.examples)


class TestCollator:

    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch):
        imgs = [item[0] for item in batch]
        image_ids = [item[1] for item in batch]

        outputs = {}
        outputs['samples'] = nested_tensor_from_tensor_list(imgs).to(self.device)
        outputs['image_id'] = image_ids
        return outputs


class TestDataset:

    def __init__(self, root, anno_file, transform=None, overfit=False, from_idx=0, to_idx=-1):
        annotations = json.load(open(anno_file))['images']
        total_images = len(annotations)
        print("Total images:", total_images)
        print(f"from idx: {from_idx} to idx: {to_idx}")

        if to_idx >= total_images - 1 or to_idx == -1:
            self.annotations = annotations[from_idx:]
        else:
            self.annotations = annotations[from_idx:to_idx]

        self.root = root
        self.transform = transform
        if self.transform is None:
            self.transform = Compose([MinMaxResize((384, 640)), ToTensor(), normalize()])

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.root, item['file_name'])
        img_id = item['id']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, img_id

    def __len__(self):
        return len(self.annotations)


class CDictionaryDataset:

    def __init__(self, examples, image_field, overfit=False):
        self.overfit = overfit
        self.image_field = image_field
        self.img2captions = {}
        self.img2image_id = {}
        self.caption_id2ref={}
        self.img2refs={}
        self.img2distillation={}
        for example in examples:
            if example.image not in self.img2captions:
                self.img2captions[example.image] = []
                self.img2refs[example.image]=example.ref_tokens_list
                self.img2distillation[example.image]=example.distillation_tokens
            self.img2captions[example.image].append(example.text)
            self.img2image_id[example.image] = example.image_id
            
        self.img_paths = list(self.img2captions.keys())

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image_id = self.img2image_id[img_path]
        captions = self.img2captions[img_path]
        ref= random.choice(self.img2refs[img_path])
        img = self.image_field.preprocess(img_path)
        # distillation=self.img2distillation[img_path]
        return img, captions, image_id,ref

    def __len__(self):
        if self.overfit:
            return OVERFIT_SIZE
        return len(self.img_paths)


class COCO(CPairedDataset):

    def __init__(
        self,
        image_field,
        text_field,
        img_root,
        ann_root,
        ref_path,
        use_restval=True,
        cut_validation=False,
        overfit=False,
        **kwargs,
    ):
        self.image_field = image_field
        self.text_field = text_field
        self.overfit = overfit
        with open(ref_path, 'r') as f:
            import random
            self.ref = json.load(f)
            #convert key to int
            self.ref = {int(k): v for k, v in self.ref.items()}
            for key,value in tqdm(self.ref.items()):
                value['neg_tokens']=[[self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(ref)] for ref in value['neg']]
                if 'distillation' in value:
                    assert 'distillation_blip2_opt_2.7' in value
                    if random.random()<1:
                        value['distillation_tokens']=[self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(value['distillation'])] 
                        value['distillation']=value['distillation']
                        value['type']='original'
                    else:
                        value['distillation_tokens']=[self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(value['distillation_blip2_opt_2.7'])]
                        value['distillation']=value['distillation_blip2_opt_2.7']
                        value['type']='blip2'
        roots = {}
        roots['train'] = {
            'img': os.path.join(img_root, 'train2014'),
            'cap': os.path.join(ann_root, 'captions_train2014.json')
        }
        roots['valid'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['test'] = {
            'img': os.path.join(img_root, 'val2014'),
            'cap': os.path.join(ann_root, 'captions_val2014.json')
        }
        roots['trainrestval'] = {
            'img': (roots['train']['img'], roots['valid']['img']),
            'cap': (roots['train']['cap'], roots['valid']['cap'])
        }

        if ann_root is not None:
            ids = {}
            ids['train'] = np.load(os.path.join(ann_root, 'coco_train_ids.npy'))
            ids['valid'] = np.load(os.path.join(ann_root, 'coco_dev_ids.npy'))
            if cut_validation:
                ids['valid'] = ids['valid'][:5000]
            ids['test'] = np.load(os.path.join(ann_root, 'coco_test_ids.npy'))
            ids['trainrestval'] = (ids['train'], np.load(os.path.join(ann_root, 'coco_restval_ids.npy')))

            if use_restval:
                roots['train'] = roots['trainrestval']
                ids['train'] = ids['trainrestval']
        else:
            ids = None

        self.train_examples, self.valid_examples, self.test_examples = self.get_samples(roots, ids)

    def split_examples(self):
        return {
            'train': self.train_examples,
            'valid': self.valid_examples,
            'test': self.test_examples,
        }

    def splits(self):
        train_split = CPairedDataset(self.train_examples, self.image_field, self.text_field, overfit=self.overfit)
        valid_split = CPairedDataset(self.valid_examples, self.image_field, self.text_field, overfit=self.overfit)
        test_split = CPairedDataset(self.test_examples, self.image_field, self.text_field, overfit=self.overfit)
        return train_split, valid_split, test_split

    def get_samples(self, roots, ids_dataset=None):
        train_samples = []
        valid_samples = []
        test_samples = []
        heights = []
        widths = []

        splits = ['valid', 'test'] if self.overfit else ['train', 'valid', 'test']
        
        for split in splits:
            if isinstance(roots[split]['cap'], tuple):
                coco_dataset = (pyCOCO(roots[split]['cap'][0]), pyCOCO(roots[split]['cap'][1]))
                root = roots[split]['img']
            else:
                coco_dataset = (pyCOCO(roots[split]['cap']),)
                root = (roots[split]['img'],)

            if ids_dataset is None:
                ids = list(coco_dataset.anns.keys())
            else:
                ids = ids_dataset[split]

            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])
            else:
                bp = len(ids)
            counter=0
            for index in tqdm(range(len(ids))):
                if index < bp:
                    coco = coco_dataset[0]
                    img_root = root[0]
                else:
                    coco = coco_dataset[1]
                    img_root = root[1]

                ann_id = ids[index]
                caption = coco.anns[ann_id]['caption']
                caption_id=coco.anns[ann_id]['id']
                img_id = coco.anns[ann_id]['image_id']
                filename = coco.loadImgs(img_id)[0]['file_name']
                height = coco.loadImgs(img_id)[0]['height']
                width = coco.loadImgs(img_id)[0]['width']
                heights.append(height)
                widths.append(width)

                # if caption_id in self.ref:
                #     ref = self.ref[caption_id]['Ref-Cap']
                # else:
                #     ref=""
                # if split=='train':
                ref_tokens_list=self.ref[img_id]['neg_tokens']
                ref_texts=self.ref[img_id]['neg']
                # else:
                #     ref_tokens_list=[]
                #     ref_texts=""
                if "distillation_tokens" in self.ref[img_id]:
                    distillation=self.ref[img_id]['distillation']
                    distillation_tokens=self.ref[img_id]['distillation_tokens']
                else:
                    distillation=caption
                    distillation_tokens=[self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(caption)]
                    counter+=1

                example = Example.fromdict({
                    'image_id': img_id,
                    'image': os.path.join(img_root, filename),
                    'ref_tokens_list':ref_tokens_list,                   
                    'ref_texts': ref_texts,
                    'caption_id': caption_id,
                    'text': caption,
                    'tokens': [self.text_field.vocab.stoi[w] for w in self.text_field.preprocess(caption)],
                    'tokenized_text': " ".join(self.text_field.preprocess(caption)),
                    'distillation': distillation,
                    'distillation_tokens': distillation_tokens,
                })

                if split == 'train':
                    train_samples.append(example)
                elif split == 'valid':
                    valid_samples.append(example)
                elif split == 'test':
                    test_samples.append(example)
            print(counter)
        if self.overfit:
            train_samples = valid_samples
        return train_samples, valid_samples, test_samples


def build_eval_dataloaders(config=None, mode='freezing', device='cpu'):
    overfit = config.dataset.overfit
    transform = get_transform(config.dataset.transform_cfg)

    text_field = TextField(vocab_path=config.dataset.vocab_path)
    valid_field = ImageField(transform=transform['valid'], **config.dataset)
    train_field = ImageField(transform=transform['train'], **config.dataset)
    examples = COCO(None, text_field, **config.dataset).split_examples()

    if mode == 'freezing' and config.optimizer.freezing_xe_epochs > 0:
        valid_field.init_hdf5_feat()
        train_field.init_hdf5_feat()

    if mode == 'finetune':
        valid_field.use_hdf5_feat = False
        train_field.use_hdf5_feat = False

    datasets = {
        'valid_dict': CDictionaryDataset(examples['valid'], valid_field, overfit=overfit),
        'test_dict': CDictionaryDataset(examples['test'], valid_field, overfit=overfit),
    }

    collators = {
        'valid_dict': DictionaryCollator(valid_field, device=device),
        'test_dict': DictionaryCollator(valid_field, device=device),
    }

    batch_size = config.optimizer.batch_size * 4 if mode == 'freezing' else config.optimizer.batch_size
    sc_batch_size = config.optimizer.batch_size if mode == 'freezing' else config.optimizer.batch_size // 4

    dataloaders = {}
    # test and valid
    dataloaders['valid_dict'] = DataLoader(
        datasets['valid_dict'],
        batch_size=max(1, sc_batch_size * 2),
        num_workers=config.optimizer.num_workers,
        collate_fn=collators['valid_dict'],
    )
    dataloaders['test_dict'] = DataLoader(
        datasets['test_dict'],
        batch_size=max(1, sc_batch_size * 2),
        num_workers=config.optimizer.num_workers,
        collate_fn=collators['test_dict'],
    )

    if getattr(config.exp, 'eval', False):
        return dataloaders

    # samplers = {
    #     'train': DistributedSampler(datasets['train'], shuffle=True),
    #     'valid': DistributedSampler(datasets['valid'], shuffle=False),
    #     'train_dict': DistributedSampler(datasets['train_dict'], shuffle=True)
    # }
    # batch_train_sampler = BatchSampler(samplers['train'], batch_size, drop_last=True)
    # batch_train_dict_sampler = BatchSampler(samplers['train_dict'], max(2, sc_batch_size), drop_last=True)

    # dataloaders['train'] = DataLoader(
    #     datasets['train'],
    #     batch_sampler=batch_train_sampler,
    #     collate_fn=collators['train'],
    #     num_workers=config.optimizer.num_workers,
    # )
    # dataloaders['valid'] = DataLoader(
    #     datasets['valid'],
    #     batch_size=batch_size,
    #     sampler=samplers['valid'],
    #     collate_fn=collators['valid'],
    #     num_workers=config.optimizer.num_workers,
    # )
    # dataloaders['train_dict'] = DataLoader(
    #     datasets['train_dict'],
    #     batch_sampler=batch_train_dict_sampler,
    #     collate_fn=collators['train_dict'],
    #     num_workers=config.optimizer.num_workers,
    # )
    return dataloaders
def build_coco_dataloaders(config=None, mode='freezing', device='cpu'):
    overfit = config.dataset.overfit
    transform = get_transform(config.dataset.transform_cfg)

    text_field = TextField(vocab_path=config.dataset.vocab_path)
    valid_field = ImageField(transform=transform['valid'], **config.dataset)
    train_field = ImageField(transform=transform['train'], **config.dataset)
    examples = COCO(None, text_field, **config.dataset).split_examples()

    if mode == 'freezing' and config.optimizer.freezing_xe_epochs > 0:
        valid_field.init_hdf5_feat()
        train_field.init_hdf5_feat()

    if mode == 'finetune':
        valid_field.use_hdf5_feat = False
        train_field.use_hdf5_feat = False

    datasets = {
        'train': CPairedDataset(examples['train'], train_field, overfit=overfit),
        'valid': CPairedDataset(examples['valid'], valid_field, overfit=overfit),
        'train_dict': CDictionaryDataset(examples['train'], train_field, overfit=overfit),
        'valid_dict': CDictionaryDataset(examples['valid'], valid_field, overfit=overfit),
        'test_dict': CDictionaryDataset(examples['test'], valid_field, overfit=overfit),
    }

    collators = {
        'train': PairedCollator(train_field, device=device),
        'valid': PairedCollator(valid_field, device=device),
        'train_dict': DictionaryCollator(train_field, device=device),
        'valid_dict': DictionaryCollator(valid_field, device=device),
        'test_dict': DictionaryCollator(valid_field, device=device),
    }

    batch_size = config.optimizer.batch_size * 4 if mode == 'freezing' else config.optimizer.batch_size
    sc_batch_size = config.optimizer.batch_size if mode == 'freezing' else config.optimizer.batch_size // 4

    dataloaders = {}
    # test and valid
    dataloaders['valid_dict'] = DataLoader(
        datasets['valid_dict'],
        batch_size=1,
        # batch_size=max(1, sc_batch_size * 2),
        num_workers=config.optimizer.num_workers,
        collate_fn=collators['valid_dict'],
    )
    dataloaders['test_dict'] = DataLoader(
        datasets['test_dict'],
        batch_size=1,
        # batch_size=max(1, sc_batch_size * 2),
        num_workers=config.optimizer.num_workers,
        collate_fn=collators['test_dict'],
    )

    if getattr(config.exp, 'eval', False):
        return dataloaders

    samplers = {
        'train': DistributedSampler(datasets['train'], shuffle=True),
        'valid': DistributedSampler(datasets['valid'], shuffle=False),
        'train_dict': DistributedSampler(datasets['train_dict'], shuffle=True)
    }
    batch_train_sampler = BatchSampler(samplers['train'], batch_size, drop_last=True)
    batch_train_dict_sampler = BatchSampler(samplers['train_dict'], max(2, sc_batch_size), drop_last=True)

    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_sampler=batch_train_sampler,
        collate_fn=collators['train'],
        num_workers=config.optimizer.num_workers,
    )
    dataloaders['valid'] = DataLoader(
        datasets['valid'],
        batch_size=batch_size,
        sampler=samplers['valid'],
        collate_fn=collators['valid'],
        num_workers=config.optimizer.num_workers,
    )
    dataloaders['train_dict'] = DataLoader(
        datasets['train_dict'],
        batch_sampler=batch_train_dict_sampler,
        collate_fn=collators['train_dict'],
        num_workers=config.optimizer.num_workers,
    )
    return dataloaders, samplers



def build_test_dataloaders(config=None, device='cpu', from_idx=0, to_idx=-1):
    datasets = {
        'test':
            TestDataset(
                root=os.path.join(os.environ['DATA_ROOT'], 'test2014'),
                anno_file=os.path.join(os.environ['DATA_ROOT'], 'annotations/image_info_test2014.json'),
                from_idx=from_idx,
                to_idx=to_idx,
            ),
        'valid':
            TestDataset(
                root=os.path.join(os.environ['DATA_ROOT'], 'val2014'),
                anno_file=os.path.join(os.environ['DATA_ROOT'], 'annotations/captions_val2014.json'),
                from_idx=from_idx,
                to_idx=to_idx,
            ),
    }
    collator = TestCollator(device=device)
    sc_batch_size = config.optimizer.batch_size

    dataloaders = {}
    # test and valid
    dataloaders['test'] = DataLoader(
        datasets['test'],
        batch_size=2,  # max(2, sc_batch_size),
        num_workers=1,  # config.optimizer.num_workers,
        collate_fn=collator,
    )
    dataloaders['valid'] = DataLoader(
        datasets['valid'],
        batch_size=2,  # max(2, sc_batch_size),
        num_workers=1,  # config.optimizer.num_workers,
        collate_fn=collator,
    )
    return dataloaders
