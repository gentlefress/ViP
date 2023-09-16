# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.tucker import *
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (BertConfig, BertTokenizer, WEIGHTS_NAME)
from models.text_encoder import BertEncoder
from models.image_encoders import InceptionV3Encoder, ResNetEncoder
from models.vip import vipUnidirectional, vipBidirectional

from utils.pretrain_config import cfg, cfg_from_file
from utils.data import save_h5, HDF5DatasetWithImage
from utils.pretrain_data_bunch import output_modes, processors, convert_examples_to_features
import h5py
import torchvision.transforms as transforms
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

LM_MODEL_CLASSES = {
    'bert': (BertConfig, BertEncoder, BertTokenizer),
}

MODEL_CLASSES = {
    'bert': (BertConfig, BertEncoder, BertTokenizer),
}

vip_CLASSES = {
    'unidirectional': vipUnidirectional,
    'bidirectional': vipBidirectional,
}

def hinge(x):
    return torch.clamp(x, min=0.)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # self.bn2 = nn.BatchNorm1d(output_size)
        self.act = nn.Tanh()
    def forward(self, x):
        out1 = self.linear1(x)
        # out2 = self.bn1(out1)
        out3 = self.act(out1)
        out4 = self.linear2(out3)
        # out5 = self.bn2(out4)
        return out4
class MLP_3d(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(MLP_3d, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        # self.bn1 = nn.BatchNorm1d(batch_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # self.bn2 = nn.BatchNorm1d(batch_size)
        self.act = nn.Tanh()
    def forward(self, x):
        out1 = self.linear1(x)
        # out2 = self.bn1(out1)
        out3 = self.act(out1)
        out4 = self.linear2(out3)
        # out5 = self.bn2(out4)
        return out4

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory,
                                            cfg.TEXT.MODEL_NAME + '_cached_lm_' + str(block_size) + '_' + filename)
        self.cached_features_file = cached_features_file

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            # with open(cached_features_file, 'rb') as handle:
            #     self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.input_ids = []
            self.attention_mask = []
            self.token_type_ids = []
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    text = line.strip()
                    inputs = tokenizer.encode_plus(
                        text,
                        None,
                        add_special_tokens=True,
                        max_length=block_size,
                    )
                    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

                    attention_mask = [1] * len(input_ids)
                    padding_length = block_size - len(input_ids)
                    pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
                    input_ids = input_ids + ([pad_token] * padding_length)
                    attention_mask = attention_mask + ([0] * padding_length)
                    pad_token_segment_id = 4 if cfg.TEXT.MODEL_TYPE in ['xlnet'] else 0
                    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

                    assert len(input_ids) == len(attention_mask) == len(token_type_ids) == block_size
                    self.input_ids.append(input_ids)
                    self.attention_mask.append(attention_mask)
                    self.token_type_ids.append(token_type_ids)

            logger.info("Saving features into cached file %s", cached_features_file)
            save_h5(cached_features_file, np.array(self.input_ids, dtype=np.long), 'input_ids')
            save_h5(cached_features_file, np.array(self.attention_mask, dtype=np.long), 'attention_mask')
            save_h5(cached_features_file, np.array(self.token_type_ids, dtype=np.long), 'token_type_ids')
            # with open(cached_features_file, 'wb') as handle:
            #     pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with h5py.File(self.cached_features_file, 'r') as dfile:
            self.len = dfile['input_ids'].shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        vals = []
        attributes = ['input_ids', 'attention_mask', 'token_type_ids']
        for attr in attributes:
            val = torch.tensor(self.get_data(attr, item))
            vals.append(val)
        return tuple(vals)

    def get_data(self, attribute, idx):
        with h5py.File(self.cached_features_file, 'r') as dfile:
            return dfile[attribute][idx]


def lm_load_and_cache_examples(args, tokenizer, file_path):
    dataset = TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    return dataset

def load_and_cache_examples(args, task, tokenizer, evaluate=False, save_interval=10000):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(cfg.DATA_DIR, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, cfg.TEXT.MODEL_NAME.split('/'))).pop(),
        str(cfg.TEXT.MAX_LEN),
        str(task)))

    cached_features_file_aug = os.path.join(cfg.DATA_DIR, 'cached_{}_{}_{}_{}'.format(
        'dev_aug' if evaluate else 'train_aug',
        list(filter(None, cfg.TEXT.MODEL_NAME.split('/'))).pop(),
        str(cfg.TEXT.MAX_LEN),
        str(task)))

    if cfg.IMAGE.ENCODER == 'resnet-50':
        img_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(cfg.IMAGE.CROP_SIZE),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomGrayscale(p=0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])
        img_notransform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(cfg.IMAGE.CROP_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])
    else:
        img_transform = transforms.Compose([transforms.Resize(int(cfg.IMAGE.CROP_SIZE * 76 / 64)),
                                            transforms.RandomCrop(cfg.IMAGE.CROP_SIZE),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                 std=[0.5, 0.5, 0.5])
                                            ])
        img_notransform = transforms.Compose([transforms.Resize(256),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", cfg.DATA_DIR)
        if os.path.isfile(cached_features_file):
            logger.info("Deleting existed cached file %s", cached_features_file)
            os.remove(cached_features_file)


    if os.path.exists(cached_features_file_aug) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file_aug)
    else:
        logger.info("Creating features from dataset file at %s", cfg.DATA_DIR)
        if os.path.isfile(cached_features_file_aug):
            logger.info("Deleting existed cached file %s", cached_features_file_aug)
            os.remove(cached_features_file_aug)

        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and cfg.TEXT.MODEL_TYPE in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(cfg.DATA_DIR,
                                              cfg.IMAGE_DIR) if evaluate else processor.get_train_examples(
            cfg.DATA_DIR, cfg.IMAGE_DIR)
        examples_aug = processor.get_dev_aug_examples(cfg.DATA_DIR,
                                              cfg.IMAGE_DIR) if evaluate else processor.get_train_aug_examples(
            cfg.DATA_DIR, cfg.IMAGE_DIR)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=cfg.TEXT.MAX_LEN,
                                                img_size=cfg.IMAGE.RESIZE_SIZE,
                                                crop_size=cfg.IMAGE.CROP_SIZE,
                                                output_mode=output_mode,
                                                pad_on_left=bool(cfg.TEXT.MODEL_TYPE in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if cfg.TEXT.MODEL_TYPE in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            # torch.save(features, cached_features_file)
            all_input_ids = []
            all_images = []
            all_attention_mask = []
            all_token_type_ids = []
            all_labels = []

            for (f_idx, f) in enumerate(features):
                all_input_ids.append(f.input_ids)
                all_images.append(f.image.encode())
                all_attention_mask.append(f.attention_mask)
                all_token_type_ids.append(f.token_type_ids)
                all_labels.append(f.label)
                if len(all_input_ids) == save_interval:
                    logger.info("Saving example %d" % (f_idx + 1))
                    save_h5(cached_features_file, np.array(all_input_ids, dtype=np.long), 'input_ids')
                    save_h5(cached_features_file, np.array(all_images), 'images')
                    save_h5(cached_features_file, np.array(all_attention_mask, dtype=np.long), 'attention_mask')
                    save_h5(cached_features_file, np.array(all_token_type_ids, dtype=np.long), 'token_type_ids')
                    if output_mode == "classification":
                        save_h5(cached_features_file, np.array(all_labels, dtype=np.long), "labels")
                    elif output_mode == "regression":
                        save_h5(cached_features_file, np.array(all_labels, dtype=np.float), "labels")
                    all_input_ids = []
                    all_images = []
                    all_attention_mask = []
                    all_token_type_ids = []
                    all_labels = []

            if len(all_input_ids) != 0:
                save_h5(cached_features_file, np.array(all_input_ids, dtype=np.long), 'input_ids')
                save_h5(cached_features_file, np.array(all_images), 'images')
                save_h5(cached_features_file, np.array(all_attention_mask, dtype=np.long), 'attention_mask')
                save_h5(cached_features_file, np.array(all_token_type_ids, dtype=np.long), 'token_type_ids')
                if output_mode == "classification":
                    save_h5(cached_features_file, np.array(all_labels, dtype=np.long), "labels")
                elif output_mode == "regression":
                    save_h5(cached_features_file, np.array(all_labels, dtype=np.float), "labels")

        feature_aug = convert_examples_to_features(examples_aug,
                                                    tokenizer,
                                                    label_list=label_list,
                                                    max_length=cfg.TEXT.MAX_LEN,
                                                    img_size=cfg.IMAGE.RESIZE_SIZE,
                                                    crop_size=cfg.IMAGE.CROP_SIZE,
                                                    output_mode=output_mode,
                                                    pad_on_left=bool(cfg.TEXT.MODEL_TYPE in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if cfg.TEXT.MODEL_TYPE in ['xlnet'] else 0,
                                                    )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file_aug)
            # torch.save(features, cached_features_file)
            all_input_ids = []
            all_images = []
            all_attention_mask = []
            all_token_type_ids = []
            all_labels = []

            for (f_idx, f) in enumerate(feature_aug):
                all_input_ids.append(f.input_ids)
                all_images.append(f.image.encode())
                all_attention_mask.append(f.attention_mask)
                all_token_type_ids.append(f.token_type_ids)
                all_labels.append(f.label)
                if len(all_input_ids) == save_interval:
                    logger.info("Saving example %d" % (f_idx + 1))
                    save_h5(cached_features_file_aug, np.array(all_input_ids, dtype=np.long), 'input_ids')
                    save_h5(cached_features_file_aug, np.array(all_images), 'images')
                    save_h5(cached_features_file_aug, np.array(all_attention_mask, dtype=np.long), 'attention_mask')
                    save_h5(cached_features_file_aug, np.array(all_token_type_ids, dtype=np.long), 'token_type_ids')
                    if output_mode == "classification":
                        save_h5(cached_features_file_aug, np.array(all_labels, dtype=np.long), "labels")
                    elif output_mode == "regression":
                        save_h5(cached_features_file_aug, np.array(all_labels, dtype=np.float), "labels")
                    all_input_ids = []
                    all_images = []
                    all_attention_mask = []
                    all_token_type_ids = []
                    all_labels = []

            if len(all_input_ids) != 0:
                save_h5(cached_features_file_aug, np.array(all_input_ids, dtype=np.long), 'input_ids')
                save_h5(cached_features_file_aug, np.array(all_images), 'images')
                save_h5(cached_features_file_aug, np.array(all_attention_mask, dtype=np.long), 'attention_mask')
                save_h5(cached_features_file_aug, np.array(all_token_type_ids, dtype=np.long), 'token_type_ids')
                if output_mode == "classification":
                    save_h5(cached_features_file_aug, np.array(all_labels, dtype=np.long), "labels")
                elif output_mode == "regression":
                    save_h5(cached_features_file_aug, np.array(all_labels, dtype=np.float), "labels")

        if args.local_rank == 0 and not evaluate:
            torch.distributed.barrier()
    dataset_aug = HDF5DatasetWithImage(cached_features_file_aug,
                                   ['input_ids', 'images', 'attention_mask', 'token_type_ids', 'labels'],
                                   img_transform)
    dataset = HDF5DatasetWithImage(cached_features_file,
                                   ['input_ids', 'images', 'attention_mask', 'token_type_ids', 'labels'],
                                   img_notransform)
    return dataset, dataset_aug


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
                           labels.tolist()]
    padding_mask = (labels == 0)
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.uint8), value=0.0)
    probability_matrix.masked_fill_(padding_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).to(torch.bool)
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).byte() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).byte() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, train_dataset_aug, model, image_encoder, lm_model, momentum_encoder, momentum_resnet, vip, config, tokenizer, tucker):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  num_workers=cfg.NUM_WORKERS, drop_last=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        args.logging_steps = len(train_dataloader) // args.gradient_accumulation_steps
        args.save_steps = args.logging_steps

    train_sampler_aug = RandomSampler(train_dataset_aug) if args.local_rank == -1 else DistributedSampler(train_dataset_aug)
    train_dataloader_aug = DataLoader(train_dataset_aug, sampler=train_sampler_aug, batch_size=args.train_batch_size,
                                  num_workers=cfg.NUM_WORKERS, drop_last=True)




    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader_aug) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader_aug) // args.gradient_accumulation_steps * args.num_train_epochs
        args.logging_steps = len(train_dataloader_aug) // args.gradient_accumulation_steps
        args.save_steps = args.logging_steps
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
                   + [p for n, p in image_encoder.named_parameters() if not any(nd in n for nd in no_decay)]
                   + [p for n, p in vip.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
                   + [p for n, p in image_encoder.named_parameters() if any(nd in n for nd in no_decay)]
                   + [p for n, p in vip.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.1),
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        image_encoder = torch.nn.DataParallel(image_encoder)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        image_encoder = torch.nn.parallel.DistributedDataParallel(image_encoder, device_ids=[args.local_rank],
                                                                  output_device=args.local_rank,
                                                                  find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset_aug))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_loss_on_dev = 10e9
    m = 0.90
    margin1 = 0.2
    margin2 = 0.4
    model.zero_grad()
    momentum_encoder.zero_grad()
    image_encoder.zero_grad()
    vip.zero_grad()
    gradlist = None
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
    cosine_loss_fct = nn.CosineEmbeddingLoss(reduction='mean')
    text_mlp = MLP(768, 256, 128).to(args.device)
    image_mlp = MLP(768, 256, 128).to(args.device)
    image_mlp_aug = MLP(768, 256, 128).to(args.device)
    text_mlp_aug = MLP(768, 256, 128).to(args.device)
    # 改一下看看效果
    text_local_mlp = MLP_3d(768, 256, 128, 32).to(args.device)
    # text_local_mlp = MLP(768, 256, 128).to(args.device)
    # text_local_mlp = nn.Linear(768, 128).to(args.device)
    # 处理特征输入队列
    # 注释
    # queue = torch.randn((128, 65536))
    # queue = queue.to(args.device)


    # for step, batch in enumerate(epoch_iterator):
    #     batch = tuple(t.to(args.device) for t in batch)
    #     text_encoder_inputs = {'input_ids': batch[0],
    #                            'attention_mask': batch[2], }
    #     if cfg.TEXT.MODEL_TYPE != 'distilbert':
    #         text_encoder_inputs['token_type_ids'] = batch[3] if cfg.TEXT.MODEL_TYPE in ['bert', 'xlnet'] else None
    #     text_fea = momentum_encoder(**text_encoder_inputs)
    #     text_fea = text_mlp(text_fea[4])
    #     queue = torch.concat([queue[:, 128*(step+1):], text_fea.permute(1, 0)], dim=1)

    # 随机初始化队列
    # queue = torch.randn((128, 65536))



    for _ in train_iterator:
        epoch_loss, epoch_step = 0.0, 0
        s_total_loss0, s_total_loss1, w_total_loss0, w_total_loss1, m_total_loss, cl_total_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        s_total_loss2, s_total_loss3, w_total_loss2 = 0.0, 0.0, 0.0
        s_total_loss4, s_total_loss5 = 0.0, 0.0
        s_total_loss6, s_total_loss7 = 0.0, 0.0
        caption_total, swap_total = 0.0, 0.0
        ce_total_loss, cos_total_loss = 0.0, 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_aug = tqdm(train_dataloader_aug, desc="Iteration", disable=args.local_rank not in [-1, 0])
        queue = torch.randn((128, 65536))
        queue = nn.functional.normalize(queue, dim=0)
        queue = queue.to(args.device)
        queue_im = torch.randn((128, 65536))
        queue_im = nn.functional.normalize(queue_im, dim=0)
        queue_im = queue_im.to(args.device)
        for step, batch in enumerate(zip(epoch_iterator, epoch_iterator_aug)):
            model.train()
            image_encoder.train()
            vip.train()
            lm_model.eval()
            momentum_resnet.eval()
            momentum_encoder.eval()
            if epoch_step != 0:
                epoch_iterator.set_description(
                    'Train Loss: {:.2f}| ce_loss: {:3.2f}| cos_loss: {:3.2f}| s_loss: {:3.2f} {:3.2f} {:3.2f} {:3.2f} {:3.2f} {:3.2f} {:3.2f} {:3.2f}| w_loss: {:3.2f} {:3.2f}| m_loss: {:3.2f}| cl_loss: {:3.2f}| caption_loss: {:3.2f}| swap_loss: {:3.2f}'.
                        format(epoch_loss / epoch_step, ce_total_loss / epoch_step,
                               cos_total_loss / epoch_step,
                               s_total_loss0 / epoch_step, s_total_loss1 / epoch_step,
                               s_total_loss2 / epoch_step, s_total_loss3 / epoch_step,
                               s_total_loss4 / epoch_step, s_total_loss5 / epoch_step,
                               s_total_loss6 / epoch_step, s_total_loss7 / epoch_step,
                               w_total_loss0 / epoch_step, w_total_loss1 / epoch_step,
                               m_total_loss / epoch_step, cl_total_loss / epoch_step,
                               caption_total / epoch_step, swap_total / epoch_step
                               ))
            batch = tuple(r.to(args.device)  for t in batch for r in t)
            image_encoder_inputs = {'x': batch[1]}
            image_encoder_inputs_aug = {'x': batch[6]}
            text_encoder_inputs = {'input_ids': batch[0],
                                   'attention_mask': batch[2], }
            text_encoder_inputs_aug = {'input_ids': batch[5],
                                   'attention_mask': batch[7], }

            if cfg.TEXT.MODEL_TYPE != 'distilbert':
                text_encoder_inputs['token_type_ids'] = batch[3] if cfg.TEXT.MODEL_TYPE in ['bert',
                                                                                            'xlnet'] else None
            if cfg.TEXT.MODEL_TYPE != 'distilbert':
                text_encoder_inputs_aug['token_type_ids'] = batch[8] if cfg.TEXT.MODEL_TYPE in ['bert',
                                                                                                'xlnet'] else None

            # XLM, DistilBERT and RoBERTa don't use segment_ids

            batch_size = batch[0].size(0)
            labels = torch.LongTensor(range(batch_size))
            labels = labels.to(args.device)

            image_features, global_image_features = image_encoder(**image_encoder_inputs)
            text_outputs = model(**text_encoder_inputs)
            with torch.no_grad():

                text_outputs_aug = momentum_encoder(**text_encoder_inputs)
                image_features_aug, global_image_features_aug = momentum_resnet(**image_encoder_inputs_aug)
                sim_t = F.cosine_similarity(text_outputs[4], text_outputs_aug[4])
                sim_t = torch.mean(sim_t, dim=-1)
                sim_i = F.cosine_similarity(global_image_features, global_image_features_aug)
                sim_i = torch.mean(sim_i, dim=-1)
                # sim_t = m
                # sim_i = m
                for param_q, param_k in zip(model.parameters(), momentum_encoder.parameters()):
                    param_k.data = param_q.data*sim_t + param_k.data*(1-sim_t)
                for param_i, param_m in zip(image_encoder.parameters(), momentum_resnet.parameters()):
                    param_m.data = param_i.data*sim_i + param_m.data*(1-sim_i)
            text_features, global_text_features = text_outputs[3], text_outputs[4]
            text_features_aug, global_text_features_aug = text_outputs_aug[3].detach(), text_outputs_aug[4].detach()

            # MLP操作共享与非共享参数
            global_image_features = image_mlp(global_image_features)
            global_image_features_aug = image_mlp_aug(global_image_features_aug)
            global_text_features = text_mlp(global_text_features)
            global_text_features_aug = text_mlp_aug(global_text_features_aug)


            # tucker
            global_features = tucker(global_text_features, global_image_features)
            global_aug_features = tucker(global_text_features_aug, global_image_features_aug)
            l_neg = torch.einsum('nc,ck->nk', [global_features, global_aug_features.T])
            cl_loss = nn.CrossEntropyLoss()(l_neg / 0.08, labels)




            w_loss0, w_loss1, s_loss0, s_loss1, m_loss, s_loss6, s_loss7 = vip(global_image_features, global_text_features,
                                                      image_features, text_features, global_image_features_aug,
                                                      global_text_features_aug, queue, queue_im,
                                                      sentence_mask=batch[2], labels=labels)

            # caption loss part
            local_text = text_local_mlp(text_features)
            lang_mask = batch[2]
            # rnn_code_normalize = nn.functional.normalize(global_text_features, dim=-1)
            cnn_code_normalize = nn.functional.normalize(global_image_features, dim=-1)
            text_normalize = nn.functional.normalize(local_text, dim=-1)
            b, lang_len, dim = text_normalize.shape
            hb = b//2
            # split to positive and negative sets
            pos_lang, neg_lang = torch.split(text_normalize, hb, dim=0)
            pos_vision, neg_vision = torch.split(cnn_code_normalize, hb, dim=0)
            # calculate positive and negative scores

            true_pos_scores = (pos_lang * pos_vision).sum(-1)  # [batchsize/2, maxlen]

            true_neg_scores = (neg_lang * neg_vision).sum(-1)
            false_pos_scores = (pos_lang * neg_vision).sum(-1)
            false_neg_scores = (neg_lang * pos_vision).sum(-1)


            # Hinge loss
            float_mask = lang_mask.type(local_text.dtype)
            pos_lang_mask, neg_lang_mask = torch.split(float_mask, hb, dim=0)
            pos_loss = hinge(margin1 - true_pos_scores + false_pos_scores) * pos_lang_mask
            neg_loss = hinge(margin1 - true_neg_scores + false_neg_scores) * neg_lang_mask

            # averaging
            cnt = float_mask.sum()
            caption_loss = (pos_loss.sum() + neg_loss.sum()) / cnt


            vip_loss = cfg.vip.LOCAL_WEIGHT * (w_loss0 + w_loss1) \
                        + cfg.vip.GLOBAL_WEIGHT * (s_loss0 + s_loss1 + s_loss6 + s_loss7) + m_loss + cl_loss*0.04
            queue = torch.concat([queue[:, batch_size:], global_text_features_aug.permute(1, 0)], dim=1)
            queue_im = torch.concat([queue_im[:, batch_size:], global_image_features_aug.permute(1, 0)], dim=1)


            global_text_norm = nn.functional.normalize(global_text_features, dim=-1)
            global_image_norm = nn.functional.normalize(global_image_features, dim=-1)

            attention = torch.div(
                torch.einsum('nc,ck->nk', [global_text_norm, global_image_norm.T]),
                math.sqrt(global_image_norm.size(-1)))
            attention = nn.Softmax()(attention)
            global_features = torch.mm(attention, global_image_features)
            attention = torch.div(
                torch.einsum('nc,ck->nk', [global_image_norm, global_text_norm.T]),
                math.sqrt(global_text_norm.size(-1)))
            attention = nn.Softmax()(attention)
            global_image = torch.mm(attention, global_text_features)

            # 归一化交叉注意力特征
            cross_image_norm = nn.functional.normalize(global_image, dim=-1)
            cross_text_norm = nn.functional.normalize(global_features, dim=-1)
            swap_loss = 0
            for i in range(batch_size):
                number = random.uniform(0, 1)
                if(number >= 0.5):
                    num = random.randint(0, batch_size-1)
                    if num != i:
                        sim = torch.matmul(global_text_norm[i][:], global_image_norm[i][:]) + \
                                0.5 * torch.matmul(cross_text_norm[i][:], cross_image_norm[i][:])
                        sim_t = torch.matmul(global_text_norm[num][:], global_image_norm[i][:]) + \
                                0.5 * torch.matmul(cross_text_norm[num][:], cross_image_norm[i][:])
                        swap_loss += hinge(margin2 - sim + sim_t)

            # MLM part  lifelong learning
            text_encoder_inputs['input_ids'] = text_encoder_inputs['input_ids'].cpu()
            text_encoder_inputs['input_ids'], mlm_labels = mask_tokens(text_encoder_inputs['input_ids'], tokenizer,
                                                                       args) if args.mlm \
                else (text_encoder_inputs['input_ids'], text_encoder_inputs['input_ids'])
            text_encoder_inputs['input_ids'] = text_encoder_inputs['input_ids'].to(args.device)
            with torch.no_grad():
                t_outputs = lm_model(**text_encoder_inputs)
            s_outputs = model(**text_encoder_inputs)

            ## CE part
            t_pred_scores = t_outputs[0].detach()  # prediction scores
            s_pred_scores = s_outputs[0]
            mask = (mlm_labels > -1).unsqueeze(-1).expand_as(t_pred_scores).to(args.device)
            t_pred_scores_slct = torch.masked_select(t_pred_scores, mask)
            t_pred_scores_slct = t_pred_scores_slct.view(-1, t_pred_scores.size(-1))
            s_pred_scores_slct = torch.masked_select(s_pred_scores, mask)
            s_pred_scores_slct = s_pred_scores_slct.view(-1, s_pred_scores.size(-1))
            assert t_pred_scores_slct.size() == s_pred_scores_slct.size()
            ce_loss = ce_loss_fct(F.log_softmax(s_pred_scores_slct / cfg.LWF.TEMPERATURE, dim=-1),
                                  F.softmax(t_pred_scores_slct / cfg.LWF.TEMPERATURE,
                                            dim=-1)) * cfg.LWF.TEMPERATURE ** 2
            kd_loss = cfg.LWF.ALPHA_CE * ce_loss


            ## Cosine part
            if cfg.LWF.LAYER_WISE:
                t_hidden_states_total = t_outputs[5]
                s_hidden_states_total = s_outputs[5]
                assert len(t_hidden_states_total) == len(s_hidden_states_total) == config.num_hidden_layers + 1
                cos_loss = 0.0
                for i in range(len(t_hidden_states_total)):
                    t_hidden_states = t_hidden_states_total[i]
                    s_hidden_states = s_hidden_states_total[i]
                    mask = batch[2].unsqueeze(-1).expand_as(t_hidden_states).byte().to(args.device)
                    assert t_hidden_states.size() == s_hidden_states.size()
                    dim = s_hidden_states.size(-1)
                    t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
                    t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                    s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
                    s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                    target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
                    cos_loss += cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
                kd_loss += cfg.LWF.ALPHA_COS * cos_loss
            else:
                t_hidden_states = t_outputs[1]
                s_hidden_states = s_outputs[1]
                mask = batch[2].unsqueeze(-1).expand_as(t_hidden_states).byte().to(args.device)
                assert t_hidden_states.size() == s_hidden_states.size()
                dim = s_hidden_states.size(-1)
                t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
                t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
                s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
                cos_loss = cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
                kd_loss += cfg.LWF.ALPHA_COS * cos_loss

            loss = vip_loss + cfg.LWF.LAMBDA0 * kd_loss + swap_loss + caption_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                ce_loss = ce_loss / args.gradient_accumulation_steps
                cos_loss = cos_loss / args.gradient_accumulation_steps
                s_loss0 = s_loss0 / args.gradient_accumulation_steps
                s_loss1 = s_loss1 / args.gradient_accumulation_steps
                w_loss0 = w_loss0 / args.gradient_accumulation_steps
                w_loss1 = w_loss1 / args.gradient_accumulation_steps
                # w_loss2 = w_loss2 / args.gradient_accumulation_steps
                m_loss = m_loss / args.gradient_accumulation_steps
                cl_loss = cl_loss / args.gradient_accumulation_steps
                swap_loss = swap_loss / args.gradient_accumulation_steps
                caption_loss = caption_loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                for name, params in model.named_parameters():
                    if name == 'cls.predictions.transform.dense.weight':
                        p_g = torch.mean(params.grad, dim=-1)
                        if step == 0:
                            gradlist = p_g.detach().cpu().numpy()
                        else:
                            gradlist = np.append(gradlist, p_g.detach().cpu().numpy())

            tr_loss += loss.item()
            epoch_loss += loss.item()
            ce_total_loss += ce_loss.item()
            cos_total_loss += cos_loss.item()
            s_total_loss0 += s_loss0.item()
            s_total_loss1 += s_loss1.item()
            s_total_loss6 += s_loss6.item()
            s_total_loss7 += s_loss7.item()
            w_total_loss0 += w_loss0.item()
            w_total_loss1 += w_loss1.item()
            m_total_loss += m_loss.item()
            cl_total_loss += cl_loss.item()
            caption_total += caption_loss.item()

            swap_total += swap_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    # torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                image_encoder.zero_grad()
                vip.zero_grad()
                global_step += 1
                epoch_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, image_encoder, lm_model, momentum_encoder, momentum_resnet, vip, config, tokenizer)
                        for key, value in results.items():
                            eval_key = 'eval_{}'.format(key)
                            logs[eval_key] = value

                        if results['loss'] < best_loss_on_dev:
                            best_loss_on_dev = results['loss']
                            # Save best model
                            output_dir = cfg.OUTPUT_DIR
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)

                            mm_to_save = momentum_encoder.module if hasattr(momentum_encoder,
                                                                    'module') else momentum_encoder  # Take care of distributed/parallel training
                            mm_to_save.save_pretrained(os.path.join(output_dir, 'momentum.bin'))
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            torch.save(image_encoder.state_dict(), os.path.join(output_dir, 'image_encoder.bin'))
                            torch.save(vip.state_dict(), os.path.join(output_dir, 'vip.bin'))
                            logger.info("Saving best model to %s", output_dir)

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs['learning_rate'] = learning_rate_scalar
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{'step': global_step}}))

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(cfg.OUTPUT_DIR, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)

                    mm_to_save = momentum_encoder.module if hasattr(momentum_encoder,
                                                                    'module') else momentum_encoder  # Take care of distributed/parallel training
                    mm_to_save.save_pretrained(output_dir, 'momentum.bin')

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    torch.save(image_encoder.state_dict(), os.path.join(output_dir, 'image_encoder.bin'))
                    torch.save(vip.state_dict(), os.path.join(output_dir, 'vip.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
    np.save('grad.npy', gradlist)
    return global_step, tr_loss / global_step


def evaluate(args, model, image_encoder, lm_model, momentum_bert, momentum_resnet, vip, config, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if cfg.TASK_NAME == "mnli" else (cfg.TASK_NAME,)
    eval_outputs_dirs = (cfg.OUTPUT_DIR, cfg.OUTPUT_DIR + '-MM') if cfg.TASK_NAME == "mnli" else (cfg.OUTPUT_DIR,)
    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_dataset_aug = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        # eval_sampler = SequentialSampler(eval_dataset)
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=cfg.NUM_WORKERS, drop_last=True)

        eval_sampler_aug = RandomSampler(eval_dataset_aug)
        eval_dataloader_aug = DataLoader(eval_dataset_aug, sampler=eval_sampler_aug, batch_size=args.eval_batch_size,
                                     num_workers=cfg.NUM_WORKERS, drop_last=True)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        eval_vip_loss, eval_ce_loss, eval_cos_loss = 0.0, 0.0, 0.0
        nb_eval_steps = 0
        ce_loss_fct = nn.KLDivLoss(reduction='batchmean')
        cosine_loss_fct = nn.CosineEmbeddingLoss(reduction='mean')
        epoch_iterator = tqdm(eval_dataloader, desc="Evaluating")
        epoch_iterator_aug = tqdm(eval_dataloader_aug, desc="Evaluating")
        queue = torch.randn((768, 65536))
        queue = nn.functional.normalize(queue, dim=0)
        queue_im = torch.randn((128, 65536))
        queue_im = nn.functional.normalize(queue_im, dim=0)
        queue_im = queue_im.to(args.device)
        m = 0.99
        for _, batch in enumerate(zip(epoch_iterator, epoch_iterator_aug)):
            model.eval()
            image_encoder.eval()
            vip.eval()
            lm_model.eval()
            batch = tuple(r.to(args.device) for t in batch for r in t)
            with torch.no_grad():
                image_encoder_inputs = {'x': batch[1]}
                text_encoder_inputs = {'input_ids': batch[0],
                                       'attention_mask': batch[2], }
                text_encoder_inputs_aug = {'input_ids': batch[5],
                                           'attention_mask': batch[7], }
                image_features_aug, global_image_features_aug = momentum_resnet(**image_encoder_inputs)
                text_outputs_aug = momentum_bert(**text_encoder_inputs_aug)

                if cfg.TEXT.MODEL_TYPE != 'distilbert':
                    text_encoder_inputs['token_type_ids'] = batch[3] if cfg.TEXT.MODEL_TYPE in ['bert',
                                                                                                'xlnet'] else None
                if cfg.TEXT.MODEL_TYPE != 'distilbert':
                    text_encoder_inputs_aug['token_type_ids'] = batch[8] if cfg.TEXT.MODEL_TYPE in ['bert',
                                                                                                    'xlnet'] else None
                for param_q, param_k in zip(model.parameters(), momentum_bert.parameters()):
                    param_k.data = param_q.data * m + param_k.data * (1 - m)
                for param_i, param_m in zip(image_encoder.parameters(), momentum_resnet.parameters()):
                    param_m.data = param_i.data * m + param_m.data * (1 - m)
                    # XLM, DistilBERT and RoBERTa don't use segment_ids

                batch_size = batch[0].size(0)
                labels = torch.LongTensor(range(batch_size))
                labels = labels.to(args.device)

                # vip part

                image_features, global_image_features = image_encoder(**image_encoder_inputs)
                text_outputs = model(**text_encoder_inputs)

                text_features, global_text_features = text_outputs[3], text_outputs[4]
                text_features_aug, global_text_features_aug = text_outputs_aug[3], text_outputs_aug[4]
                w_loss0, w_loss1, s_loss0, s_loss1, m_loss, s_loss2, s_loss3 = vip(global_image_features,
                                                                                    global_text_features,
                                                                                    image_features, text_features,
                                                                                    global_image_features_aug,
                                                                                    global_text_features_aug, queue,
                                                                                    queue_im,
                                                                                    sentence_mask=batch[2],
                                                                                    labels=labels)
                
                vip_loss = cfg.vip.LOCAL_WEIGHT * (w_loss0 + w_loss1) \
                            + cfg.vip.GLOBAL_WEIGHT * (s_loss0 + s_loss1 + s_loss2 + s_loss3) + m_loss

                # MLM part
                text_encoder_inputs['input_ids'] = text_encoder_inputs['input_ids'].cpu()
                text_encoder_inputs['input_ids'], mlm_labels = mask_tokens(text_encoder_inputs['input_ids'],
                                                                            tokenizer, args) if args.mlm \
                    else (text_encoder_inputs['input_ids'], text_encoder_inputs['input_ids'])
                text_encoder_inputs['input_ids'] = text_encoder_inputs['input_ids'].to(args.device)
                t_outputs = lm_model(**text_encoder_inputs)
                s_outputs = model(**text_encoder_inputs)

                ## CE part
                t_pred_scores = t_outputs[0].detach()
                s_pred_scores = s_outputs[0]
                mask = (mlm_labels > -1).unsqueeze(-1).expand_as(t_pred_scores).to(args.device)
                t_pred_scores_slct = torch.masked_select(t_pred_scores, mask)
                t_pred_scores_slct = t_pred_scores_slct.view(-1, t_pred_scores.size(-1))
                s_pred_scores_slct = torch.masked_select(s_pred_scores, mask)
                s_pred_scores_slct = s_pred_scores_slct.view(-1, s_pred_scores.size(-1))
                assert t_pred_scores_slct.size() == s_pred_scores_slct.size()
                ce_loss = ce_loss_fct(F.log_softmax(s_pred_scores_slct / cfg.LWF.TEMPERATURE, dim=-1),
                                      F.softmax(t_pred_scores_slct / cfg.LWF.TEMPERATURE,
                                                dim=-1)) * cfg.LWF.TEMPERATURE ** 2
                kd_loss = cfg.LWF.ALPHA_CE * ce_loss

                ## Cosine part
                if cfg.LWF.LAYER_WISE:
                    t_hidden_states_total = t_outputs[5]
                    s_hidden_states_total = s_outputs[5]
                    assert len(t_hidden_states_total) == len(s_hidden_states_total) == config.num_hidden_layers + 1
                    cos_loss = 0.0
                    for i in range(len(t_hidden_states_total)):
                        t_hidden_states = t_hidden_states_total[i]
                        s_hidden_states = s_hidden_states_total[i]
                        mask = batch[2].unsqueeze(-1).expand_as(t_hidden_states).byte().to(args.device)
                        assert t_hidden_states.size() == s_hidden_states.size()
                        dim = s_hidden_states.size(-1)
                        t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
                        t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                        s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
                        s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                        target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
                        cos_loss += cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
                    kd_loss += cfg.LWF.ALPHA_COS * cos_loss
                else:
                    t_hidden_states = t_outputs[1]
                    s_hidden_states = s_outputs[1]
                    mask = batch[2].unsqueeze(-1).expand_as(t_hidden_states).byte().to(args.device)
                    assert t_hidden_states.size() == s_hidden_states.size()
                    dim = s_hidden_states.size(-1)
                    t_hidden_states_slct = torch.masked_select(t_hidden_states, mask)  # (bs * seq_length * dim)
                    t_hidden_states_slct = t_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                    s_hidden_states_slct = torch.masked_select(s_hidden_states, mask)  # (bs * seq_length * dim)
                    s_hidden_states_slct = s_hidden_states_slct.view(-1, dim)  # (bs * seq_length, dim)
                    target = s_hidden_states_slct.new(s_hidden_states_slct.size(0)).fill_(1)  # (bs * seq_length,)
                    cos_loss = cosine_loss_fct(s_hidden_states_slct, t_hidden_states_slct, target)
                    kd_loss += cfg.LWF.ALPHA_COS * cos_loss

                tmp_eval_loss = vip_loss + cfg.LWF.LAMBDA0 * kd_loss

                eval_loss += tmp_eval_loss.mean().item()
                eval_vip_loss = 0.0
                eval_ce_loss += ce_loss.mean().item()
                eval_cos_loss += cos_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_vip_loss = eval_vip_loss / nb_eval_steps
        eval_ce_loss = eval_ce_loss / nb_eval_steps
        eval_cos_loss = eval_cos_loss / nb_eval_steps

        result = {'loss': eval_loss,
                  'vip_loss': eval_vip_loss,
                  'ce_loss': eval_ce_loss,
                  'cos_loss': eval_cos_loss}
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--cfg", default=None, type=str, required=True)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--multi_gpu", default=False, action='store_true')

    ## Other parameters
    parser.add_argument("--evaluate_during_training", default=True, action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")

    parser.add_argument("--mlm", action='store_true', default=True,
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=43,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if args.cfg is not None:
        cfg_from_file(args.cfg)
        cfg.GPU_ID = args.gpu
        args.per_gpu_train_batch_size = cfg.BATCH_SIZE
        args.per_gpu_eval_batch_size = cfg.BATCH_SIZE
        args.gradient_accumulation_steps = cfg.GRAD_ACCUM
        args.learning_rate = cfg.LR
        args.num_train_epochs = cfg.EPOCH
        args.output_dir = cfg.OUTPUT_DIR

    if os.path.exists(cfg.OUTPUT_DIR) and os.listdir(
            cfg.OUTPUT_DIR) and cfg.TRAIN and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                cfg.OUTPUT_DIR))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        if args.no_cuda:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(cfg.GPU_ID))
        if args.multi_gpu is True:
            args.n_gpu = torch.cuda.device_count()
        else:
            args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    cfg.TASK_NAME = cfg.TASK_NAME.lower()
    if cfg.TASK_NAME not in processors:
        raise ValueError("Task not found: %s" % (cfg.TASK_NAME))
    processor = processors[cfg.TASK_NAME]()
    args.output_mode = output_modes[cfg.TASK_NAME]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = LM_MODEL_CLASSES[cfg.TEXT.MODEL_TYPE]
    lm_config = config_class.from_pretrained(cfg.TEXT.MODEL_NAME)
    lm_tokenizer = tokenizer_class.from_pretrained(cfg.TEXT.MODEL_NAME,
                                                   do_lower_case=cfg.TEXT.LOWER_CASE)
    if args.block_size <= 0:
        args.block_size = lm_tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, lm_tokenizer.max_len_single_sentence)

    if cfg.LWF.LAYER_WISE:
        logger.info('Do layer-wise knowledge distillation')
        lm_config.output_hidden_states = True

    lm_model = model_class.from_pretrained(cfg.TEXT.MODEL_NAME,
                                           from_tf=bool('.ckpt' in cfg.TEXT.MODEL_NAME),
                                           config=lm_config)
    lm_model.to(args.device)
    for name, m in lm_model.named_modules():
        for n, p in m.named_parameters():
            p.requires_grad = False

    config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg.LWF.MODEL_TYPE]
    config = config_class.from_pretrained(cfg.LWF.MODEL_NAME,
                                          num_labels=num_labels,
                                          finetuning_task=cfg.TASK_NAME)
    tokenizer = tokenizer_class.from_pretrained(cfg.LWF.MODEL_NAME,
                                                do_lower_case=cfg.LWF.LOWER_CASE)

    if cfg.LWF.LAYER_WISE:
        logger.info('Do layer-wise knowledge distillation')
        config.output_hidden_states = True

    model = model_class.from_pretrained(cfg.LWF.MODEL_NAME,
                                        from_tf=bool('.ckpt' in cfg.LWF.MODEL_NAME),
                                        config=config)

    momentum_encoder = model_class.from_pretrained(cfg.LWF.MODEL_NAME,
                                        from_tf=bool('.ckpt' in cfg.LWF.MODEL_NAME),
                                        config=config)

    if cfg.LWF.FREEZE_POS_EMBS:
        logger.info('Freeze Pos Embeddings')
        model.bert.embeddings.position_embeddings.weight.requires_grad = False

    model.to(args.device)
    momentum_encoder.to(args.device)

    logger.info("Image Encoder: %s", cfg.IMAGE.ENCODER)
    if cfg.IMAGE.ENCODER == 'resnet-50':
        image_encoder = ResNetEncoder(project_size=config.hidden_size)
        momentum_resnet = ResNetEncoder(project_size=config.hidden_size).to(args.device)
    else:
        image_encoder = InceptionV3Encoder(project_size=config.hidden_size)
        momentum_resnet = ResNetEncoder(project_size=config.hidden_size)

    image_encoder.to(args.device)
    # momentum_resnet.to(args.device)
    logger.info("vip type: %s", cfg.SMOOTH.TYPE)
    vip = vip_CLASSES[cfg.SMOOTH.TYPE](config.hidden_size)
    vip.to(args.device)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    tucker = TuckerFusion(128, 128, 128, 128, 5).to(args.device)
    # Training
    if cfg.TRAIN:
        train_dataset, train_dataset_aug = load_and_cache_examples(args, cfg.TASK_NAME, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, train_dataset_aug, model, image_encoder, lm_model, momentum_encoder, momentum_resnet, vip, config, tokenizer, tucker)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    logger.info("Evaluation parameters %s", args)

    # Evaluation
    results = {}
    if cfg.EVAL and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(cfg.OUTPUT_DIR, do_lower_case=cfg.TEXT.LOWER_CASE)
        checkpoints = [cfg.OUTPUT_DIR]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(cfg.OUTPUT_DIR + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            image_encoder.load_state_dict(torch.load(os.path.join(checkpoint, 'image_encoder.bin'),
                                                     map_location=lambda storage, loc: storage.cuda(cfg.GPU_ID)))
            vip.load_state_dict(torch.load(os.path.join(checkpoint, 'vip.bin'),
                                            map_location=lambda storage, loc: storage.cuda(cfg.GPU_ID)))
            result = evaluate(args, model, image_encoder, lm_model, momentum_encoder, momentum_resnet, vip, config, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
