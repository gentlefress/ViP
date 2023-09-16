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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, SequentialSampler)
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm

from transformers import (BertConfig, BertTokenizer)
from transformers import BertModel
# from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
# from transformers import DebertaConfig, DebertaTokenizer, DebertaModel
# from transformers import XLNetModel, XLNetConfig, XLNetTokenizer
from transformers import AutoModel, AutoConfig, AutoTokenizer, LxmertModel
from utils.unsupervised_config import cfg, cfg_from_file
from utils.data import save_h5, HDF5Dataset
from utils.finetune_data_bunch import compute_metrics
from utils.unsupervised_data_bunch import convert_examples_to_features, output_modes, processors
from transformers.data.metrics import acc_and_f1

logger = logging.getLogger(__name__)

# ALL_MODELS = sum((tuple(conf.BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()) for conf in (BertConfig,)), ())
MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, tokenizer, prefix="", set_type="dev", epsilon=0.0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if cfg.TASK_NAME in ("mnli", "mini-mnli") else (cfg.TASK_NAME,)
    # eval_task_names = ["mini-mnli"]
    # args.output_mode = 'classification'

    results = {}
    random_results = {}
    for eval_task in eval_task_names:
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, set_type=set_type)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        preds = None
        out_label_ids = None
        output_list = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1], }

                if cfg.TEXT.MODEL_TYPE != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if cfg.TEXT.MODEL_TYPE in ['bert',
                                                                                   'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = model(**inputs)
                # output = outputs.last_hidden_state[:, 0, :]
                # print(outputs[0].shape)
                # print(outputs.hidden_states)
                output = outputs[1]  # text features
                #

                inputs = {'input_ids': batch[3],
                          'attention_mask': batch[4], }

                if cfg.TEXT.MODEL_TYPE != 'distilbert':
                    inputs['token_type_ids'] = batch[5] if cfg.TEXT.MODEL_TYPE in ['bert',
                                                                                   'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = model(**inputs)
                # output2 = outputs.last_hidden_state[:, 0, :]

                output2 = outputs[1]
                cosine_similarity = F.cosine_similarity(output, output2, dim=-1)
                output = F.normalize(output, dim=-1)
                label = batch[-1].float()
                label = F.normalize(label, dim=-1)
                # cosine_similarity = F.cosine_similarity(output[:, output.size(-1)//2:], output2[:, output2.size(-1)//2:], dim=-1)
            if preds is None:
                preds = cosine_similarity.detach().cpu().numpy()
                out_label_ids = label.detach().cpu().numpy()
                output_list = output.detach().cpu().numpy()
            else:
                preds = np.append(preds, cosine_similarity.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, label.detach().cpu().numpy(), axis=0)
                output_list = np.append(output_list, output.detach().cpu().numpy(), axis=0)
        if args.output_mode == "classification":
            preds = np.squeeze(preds)
            preds_label_ids = np.where(preds > epsilon, 1, 0)
            np.save(os.path.join(cfg.OUTPUT_DIR, 'preds.npy'), preds)
            np.save(os.path.join(cfg.OUTPUT_DIR, 'out_label_ids.npy'), out_label_ids)
            np.save(os.path.join(cfg.OUTPUT_DIR, 'output.npy'), output_list)

            two_classes_out_label_ids = np.where(out_label_ids == 1, 1, 0)
            print(preds[:100])
            print(two_classes_out_label_ids[:100])

            result = acc_and_f1(preds_label_ids, two_classes_out_label_ids)
            results.update(result)

            random_preds = np.random.choice(np.array(2), size=preds.shape, p=[2/3, 1/3])
            print(random_preds[:100])
            print(two_classes_out_label_ids[:100])
            random_result = acc_and_f1(random_preds, two_classes_out_label_ids)
            random_results.update(random_result)

        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
            result = compute_metrics(eval_task, preds, out_label_ids)
            results.update(result)

        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        logger.info("***** Random results {} *****".format(prefix))
        for key in sorted(random_results.keys()):
            logger.info("  %s = %s", key, str(random_results[key]))

    return results


def load_and_cache_examples(args, task, tokenizer, set_type='train', save_interval=10000):
    if args.local_rank not in [-1, 0] and set_type == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(cfg.DATA_DIR, 'unsupervised_cached_{}_{}_{}_{}'.format(set_type,
                                                                                               list(filter(None,
                                                                                                           cfg.TEXT.MODEL_NAME.split(
                                                                                                               '/'))).pop(),
                                                                                               str(cfg.TEXT.MAX_LEN),
                                                                                               str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", cfg.DATA_DIR)
        if os.path.isfile(cached_features_file):
            logger.info("Deleting existed cached file %s", cached_features_file)
            os.remove(cached_features_file)

        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and cfg.TEXT.MODEL_TYPE in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if set_type == 'train':
            examples = processor.get_train_examples(cfg.DATA_DIR)
        elif set_type == 'dev':
            examples = processor.get_dev_examples(cfg.DATA_DIR)
        else:
            examples = processor.get_test_examples(cfg.DATA_DIR)
        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=cfg.TEXT.MAX_LEN,
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
            all_attention_mask = []
            all_token_type_ids = []

            all_input_ids2 = []
            all_attention_mask2 = []
            all_token_type_ids2 = []

            all_labels = []
            for (f_idx, f) in enumerate(features):
                all_input_ids.append(f.input_ids)
                all_attention_mask.append(f.attention_mask)
                all_token_type_ids.append(f.token_type_ids)

                all_input_ids2.append(f.input_ids2)
                all_attention_mask2.append(f.attention_mask2)
                all_token_type_ids2.append(f.token_type_ids2)

                all_labels.append(f.label)

                if len(all_input_ids) == save_interval:
                    logger.info("Saving example %d" % (f_idx + 1))
                    save_h5(cached_features_file, np.array(all_input_ids, dtype=np.long), 'input_ids')
                    save_h5(cached_features_file, np.array(all_attention_mask, dtype=np.long), 'attention_mask')
                    save_h5(cached_features_file, np.array(all_token_type_ids, dtype=np.long), 'token_type_ids')

                    save_h5(cached_features_file, np.array(all_input_ids2, dtype=np.long), 'input_ids2')
                    save_h5(cached_features_file, np.array(all_attention_mask2, dtype=np.long), 'attention_mask2')
                    save_h5(cached_features_file, np.array(all_token_type_ids2, dtype=np.long), 'token_type_ids2')

                    if output_mode == "classification":
                        save_h5(cached_features_file, np.array(all_labels, dtype=np.long), "labels")
                    elif output_mode == "regression":
                        save_h5(cached_features_file, np.array(all_labels, dtype=np.float), "labels")
                    all_input_ids = []
                    all_attention_mask = []
                    all_token_type_ids = []
                    all_input_ids2 = []
                    all_attention_mask2 = []
                    all_token_type_ids2 = []
                    all_labels = []

            if len(all_input_ids) != 0:
                save_h5(cached_features_file, np.array(all_input_ids, dtype=np.long), 'input_ids')
                save_h5(cached_features_file, np.array(all_attention_mask, dtype=np.long), 'attention_mask')
                save_h5(cached_features_file, np.array(all_token_type_ids, dtype=np.long), 'token_type_ids')

                save_h5(cached_features_file, np.array(all_input_ids2, dtype=np.long), 'input_ids2')
                save_h5(cached_features_file, np.array(all_attention_mask2, dtype=np.long), 'attention_mask2')
                save_h5(cached_features_file, np.array(all_token_type_ids2, dtype=np.long), 'token_type_ids2')

                if output_mode == "classification":
                    save_h5(cached_features_file, np.array(all_labels, dtype=np.long), "labels")
                elif output_mode == "regression":
                    save_h5(cached_features_file, np.array(all_labels, dtype=np.float), "labels")

    if args.local_rank == 0 and set_type == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    dataset = HDF5Dataset(cached_features_file, ['input_ids', 'attention_mask', 'token_type_ids',
                                                 'input_ids2', 'attention_mask2', 'token_type_ids2',
                                                 'labels'])

    # dataset = HDF5Dataset(cached_features_file, ['input_ids', 'attention_mask', 'token_type_ids',
    #                                              'labels'])
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--cfg", default=None, type=str, required=True)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--multi_gpu", default=False, action='store_true')

    ## Few shot parameters
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("-e", type=float, default=0.0)

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
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
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
        args.per_gpu_eval_batch_size = cfg.BATCH_SIZE * 4
        args.gradient_accumulation_steps = cfg.GRAD_ACCUM
        args.learning_rate = cfg.LR
        args.num_train_epochs = cfg.EPOCH

    if os.path.exists(cfg.OUTPUT_DIR) and os.listdir(
            cfg.OUTPUT_DIR) and cfg.TRAIN and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                cfg.OUTPUT_DIR))
    else:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

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

    # Prepare Few Shot settings
    if args.num_samples > 0:
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + '-' + str(args.num_samples)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    cfg.TEXT.MODEL_TYPE = cfg.TEXT.MODEL_TYPE.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[cfg.TEXT.MODEL_TYPE]
    config = config_class.from_pretrained(cfg.TEXT.MODEL_NAME,
                                          num_labels=num_labels,
                                          finetuning_task=cfg.TASK_NAME)
    tokenizer = tokenizer_class.from_pretrained(cfg.TEXT.MODEL_NAME,
                                                do_lower_case=cfg.TEXT.LOWER_CASE)
    model = model_class.from_pretrained(cfg.TEXT.MODEL_NAME,
                                        from_tf=bool('.ckpt' in cfg.TEXT.MODEL_NAME),
                                        config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    args.output_mode = "classification"
    # Evaluation
    print(args)
    results = {}
    test_results = {}
    if cfg.EVAL and args.local_rank in [-1, 0]:
        prefix = ""
        test_result = evaluate(args, model, tokenizer, prefix=prefix, set_type="test", epsilon=args.e)
        test_result = dict((k, v) for k, v in test_result.items())
        test_results.update(test_result)

    return results


if __name__ == "__main__":
    main()
