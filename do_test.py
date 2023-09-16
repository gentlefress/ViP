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
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
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


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--cfg", default=None, type=str, required=True)
    parser.add_argument("--gpu", default=0, type=int)
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

    # Evaluation
    results = {}
    test_results = {}
    if cfg.EVAL and args.local_rank in [-1, 0]:
        prefix = ""
        test_result = evaluate(args, model, tokenizer, prefix=prefix, set_type="test", epsilon=args.e)
        test_result = dict((k, v) for k, v in test_result.items())
        test_results.update(test_result)

    return results