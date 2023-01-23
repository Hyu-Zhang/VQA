#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import logging
import os
import sys
import csv
import numpy as np
import torch
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging
from omegaconf import DictConfig

from utils import checkpoint_utils
from utils.eval_utils import eval_step, merge_results
from utils.zero_shot_utils import zero_shot_step

from tqdm import tqdm
from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import json

from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def main(cfg: DictConfig, **kwargs):
    utils.import_user_module(cfg.common)

    reset_logging()
    logger.info(cfg)

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Load ensemble
    overrides = eval(cfg.common_eval.model_overrides)
    # Deal with beam-search / all-candidate VQA eval
    if cfg.task._name == "vqa_gen":
        overrides['val_inference_type'] = "beamsearch" if kwargs['beam_search_vqa_eval'] else "allcand"

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    if kwargs["zero_shot"]:
        task = tasks.setup_task(cfg.task)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
    else:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Move models to GPU
    for model, ckpt_path in zip(models, utils.split_paths(cfg.common_eval.path)):
        if kwargs['ema_eval']:
            logger.info("loading EMA weights from {}".format(ckpt_path))
            model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm"),
    )

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    results = []
    score_sum = torch.FloatTensor([0]).cuda()
    score_cnt = torch.FloatTensor([0]).cuda()
    for sample in progress:
        if "net_input" not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if cfg.common.fp16 else sample
        with torch.no_grad():
            if kwargs["zero_shot"]:
                result, scores = zero_shot_step(task, generator, models, sample)
            else:
                result, scores = eval_step(task, generator, models, sample, **kwargs)
        results += result

        score_sum += sum(scores) if scores is not None else 0
        score_cnt += len(scores) if scores is not None else 0
        progress.log({"sentences": sample["nsentences"]})

    merge_results(task, cfg, logger, score_cnt, score_sum, results)


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--ema-eval", action='store_true', help="Use EMA weights to make evaluation.")
    parser.add_argument("--beam-search-vqa-eval", action='store_true', help="Use beam search for vqa evaluation (faster inference speed but sub-optimal result), if not specified, we compute scores for each answer in the candidate set, which is slower but can obtain best result.")
    parser.add_argument("--zero-shot", action='store_true')
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(
        cfg, main, ema_eval=args.ema_eval, beam_search_vqa_eval=args.beam_search_vqa_eval, zero_shot=args.zero_shot
    )


if __name__ == "__main__":
    # pre-processing
    print("pre-processing the input files of VQA task")
    file_dir = '/mnt/data/test.csv'
    img_dir = '/mnt/data/imgs/'

    img2id = {}
    id = 0
    count = 0

    new_file = './results/vqa_input_test.tsv'

    with open(new_file, 'w', encoding='utf-8') as writer:
        csv_file = pd.read_csv(file_dir)
        progress = tqdm(csv_file.iterrows(), total=len(csv_file))
        for i, row in progress:
            img_name, question = row['image'], row['question']
            img = Image.open(os.path.join(img_dir,img_name)) # path to file
            img_buffer = BytesIO()
            img.save(img_buffer, format=img.format)
            byte_data = img_buffer.getvalue()
            base64_str = base64.b64encode(byte_data) # bytes
            base64_str = base64_str.decode("utf-8") # str
            # print(base64_str)
            if img_name not in img2id:
                img2id[img_name] = id
                id += 1
            count += 1
            new_line = str(count) + '\t' + str(id) + '\t' + question  + '\t' + '1.0|!+space' + '\t' + 'space' + '\t' + base64_str + '\n'
            writer.write(new_line)
    # generate text answers
    print("generate the text answers of VQA task")
    cli_main()
    # pre-processing
    print("pre-processing the input files of VG task")
    data = json.load(open('./results/vqa_test_beam/test_predict.json'))
    qa_dict = {}
    for item in data:
        qa_dict[item["question_id"]] = item['answer']

    file_dir = '/mnt/data/test.csv'
    new_file = './results/ref_input_test.csv'

    with open(new_file, 'w', encoding='utf-8') as f:
        csv_file = pd.read_csv(file_dir)
        progress = tqdm(csv_file.iterrows(), total=len(csv_file))
        head_line = ['image', 'question', 'answer']
        writer = csv.writer(f)
        writer.writerow(head_line)
        for i, row in progress:
            img_name, question = row['image'], row['question']
            new_line = [img_name, question, qa_dict[i+1]]
            writer.writerow(new_line)
    # generate object coordinate
    print("generate the object coordinate answers of VG task")

    # Register refcoco task
    tasks.register_task('refcoco', RefcocoTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False

    """## **Build Model**
    Below you can build your model and load the weights from the given checkpoint, and also build a generator. 
    """

    # Load pretrained ckpt & config
    overrides={"bpe_dir":"utils/BPE"}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths('./checkpoints/ref_large_best.pt'),
            arg_overrides=overrides
        )

    cfg.common.seed = 7
    cfg.generation.beam = 5
    cfg.generation.min_len = 4
    cfg.generation.max_len_a = 0
    cfg.generation.max_len_b = 4
    cfg.generation.no_repeat_ngram_size = 3

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    """## **Preprocess**
    We demonstrate the required transformation fucntions for preprocessing inputs.
    """

    # Image transform
    from torchvision import transforms
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()
    def encode_text(text, length=None, append_bos=False, append_eos=False):
        s = task.tgt_dict.encode_line(
            line=task.bpe.encode(text.lower()),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s

    # Construct input for refcoco task
    patch_image_size = cfg.task.patch_image_size
    def construct_sample(image: Image, text: str):
        w, h = image.size
        w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
        h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
        patch_image = patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True, append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
        sample = {
            "id":np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask,
            },
            "w_resize_ratios": w_resize_ratio,
            "h_resize_ratios": h_resize_ratio,
            "region_coords": torch.randn(1, 4)
        }
        return sample
    
    # Function to turn FP32 to FP16
    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    """## **Run Inference**
    Download an image and run the following scripts to generate the result.
    """

    # Download an image from COCO or you can use other images with wget
    import os
    from tqdm.auto import tqdm
    import pandas as pd

    img_dir = '/mnt/data/imgs/'
    images = os.listdir(img_dir)
    result_list = []
    csv_file = pd.read_csv('./results/ref_input_test.csv')
    progress = tqdm(csv_file.iterrows(), total=len(csv_file))
    for i, row in progress:
        img_url = row['image']
        img_name = img_url.strip().split('/')[-1]
        assert img_name in images
        image = Image.open(os.path.join(img_dir, img_name))
        question = row['question']
        temp = row['answer']
        answer = question + ' ' + temp

        # Construct input sample & preprocess for GPU if cuda available
        sample = construct_sample(image, answer)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

        # Run eval step for refcoco
        with torch.no_grad():
            result, scores = eval_step(task, generator, models, sample)
        result_list.append([img_url, int(result[0]["box"][0]), int(result[0]["box"][1]), int(result[0]["box"][2]), int(result[0]["box"][3])])

    import csv
    file = open('./results/temp.csv', 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(['image', 'left', 'top', 'right', 'bottom'])
    for item in result_list:
        csv_writer.writerow([item[0], item[1], item[2], item[3], item[4]])
    file.close()
    # post-processing
    print("post-processing")
    from tqdm.auto import tqdm
    new_file = '/mnt/output/answer.csv'
    with open(new_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        csv_file = pd.read_csv('./results/temp.csv')
        progress = tqdm(csv_file.iterrows(), total=len(csv_file))
        head_line = ['image', 'left', 'top', 'right', 'bottom']
        writer.writerow(head_line)
        for i, row in progress:
            img_url = row['image']

            left, right = row['left'], row['right']
            if row['left'] >= row['right']:
                left = row['right']
                right = row['left']

            top, bottom = row['top'], row['bottom']
            if row['top'] >= row['bottom']:
                top = row['bottom']
                bottom = row['top']
            new_line = [img_url, left, top, right, bottom]
            writer.writerow(new_line)
    print("generate successfully")

