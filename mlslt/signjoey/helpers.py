# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional, List, Union
import numpy as np
import zipfile
from pathlib import Path

import torch
from torch import nn, Tensor
from torchtext.data import Dataset
import yaml
from signjoey.vocabulary import GlossVocabulary, TextVocabulary


def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        if platform == "linux":
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        logger.info("Hello! This is Joey-NMT.")
        return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    logging_function: Callable[[str], None],
):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param gls_vocab:
    :param txt_vocab:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain {:d},\n\tvalid {:d},\n\ttest {:d}".format(
            len(train_data),
            len(valid_data),
            len(test_data) if test_data is not None else 0,
        )
    )

    logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(vars(train_data[0])["gls"]), " ".join(vars(train_data[0])["txt"])
        )
    )

    logging_function(
        "First 10 words (gls): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )
    logging_function(
        "First 10 words (txt): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(txt_vocab.itos[:10]))
        )
    )

    logging_function("Number of unique glosses (types): {}".format(len(gls_vocab)))
    logging_function("Number of unique words (types): {}".format(len(txt_vocab)))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu")
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save_dir_to_zip(dir_path: Path, zip_path: Path):
    zip = zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED,
    )
    for file in dir_path.rglob("*"):
        save_path = file.relative_to(dir_path)
        zip.write(file, save_path)
    zip.close()


def print_trainable_parameters(model, logger):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
    
    trainable_params = [
        n for (n, p) in model.named_parameters() if p.requires_grad
    ]
    logger.info("Trainable parameters: %s", sorted(trainable_params))
    
    non_trainable_params = [
        n for (n, p) in model.named_parameters() if not p.requires_grad
    ]
    logger.info("Non-trainable parameters: %s", sorted(non_trainable_params))
    

def save_adapters(model, model_dir, adapter_names, adapter_prefix='adapter_'):
    adapters_state_dict = {}

    # Iterate through each encoder layer and save the adapter state dicts
    for adapter_name in adapter_names:
        for layer_idx, layer in enumerate(model.encoder.layers):
            adapter_module = layer.adapter_modules[adapter_name]
            for name, param in adapter_module.named_parameters():
                key = f"encoder.layers.{layer_idx}.adapter_modules.{adapter_name}.{name}"
                adapters_state_dict[key] = param

        save_path = os.path.join(model_dir, f"{adapter_prefix}{adapter_name}.pt")
        torch.save(adapters_state_dict, save_path)


def load_adapters(model, model_dir, adapter_names=["GSG", "CSL", "ASE"], adapter_prefix='adapter_'):
    for adapter_name in adapter_names:
        load_path = os.path.join(model_dir, f"{adapter_prefix}{adapter_name}.pt")
        adapters_state_dict = torch.load(load_path)

        base_model_params = dict(model.named_parameters())
        for name, param in adapters_state_dict.items():
            if name in base_model_params:
                base_model_params[name].data.copy_(param.data)
            else:
                print(f"Parameter {name} not found in model")


# def save_adapters(model, model_dir, adapter_name='ASE', adapter_prefix='adapter_'):
#     adapters_state_dict = {}

#     # Iterate through each encoder layer and save the adapter state dicts
#     for layer_idx, layer in enumerate(model.encoder.layers):
#         adapter_module = layer.adapter_modules[adapter_name]
#         for name, param in adapter_module.named_parameters():
#             key = f"encoder.layers.{layer_idx}.adapter_modules.{adapter_name}.{name}"
#             adapters_state_dict[key] = param.cpu()

#     # Iterate through each decoder layer and save the adapter state dicts
#     for layer_idx, layer in enumerate(model.decoder.layers):
#         adapter_module = layer.adapter_modules[adapter_name]
#         for name, param in adapter_module.named_parameters():
#             key = f"decoder.layers.{layer_idx}.adapter_modules.{adapter_name}.{name}"
#             adapters_state_dict[key] = param.cpu()

#     save_path = os.path.join(model_dir, f"{adapter_prefix}{adapter_name}.pt")
#     torch.save(adapters_state_dict, save_path)


# def load_adapters(model, model_dir, adapter_names=["GSG", "CSL", "ASE"], adapter_prefix='adapter_'):
#     for adapter_name in adapter_names:
#         load_path = os.path.join(model_dir, f"{adapter_prefix}{adapter_name}.pt")
#         adapters_state_dict = torch.load(load_path)

#         base_model_params = dict(model.named_parameters())
#         for name, param in adapters_state_dict.items():
#             if name in base_model_params:
#                 print("Before loading adapter", name, base_model_params[name].data)
#                 base_model_params[name].data.copy_(param.data)
#                 print("After loading adapter", name, base_model_params[name].data)
#             else:
#                 print(f"Parameter {name} not found in model")

def freeze_langs_spec_parameters(model, sign_langs: Union[str, List[str]], logger: Logger) -> None:
    """
    Freeze the parameters of the specified languages in the model,

    :param model: freeze parameters of this model
    :param sign_langs: list of sign languages
    :param logger: logger to log the freezing of parameters
    """
    to_freeze_params = ['encoder.logits', 'decoder.logits', 'langs_gate', 'adapter']
    all_langs_set = set(model.sign_lang_vocab.stoi.keys())
    if sign_langs == 'all':
        sign_langs = all_langs_set
        
    to_freeze_langs = all_langs_set.difference(set(sign_langs))
    to_freeze_langs_ids = [model.sign_lang_vocab.stoi[lang] for lang in to_freeze_langs]
    logger.info(f"Freezing parameters for languages: {to_freeze_langs}, ids: {to_freeze_langs_ids}")
    
    for name, param in model.named_parameters():
        for to_freeze_param in to_freeze_params:
            if any([f"{to_freeze_param}.{lang}" in name for lang in to_freeze_langs_ids]):
                param.requires_grad = False
           