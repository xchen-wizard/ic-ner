from __future__ import annotations

import json
import logging
import os
import random
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from transformers import AlbertConfig
from transformers import AlbertTokenizer
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import DistilBertConfig
from transformers import DistilBertTokenizer

from models import JointBERT

# from model import JointBERT, JointDistilBERT, JointAlbert
MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    # 'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    # 'albert': (AlbertConfig, JointAlbert, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1',
}


def get_intent_labels_dict(args):
    with open(os.path.join(args.data_dir, 'intent_labels_dict.json')) as f:
        return json.load(f)


def get_slot_labels_dict(args):
    with open(os.path.join(args.data_dir, 'slot_labels_dict.json')) as f:
        return json.load(f)


def init_logger():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )


def convert_prodigy_tags_to_BIO(texts: list[str], prodigy_tags: list[list[dict]]) -> tuple[list[list[str]], list[list[str]]]:
    """
    texts: list of texts
    prodigy_tags: prodigy style NER tags
    {'label': str, 'start': int, 'end': int}

    Returns:
        list of token, tag tuples in BIO format
    """
    list_of_tokens, list_of_tags = [], []
    for text, tag_spans in zip(texts, prodigy_tags):
        if tag_spans:
            tokens, tags = [], []
            # sort tag_span by start idx
            tag_spans = sorted(tag_spans, key=lambda x: x['start'])
            o_start = 0
            for tag_span in tag_spans:
                # tag the substring priro to start
                o_toks, o_tags = tag_tokens(
                    text=text[o_start:tag_span['start']],
                    tag='O',
                )
                bi_toks, bi_tags = tag_tokens(
                    text=text[tag_span['start']: tag_span['end']],
                    tag=tag_span['label'],
                )
                o_start = tag_span['end']  # update the start idx of the O
                tokens.extend(o_toks + bi_toks)
                tags.extend(o_tags + bi_tags)
            if o_start < len(text):  # more Os at end of text
                o_toks, o_tags = tag_tokens(text[o_start:])
                tokens.extend(o_toks)
                tags.extend(o_tags)

        else:
            tokens = text.strip().split()
            tags = ['O'] * len(tokens)

        assert len(tokens) == len(tags), f'{tokens}\n{tags}\nlen not the same!'
        list_of_tokens.append(tokens)
        list_of_tags.append(tags)
    return list_of_tokens, list_of_tags


def tag_tokens(text: str, tag: str = 'O'):
    tokens = text.strip().split()
    if tag == 'O':
        return (tokens, [tag] * len(tokens))
    else:
        # do B- I- on tokens
        tags = ['B-' + tag]
        for i in range(1, len(tokens)):
            tags.append('I-' + tag)
        return (tokens, tags)


def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(MODEL_PATH_MAP[args.model_type])


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    # assert len(intent_preds) == len(intent_labels) == len(
    #     slot_preds,
    # ) == len(slot_labels)
    results = {}
    if intent_labels and intent_preds:
        intent_result = get_intent_acc(intent_preds, intent_labels)
        results.update(intent_result)

    if slot_preds and slot_labels:
        slot_result = get_slot_metrics(slot_preds, slot_labels)
        results.update(slot_result)
    # sementic_result = get_sentence_frame_acc(
    #     intent_preds, intent_labels, slot_preds, slot_labels,
    # )
    # results.update(sementic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        'slot_precision': precision_score(labels, preds),
        'slot_recall': recall_score(labels, preds),
        'slot_f1': f1_score(labels, preds),
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        'intent_acc': acc,
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)
    Returns what percentage of results we got are correct per frame
    """

    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        'sementic_frame_acc': sementic_acc,
    }
