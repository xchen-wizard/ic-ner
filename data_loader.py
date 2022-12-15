from __future__ import annotations

import json
import logging
import os
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Union

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from utils import convert_amazon_tags_to_BIO
from utils import convert_prodigy_tags_to_BIO
from utils import get_intent_labels_dict
from utils import get_slot_labels_dict


logger = logging.getLogger()


class InputExample(NamedTuple):
    """
    input examples class
    Args:
        guid: unique example id
        tokens: list of tokens (space delimited) in a message/text
        intent_label: (optional) intent label id
        slot_labels: (optional) list of slot labels in BIO format

    """
    guid: str
    tokens: list[str]
    intent_label: str | None
    slot_labels: list[str] | None


class InputFeatures(NamedTuple):
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    intent_label_id: int
    slot_labels_ids: list[int] | int


class Processor:
    """
    utility class to convert pd.DataFrame into list of InputExamples
    """

    def __init__(self, args):
        self.args = args
        self.intent_labels_dict = {}
        self.slot_labels_dict = {}

    def _read_file(self, input_file):
        if self.args.file_format == 'json':
            # jsonl
            df = pd.read_json(input_file, lines=True)
        else:
            # try csv
            df = pd.read_csv(input_file)
        return df

    def _create_examples(
        self, texts: list[str], intents: list[str] | list[None],
        slots: list[list[str]] | list[dict] | list[None] = [None], mode: str = 'train',
    ):
        """
        update the intent_labels_dict and slot_labels_dict with key (label), value (id)
        return a list of InputExample objects
        """
        # make sure the slot format is in BIO
        if slots[0] is not None and self.args.slot_label_style == 'prodigy':
            tokens_list, slots = convert_prodigy_tags_to_BIO(
                texts, slots,  # type: ignore
            )
        elif self.args.slot_label_style == 'amazon':
            # parse Amazon style slot annotation
            tokens_list, slots = convert_amazon_tags_to_BIO(texts)
        else:  # bio format or intents only
            tokens_list = [text.strip().split() for text in texts]

        if mode == 'train':
            # update the intent and slot label to id dicts
            self.intent_labels_dict = {
                intent: idx for idx, intent in enumerate(
                    set(intents),
                ) if intent is not None
            }
            slots_flat_list = [
                slot for slot_labels in slots if slot_labels is not None for slot in slot_labels
            ]
            self.slot_labels_dict = {
                slot_label: idx + 1 for idx,  # make space for ignore_index = 0
                slot_label in enumerate(set(slots_flat_list))
            }
            # add a slot label for the cls, sep and pad tokens - only a problem when we use pytorch CRF which doesn't allow -100 as label id
            if self.slot_labels_dict:
                self.slot_labels_dict['PAD'] = self.args.ignore_index

            # cast to json files
            with open(os.path.join(self.args.data_dir, 'intent_labels_dict.json'), 'w') as f:
                json.dump(self.intent_labels_dict, f)
            with open(os.path.join(self.args.data_dir, 'slot_labels_dict.json'), 'w') as f:
                json.dump(self.slot_labels_dict, f)
        else:
            self.intent_labels_dict = get_intent_labels_dict(self.args)
            self.slot_labels_dict = get_slot_labels_dict(self.args)

        # make list of InputExample with text data
        examples = []
        for idx, (tokens, intent, slot_labels) in enumerate(zip(tokens_list, intents, slots)):
            examples.append(
                InputExample(
                    guid=f'{mode}-{idx}',
                    tokens=tokens,
                    intent_label=intent,
                    slot_labels=slot_labels,  # type: ignore
                ),
            )
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode (str): train, validation, test
            file_format (str): json (json lines) by default or csv
        """
        data_path = os.path.join(self.args.data_dir, mode)
        df = self._read_file(data_path)
        if self.args.intent_label_column_name not in df.columns \
                and self.args.slot_label_column_name not in df.columns:
            raise Exception('Need either intent or slot labels in dataframe')
        if self.args.intent_label_column_name not in df.columns:
            logger.warning('no intents in dataset')
            df[self.args.intent_label_column_name] = None
        elif self.args.slot_label_column_name not in df.columns:
            logger.warning('no slot labels in dataset')
            df[self.args.slot_label_column_name] = None
        return self._create_examples(
            texts=df[self.args.text_column_name],
            intents=df[self.args.intent_label_column_name],
            slots=df[self.args.slot_label_column_name],
            mode=mode,
        )


def convert_examples_to_features(
    examples, max_seq_len, tokenizer,
    intent_labels_dict={},
    slot_labels_dict={},
    ignore_index=0,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    # mark the initial wp with the label from the token, if False mark all wps with the label
    align_label_with_initial=True,
):
    """
    convert raw strings into token ids and relevant meta data for model training
    padding!!! be sure you truncate from left!
    Args:
        examples:
        max_seq_len:
        tokenizer:
        intent_labels_dict:
        slot_labels_dict:
        ignore_index:
        cls_token_segment_id:
        pad_token_segment_id:
        sequence_a_segment_id:
        mask_padding_with_zero:
        align_label_with_initial:

    Returns:

    """
    logger.info(f'intent_labels_dict: {intent_labels_dict}')
    logger.info(f'slot_labels_dict: {slot_labels_dict}')
    # Setting based on the current model type
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    unk_token_id = tokenizer.unk_token_id
    pad_token_id = tokenizer.pad_token_id

    max_seq_len = min(tokenizer.model_max_length, max_seq_len)

    features = []
    for example in examples:
        # convert token into wordpieces and align with slot_ids
        if example.slot_labels:
            input_ids = []
            slot_labels_ids = []
            token_type_ids = []
            attention_mask = []
            # add cls token
            input_ids.append(cls_token_id)
            slot_labels_ids.append(ignore_index)
            token_type_ids.append(cls_token_segment_id)
            attention_mask.append(1)
            for token, slot_label in zip(example.tokens, example.slot_labels):
                wps = tokenizer.tokenize(token)
                if not wps:
                    wps = [unk_token_id]
                else:
                    wps = tokenizer.convert_tokens_to_ids(wps)
                input_ids.extend(wps)
                token_type_ids.extend([sequence_a_segment_id] * len(wps))
                attention_mask.extend([1] * len(wps))
                # convert slot label to id: if a slot_label is not in existing slot_labels_dict
                # treat it as an "O"
                slot_label_id = slot_labels_dict.get(
                    slot_label, slot_labels_dict.get('O'),
                )
                # tag the first wp only, ignore_index for the rest
                slot_labels_ids.extend(
                    [slot_label_id] + [ignore_index] * (len(wps) - 1)
                    if align_label_with_initial else [slot_label_id] * len(wps),
                )
            # truncate or pad
            # truncate from right to ensure that user says is included
            # make sure CLS heads the seq and SEP ends the seq
            # ToDo: what do you do if user says exceeds the max_seq_len?
            if len(input_ids) > max_seq_len - 1:
                input_ids = input_ids[1 - max_seq_len:] + [sep_token_id]
                slot_labels_ids = slot_labels_ids[1 - max_seq_len:] \
                    + [ignore_index]
                if input_ids[0] != cls_token_id:
                    # swap the first token with the CLS
                    input_ids[0] = cls_token_id
                    slot_labels_ids[0] = ignore_index
                # sep_token is seq_a_type_id
                token_type_ids = token_type_ids[-max_seq_len:]
                # sep_token is attended to
                attention_mask = attention_mask[-max_seq_len:]
            else:  # else pad
                token_type_ids = token_type_ids + \
                    [sequence_a_segment_id] + [pad_token_segment_id] * \
                    (max_seq_len - len(input_ids) - 1)
                attention_mask = attention_mask + \
                    [1] + [0] * (
                        max_seq_len - len(input_ids) - 1
                    )  # padding not attended to
                input_ids = input_ids + \
                    [sep_token_id] + [pad_token_id] * \
                    (max_seq_len - len(input_ids) - 1)
                slot_labels_ids = slot_labels_ids + \
                    [ignore_index] * (max_seq_len - len(slot_labels_ids))

            assert len(token_type_ids) == len(input_ids) == len(attention_mask) == len(slot_labels_ids),\
                'len mismatch between tokens and token_type_ids/attent_maks/slot_label_ids'

            input_features = {
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'slot_labels_ids': slot_labels_ids,
            }
        else:  # only tokens and intent labels available
            input_features = tokenizer(
                example.tokens, max_length=max_seq_len,
                truncation=True, padding='max_length', is_split_into_words=True,
            )
            input_features.update({
                'slot_labels_ids': ignore_index,
            })
        # get intent id if intent exists
        intent_label_id = intent_labels_dict.get(
            example.intent_label, ignore_index,
        )
        input_features.update({
            'intent_label_id': intent_label_id,
        })

        features.append(InputFeatures(**input_features))
    return features


def load_examples(args, tokenizer, mode):
    processor = Processor(args)

    examples = processor.get_examples(mode)

    features = convert_examples_to_features(
        examples=examples,
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
        ignore_index=args.ignore_index,
        intent_labels_dict=processor.intent_labels_dict,
        slot_labels_dict=processor.slot_labels_dict,
        align_label_with_initial=args.align_label_with_initial,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long,
    )
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long,
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long,
    )
    all_intent_label_ids = torch.tensor(
        [f.intent_label_id for f in features], dtype=torch.long,
    )
    all_slot_labels_ids = torch.tensor(
        [f.slot_labels_ids for f in features], dtype=torch.long,
    )

    dataset = TensorDataset(
        all_input_ids, all_attention_mask,
        all_token_type_ids, all_intent_label_ids, all_slot_labels_ids,
    )

    return dataset
