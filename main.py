from __future__ import annotations

import argparse

from data_loader import load_examples
from trainer import Trainer
from utils import init_logger
from utils import load_tokenizer
from utils import set_seed


def main(args):
    init_logger()
    if args.seed:
        set_seed(args)
    tokenizer = load_tokenizer(args)

    # add special token speaker tags into the vocab
    if args.add_speaker_tokens:
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[AGENT]', '[USER]']},
        )
    # reminder: model needs to resize embedding size too!

    train_dataset = load_examples(args, tokenizer, mode='train')
    val_dataset = load_examples(args, tokenizer, mode='validation')
    test_dataset = load_examples(args, tokenizer, mode='test')

    trainer = Trainer(
        args, train_dataset, val_dataset,
        test_dataset, len(tokenizer),
    )

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate('test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--text_column_name', default='text',
        type=str, help='name of the text column in dataframe',
    )
    parser.add_argument(
        '--intent_label_column_name', default='intent',
        type=str, help='name of the intent column in dataframe',
    )
    parser.add_argument(
        '--slot_label_column_name', default='slot_labels',
        type=str, help='name of the slot label column in dataframe',
    )
    parser.add_argument(
        '--data_dir', required=True, type=str,
        help='path to the train/va/test data files',
    )
    parser.add_argument(
        '--file_format', default='json', type=str,
        choices=['json', 'csv'], help='data format, json lines by default or csv',
    )
    parser.add_argument(
        '--slot_label_style', default='prodigy', type=str,
        choices=['prodigy', 'bio', 'amazon'], help='slot label style: prodigy or bio or amazon',
    )
    parser.add_argument(
        '--model_type', required=True, type=str,
        help='the type of transformer model to fine-tune',
    )
    parser.add_argument(
        '--seed', type=int, default=123,
        help='you may choose to seed the model training',
    )
    parser.add_argument(
        '--max_seq_len', type=int, default=512,
        help='the max seq len in terms of wordpieces',
    )
    parser.add_argument(
        '--train_batch_size', default=32,
        type=int, help='Batch size for training.',
    )
    parser.add_argument(
        '--eval_batch_size', default=64,
        type=int, help='Batch size for evaluation.',
    )
    parser.add_argument(
        '--learning_rate', default=5e-5,
        type=float, help='The initial learning rate for Adam.',
    )
    parser.add_argument(
        '--num_train_epochs', default=10.0, type=float,
        help='Total number of training epochs to perform.',
    )
    parser.add_argument(
        '--weight_decay', default=0.0,
        type=float, help='Weight decay if we apply some.',
    )
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    parser.add_argument(
        '--adam_epsilon', default=1e-8,
        type=float, help='Epsilon for Adam optimizer.',
    )
    parser.add_argument(
        '--max_grad_norm', default=1.0,
        type=float, help='Max gradient norm.',
    )
    parser.add_argument(
        '--max_steps', default=-1, type=int,
        help='If > 0: set total number of training steps to perform. Override num_train_epochs.',
    )
    parser.add_argument(
        '--warmup_steps', default=0, type=int,
        help='Linear warmup over warmup_steps.',
    )
    parser.add_argument(
        '--dropout_rate', default=0.1,
        type=float, help='Dropout for fully-connected layers',
    )

    parser.add_argument(
        '--logging_steps', type=int,
        default=200, help='Log every X updates steps.',
    )
    parser.add_argument(
        '--save_steps', type=int, default=200,
        help='Save checkpoint every X updates steps.',
    )

    parser.add_argument(
        '--do_train', action='store_true',
        help='Whether to run training.',
    )
    parser.add_argument(
        '--do_eval', action='store_true',
        help='Whether to run eval on the test set.',
    )
    parser.add_argument(
        '--no_cuda', action='store_true',
        help='Avoid using CUDA when available',
    )

    parser.add_argument(
        '--ignore_index', default=0, type=int,
        help='Specifies a target value that is ignored and does not contribute to the input gradient',
    )

    parser.add_argument(
        '--slot_loss_coef', type=float,
        default=1.0, help='Coefficient for the slot loss.',
    )
    parser.add_argument(
        '--intent_loss_coef', type=float,
        default=1.0, help='Coefficient for the intent loss.',
    )
    # CRF option
    parser.add_argument(
        '--use_crf', action='store_true',
        help='Whether to use CRF',
    )

    parser.add_argument(
        '--early_stopping', action='store_true',
        help='Whether to do early stopping',
    )
    parser.add_argument(
        '--task', type=str, default='joint',
        choices=['joint', 'intent', 'slot'], help='select the task to train for',
    )
    parser.add_argument(
        '--model_dir', type=str, default='./',
        help='dir to save checkpoints',
    )
    parser.add_argument(
        '--align_label_with_initial', action='store_true',
        help='True to align the label with the first wp in the token, else to use the same label for all wps in token',
    )
    parser.add_argument(
        '--add_speaker_tokens', action='store_true',
        help='whether to add speaker tokens to the special tokens vocab',
    )
    parser.add_argument(
        '--save_preds', action='store_true',
        help='whether to save preds for test data',
    )

    args = parser.parse_args()
    main(args)
