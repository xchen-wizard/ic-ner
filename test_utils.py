from __future__ import annotations

import logging

from utils import convert_amazon_tags_to_BIO
from utils import convert_prodigy_tags_to_BIO

logger = logging.getLogger()


def test_convert_prodigy_tags_to_BIO():
    texts = [
        'friends Hippo in Chinos Green YES',  # end O
        "So I can't order one dinkum doll and one Designs swaddle citron Rose Stone Sea colors Rose",
        ' no  product mention in this string',
    ]  # noqa: E128

    prodigy_tags = [
        [{'label': 'product', 'start': 0, 'end': 29}],
        [
            {'label': 'product', 'start': 21, 'end': 32},
            {'label': 'product', 'start': 41, 'end': 90},
        ],
        [],
    ]

    list_of_tokens, list_of_tags = convert_prodigy_tags_to_BIO(
        texts, prodigy_tags,
    )
    logger.info(list_of_tokens)
    logger.info(list_of_tags)


def test_convert_amazon_tags_to_BIO():
    texts = ['{SALTED SALTED:Product}, {STRAWBERRY:Product}, {CHOCOLATE:Product}, {CHOCOLATE:Product}, {MANGO 24:Product} and {VEGAN MANGO:Product}']
    list_of_tokens, list_of_tags = convert_amazon_tags_to_BIO(texts)
    logger.info(list_of_tokens)
    logger.info(list_of_tags)
