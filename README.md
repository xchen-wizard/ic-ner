# ic-ner
Intent Classification and NER jointly trained model based on transformers
## How to install & run
1. Install dependencies
```commandline
poetry install
```
2. Put train, validation, and test data in a single dir.
Dir structure:
   - data
     - train
     - validation
     - test
3. Example run with train, validation and test data
```
poetry run python main.py --text_column_name text \
--intent_label_column_name label \
--data_dir ./data \
--file_format csv \
--slot_label_style amazon \
--model_type bert \
--max_seq_len 200 \
--do_train \
--do_eval \
--task joint \
--model_dir ./temp \
--add_speaker_tokens \
--save_preds
```
