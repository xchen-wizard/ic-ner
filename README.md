# ic-ner
Intent Classification and NER jointly trained model based on transformers
## How to run
```
python main.py --text_column_name text \
--intent_label_column_name label \
--data_dir ./data \
--file_format csv \
--slot_label_style amazon \
--model_type bert \
--max_seq_len 200 \
--do_train --task joint \
--model_dir ./temp \
--add_speaker_tokens


```
