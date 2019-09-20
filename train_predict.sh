#!/bin/bash
# 设置bert/roberta模型、分类数据的环境变量
export BERT_BASE_DIR='/home/jovenchu/text_classifier/faster_transformer/uncased_L-12_H-768_A-12_roberta'
export IMDB_DIR='/home/jovenchu/text_classifier/faster_transformer/data'

# 训练分类模型
cd bert/
python run_classifier.py   --task_name=Imdb   --do_train=true  --data_dir=$IMDB_DIR   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --max_seq_length=128   --eval_batch_size=16   --output_dir=../imdb_output

# 模型的F32转F16格式，需要更改模型的路径
cd ../
export MODEL='imdb_output/model.ckpt-125'
python ckpt_type_convert.py --init_checkpoint=$MODEL --fp16_checkpoint=imdb_output/fp16_model.ckpt

# 开始fastertf加速后的预测
python run_classifier_fastertf.py   --task_name=Imdb   --do_eval=true   --data_dir=$IMDB_DIR   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=imdb_output/fp16_model.ckpt   --max_seq_length=128   --eval_batch_size=16   --output_dir=imdb_output   --floatx=float16