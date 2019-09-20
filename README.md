# FasterTransformer_Bert
Using FasterTransformer for accelerating the predict speed of bert and roberta


## Model

1. Modify the original code of [bert-master](https://github.com/google-research/bert):

   * Copy the code of [tensorflow_bert](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer/sample/tensorflow_bert) to [Bert-master](https://github.com/google-research/bert) path.

   * Copy the file of [Faster Transformer](https://github.com/NVIDIA/DeepLearningExamples/tree/master/FasterTransformer) to  [Bert-master](https://github.com/google-research/bert) path.

   * Collection bert-base model **uncased_L-12_H-768_A-12** and test classifier data set **IMDB** by processing to `.tsv` format
        * Link：https://pan.baidu.com/s/1SwSji_B8lCr_IIjkMpheZw
        * Password：jug5

   * Add the code of loding  the train/dev/test data set in `run_classifier.py`：

     ```python
     class ImdbProcessor(DataProcessor):
       """Processor for the IMDB data set."""
     
       def get_train_examples(self, data_dir):
         """See base class."""
         return self._create_examples(
             self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
       def get_dev_examples(self, data_dir):
         """See base class."""
         return self._create_examples(
             self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
       def get_test_examples(self, data_dir):
         """See base class."""
         return self._create_examples(
             self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
       def get_labels(self):
         """See base class."""
         return ["pos", "neg"]
       def _create_examples(self, lines, set_type):
         """Creates examples for the training and dev sets."""
         # get the examples of IMDB data
         examples = []
         for (i, line) in enumerate(lines):
           if set_type == "test":
             continue
           guid = "%s-%s" % (set_type, i)
           if set_type == "test":
             text_a = tokenization.convert_to_unicode(line[1])
             label = "0"
           else:
             text_a = tokenization.convert_to_unicode(line[1])
             label = tokenization.convert_to_unicode(line[0])
           examples.append(
               InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
         return examples
     ```

   * Add the code of counting the time of train/evaluation/predict in  `run_classifier.py`：

     ```python
     if FLAGS.do_train:
         train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
         file_based_convert_examples_to_features(
             train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
         tf.logging.info("***** Running training *****")
         tf.logging.info("  Num examples = %d", len(train_examples))
         tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
         tf.logging.info("  Num steps = %d", num_train_steps)
         train_input_fn = file_based_input_fn_builder(
             input_file=train_file,
             seq_length=FLAGS.max_seq_length,
             is_training=True,
             drop_remainder=True)
         #  counting the time of train
         start = time.time()
         estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
         elapsed = time.time() - start
         print("training finished, time used:{},average {} per sample".format(elapsed, elapsed/len(train_examples)))
     ```

2. Environment requirements:

   * Create environment 、install python packages 、 compiler environment 、 Generate optimized **GEMM** algorithm file:

     ```shell
     # you should change the parameter by yourself
     $ bash requirements.sh
     ```

   * Starting environment

     ```shell
     $ source activate fastertf
     ```

   * Running：

       ```shell
       # you should change the parameter by your path of data and pretrained embedding
       $ bash train_predict.sh 
       ```

3. Optimization principle:

   * In TensorFlow, each basic OP corresponds to a GPU kernel call, and multiple memory reads and writes, which adds a lot of extra overhead. TensorFlow XLA can alleviate this problem to some extent. It will merge some basic OPs to reduce the scheduling and memory reading and writing of the GPU kernel. But in most cases, XLA still can't achieve optimal performance, especially for the computationally intensive case of BERT, any performance improvement will save a lot of computing resources.

   * As we mentioned earlier, OP Fusion can reduce GPU scheduling and memory read and write, which in turn improves performance. For the sake of maximizing performance, inside the Faster Transformer, we merge all the kernels except matrix multiplication as much as possible. The calculation flow of the single-layer Transformer is shown in the following figure:

     ![img](http://ww1.sinaimg.cn/large/006tNc79gy1g61kxcx3y0j30u015mwln.jpg)

   * Others

## Result

1. Parameter setting：

   - max_seq_length：128
   - train_batch_size：16
   - eval_batch_size：16
   - predict_batch_size：16
   - learning_rate：5e-5
   - num_train_epochs：1.0
   - save_checkpoints_steps：100
   - buffer_size = 2000(match your sample size of training data,modify in `run_classifier.py`)

2. Bert Result:

   * Bert_train:

     INFO:tensorflow:***** Running training *****

     INFO:tensorflow:  Num examples = 2000

     INFO:tensorflow:  Batch size = 16

     INFO:tensorflow:  Num steps = 125

     INFO:tensorflow:Loss for final step: 0.6994.

     training finished, time used:57.629658222198486,average 0.028814829111099245 per sample

   * Bert_evaluation：

     evaluation finished, time used:11.677468538284302,average 0.005838734269142151 per sample

     INFO:tensorflow:***** Eval results *****

     INFO:tensorflow:  eval_accuracy = 0.5

     INFO:tensorflow:  eval_loss = 0.69381195

     INFO:tensorflow:  global_step = 375

     INFO:tensorflow:  loss = 0.69381195

   * fastertf_evaluation：

     evaluation finished, time used:5.24286961555481,average 0.0026214348077774046 per sample

     INFO:tensorflow:***** Eval results *****

     INFO:tensorflow:  eval_accuracy = 0.5

     INFO:tensorflow:  eval_loss = 0.69376516

     INFO:tensorflow:  global_step = 375

     INFO:tensorflow:  loss = 0.69367576

   Summary of experimental results ：

   | Task classification  | Sample | Total time |   Time per sample   |
   | :------------------: | :----: | :--------: | :-----------------: |
   |      Bert_train      |  2000  |  57.63 s   |   0.029 s/sample    |
   |   Bert_evaluation    |  2000  |  11.68 s   |   0.0058 s/sample   |
   | Faster TF_evaluation |  2000  | **5.25 s** | **0.0026 s/sample** |

   <font size="2" color="gray">Note: The experimental configuration is 11G Nvidia RTX2080Ti, Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz, 16G RAM, 2T hard disk</font>

3. Roberta Result:

   | Task classification  | Sample | Total time |   Time per sample   |
   | :------------------: | :----: | :--------: | :-----------------: |
   |    Roberta_train     |  2000  |  58.99 s   |   0.029 s/sample    |
   |  Roberta_evaluation  |  2000  |  11.84 s   |   0.0059 s/sample   |
   | Faster TF_evaluation |  2000  | **5.45 s** | **0.0027 s/sample** |

   <font size="2" color="gray">Note: The experimental configuration is 11G Nvidia RTX2080Ti, Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz, 16G RAM, 2T hard disk</font>