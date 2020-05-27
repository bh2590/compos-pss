CS224U Compisitionality of sentence representations

### Setup
```bash
pip install -r requirements.txt
```


### PSS experiments

For all sentence encoders except BERT we use TFHub and the notebook text_classification_with_tf_hub_2.ipynb contains the steps.

All you need to do is change the values of the keys in param_vals dictionary which can take the following values

```bash
model_name={use_dan,use_large,elmo}
finetune_module={True,False}
last_layer={dnn,linear}
```

Then for BERT install pytorch-pretrained-bert==0.6.2. Then

```bash
cd pytorch-pretrained-BERT
export TASK_NAME=yelp
python run_classifier.py   --task_name $TASK_NAME   --do_train   --do_eval   --do_lower_case --do_comp_eval \
 --data_dir $GLUE_DIR/$TASK_NAME   --bert_model bert-base-uncased   --max_seq_length 128   --train_batch_size 64 \
 --output_dir weights/$TASK_NAME/
```


### TRE experiments

This code builds on the TRE code in https://github.com/jacobandreas/tre which was done for bigram phrases. We extend it to sentences

First enter the project directory

```bash
cd project/
```

Then there are 3 steps
```bash
1. Calculate sentence embeddings for different encoder types (can be at sentence or phrase level however, the main goal of the paper is to extend it to sentences)
2. Calculate the WNS scores for SST
3. Calculate TRE and the rank correlations
```

### To calculate sentence embeddings:
```
python run_encoding.py --dataset_name=<dataset-name> --encoder_name=<encoder-name>
```

### To calculate WNS scores:
```
python run_compositionality_score_gen.py --compositionality_type=node_switching
```

### To calculate TRE scores and rank correlations:
```
python run_tre.py --dataset_name=<dataset-name> --encoder_name=<encoder-name>
```

