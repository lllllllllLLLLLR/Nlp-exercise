#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, classification_report
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import TrainingArguments, AutoTokenizer, EvalPrediction, AutoModelForSequenceClassification, Trainer
import sys
import os
import argparse
import torch
import ast
from datasets import load_dataset, ClassLabel, Value, Sequence, load_from_disk
from tqdm import tqdm
import time



parser = argparse.ArgumentParser(description='Finetuning BERT kind models for multi-class classification')
parser.add_argument('--model', type=str, help='Huggingface BERT model to be called', required=True)
parser.add_argument('--train', metavar='csv', type=str, help='Train processed input saved directory', required=True)
parser.add_argument('--test', metavar='csv', type=str, help='Test processed input saved directory', required=True)
parser.add_argument('--valid', metavar='csv', type=str, help='Validation processed input saved directory', required=True)
parser.add_argument('--epochs',  type=int, help='Numbers of epochs (default 5)', default=5)
parser.add_argument('--batch',  type=int, help='batch size (default 8)', default=8)
parser.add_argument('--max_len',  type=int, help='maximum text length (default 128)', default=128)
parser.add_argument('--lr',  type=float, help='learning rate (default 1e-05)', default=1e-05)



args = parser.parse_args()


bert_model = args.model
MAX_LEN = args.max_len
TRAIN_BATCH_SIZE = args.batch
VALID_BATCH_SIZE = 4
EPOCHS = args.epochs
LEARNING_RATE = args.lr

tokenizer = AutoTokenizer.from_pretrained(bert_model)
train_binary = args.train + "_" + bert_model.replace("/","_") + "_" + str(MAX_LEN) + ".bin"
valid_binary = args.valid + "_" + bert_model.replace("/","_") + "_" + str(MAX_LEN) + ".bin"
test_binary = args.test + "_" + bert_model.replace("/","_") + "_" + str(MAX_LEN) + ".bin"




def string_list_2_list(x):
    if type(x["label"][0]) == list:
        x["label"] = list(map(ast.literal_eval,x["label"]))
    return x

def encode(examples):
#    print (examples) 
    r = tokenizer(list(map(str, examples['review'])), padding=True, truncation=True, max_length=MAX_LEN)
    #r = tokenizer(list(map(str, examples['text'])), padding=True, truncation=True, max_length=MAX_LEN)
    return r

def preprocess_labels(examples):
    if type(examples['label'][0]) == list:
        examples['label'] = [[float(x) for x in label] for label in examples['label']]
    else:
        examples['label'] = [float(x) for x in examples['label']]
    return examples

def padding(examples):
    for key in ['input_ids', 'attention_mask']:
        for i, element in enumerate(examples[key]):
            if len(element) < MAX_LEN:
                for i in range(MAX_LEN - len(element)):
                    element.append(0)
            else:
                examples[key][i] = element[:MAX_LEN]
    return examples


def preprocess_dataset(csvfile, binfile, kind='train'):
    l_data = load_dataset('csv', data_files={kind: [csvfile]})
    #data = load_dataset(csvfile, split=kind)
    l_data = l_data.map(string_list_2_list, batched=True)
    l_data = l_data.map(preprocess_labels, batched=True)
    l_data = l_data.map(encode, batched=True)
    new_features = l_data[kind].features.copy()
    print(new_features)
    if type(l_data[kind]["label"][0]) == list:
        new_features["label"] = Sequence(feature=Value(dtype='float'))
    else:
        new_features["label"] = feature=Value(dtype='float')
    print(new_features)
    l_data = l_data[kind].cast(new_features)
    l_data = l_data.map(padding, batched=True)
    l_data.save_to_disk(binfile)



if not os.path.exists(train_binary) :
    preprocess_dataset(args.train, train_binary, kind='train')
if not os.path.exists(valid_binary) :
    preprocess_dataset(args.valid, valid_binary, kind='valid')
if not os.path.exists(test_binary) :
    preprocess_dataset(args.test, test_binary, kind='test')




data_train = load_from_disk(train_binary)
data_test = load_from_disk(test_binary)
data_valid = load_from_disk(valid_binary)

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision_weighted_average = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    recall_average = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_weighted_average = 2.0 * ((precision_weighted_average * recall_average) / ( precision_weighted_average + recall_average ))
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    print(classification_report(y_pred, y_true))
    # return as dictionary
    metrics = {'precision weighted': precision_weighted_average,
               'recall weighted': recall_average,
               'f1 weighted': f1_weighted_average,
               'accuracy': accuracy}
    return metrics

def multi_label_metrics_v2(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    return (classification_report(y_true, y_pred))

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds,labels=p.label_ids)
    return result

def padding(examples):
    for key in ['input_ids', 'attention_mask']:
        for i, element in enumerate(examples[key]):
            if len(element) < 20:
                for i in range(20 - len(element)):
                    element.append(0)
            else:
                examples[key][i] = element[:20]
    return examples

data_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
data_valid.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
data_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

print("TRAIN Dataset: {}".format(data_train.shape))
print("VALID Dataset: {}".format(data_valid.shape))
print("TEST Dataset: {}".format(data_test.shape))

nb_labels = 1
if type(data_test["label"][0]) == list:
    nb_labels = len(data_test["label"][0])
print(data_test["review"][0])
print(data_test["label"][0])

model = AutoModelForSequenceClassification.from_pretrained(bert_model, num_labels=nb_labels)
for name, param in model.named_parameters():
    print(name, param.requires_grad)

training_args = TrainingArguments(
    f"finetune_" + bert_model.replace("/","_") + "_"+str(EPOCHS)+"_"+str(TRAIN_BATCH_SIZE),
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=EPOCHS,
    load_best_model_at_end=True)
print(data_train)
print(data_test)

trainer = Trainer(model=model, args=training_args, train_dataset=data_train, eval_dataset=data_valid, tokenizer=tokenizer, compute_metrics=compute_metrics)
start = time.time()
trainer.train()
end = time.time()
elapsed = end - start

print("************** RESULTS for " + bert_model.replace("/","_") + " *****************")
print("Training time: " + str(elapsed) + " ms")
results_valid = trainer.evaluate(data_valid)
print(results_valid)
results_test = trainer.evaluate(data_test)
print(results_test)
