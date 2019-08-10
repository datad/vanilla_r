import os
import argparse
import logging

import torch
import torchtext


from trainer import SupervisedTrainer
from loss import Perplexity, NLLLoss
from models import EncoderRNN, DecoderRNN, Seq2seq, TopKDecoder
from optim import Optimizer
from dataset import SourceField, TargetField

import pandas as pd
import math
import random
import numpy as np
import time

###for tensorboard
import torchvision.utils as vutils
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter


################
# settings
################
print("EXPERIMENT 14")
writer = SummaryWriter(logdir='runs/exp14', comment='experiment_14')
expt_dir='./expt_dir_14'
resume=False
batch_size = 32
#load_checkpoint = ? #'The name of the checkpoint to load, usually an encoded time string'
resume = False #`'Indicates if training has to be resumed from the latest checkpoint'`
log_level = 'info'
loggingFileName = 'app14.log'
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(filename=loggingFileName, filemode='w', format=LOG_FORMAT, level=log_level.upper())

#experiment 12
#load summaries

df_train.to_csv('data/save_train_data.csv', index=None)
df_vali.to_csv('data/save_vali_data.csv', index=None)
df_test.to_csv('data/save_test_data.csv', index=None)

train_path = 'save_train_data.csv'
dev_path = 'save_vali_data.csv'
test_path = 'save_test_data.csv'


#################
# Prepare dataset

start_time = time.time()

src = SourceField()
tgt = TargetField()
max_len = 2300 #DD

def len_filter(data):
    return len(data.src) <= max_len and len(data.tgt) <= max_len

train, dev, test = torchtext.data.TabularDataset.splits(
    path='./', train=train_path,
    validation = dev_path, test = test_path, format='csv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter )

src.build_vocab(train, max_size=50000)
tgt.build_vocab(train, max_size=50000)
input_vocab = src.vocab
output_vocab = tgt.vocab

# Prepare loss
weight = torch.ones(len(tgt.vocab))
pad = tgt.vocab.stoi[tgt.pad_token]

loss = NLLLoss(weight, pad)
if torch.cuda.is_available():
    loss.cuda()
else:
    print("*********** no cuda **************")

seq2seq_m = None

# Initialize model
hidden_size = 512
bidirectional = True
num_epochs = 500

# Initialize models
encoder = EncoderRNN(len(input_vocab), max_len, hidden_size,
                     bidirectional=True, rnn_cell='gru', variable_lengths=True)
#attention hidden_size = hidden_size
#KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
decoder = DecoderRNN(len(output_vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                     dropout_p=0.5, use_attention=True, bidirectional=bidirectional,
                     eos_id=tgt.eos_id, sos_id=tgt.sos_id)

seq2seq_m = Seq2seq(encoder, decoder)
if torch.cuda.is_available():
    seq2seq_m.cuda()


#initialize random tensor
for param in seq2seq_m.parameters():
    param.data.uniform_(-0.08, 0.08)


t = SupervisedTrainer(loss=loss, batch_size=batch_size,
                      checkpoint_every = 50,
                      print_every=10, expt_dir=expt_dir)

optimizer = Optimizer( torch.optim.Adam(seq2seq_m.parameters(), lr=0.001,
                        betas=(0.9, 0.999)) )
# scheduler = StepLR(optimizer.optimizer, 1)
# optimizer.set_scheduler(scheduler)


################################
seq2seq_m = t.train(seq2seq_m, train,
                  num_epochs=num_epochs, dev_data=dev,
                  optimizer=optimizer,
                  teacher_forcing_ratio=0.5,
                  resume=resume)

e = int(time.time() - start_time)
print('ELAPSED TIME TRAINING ~> {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))

#save the model
import pickle
filen = open('input_vocab.obj','wb')
pickle.dump(input_vocab, filen)

fileo = open('output_vocab.obj','wb')
pickle.dump(output_vocab, fileo)

torch.save(seq2seq_m, 'model.pt')

fileo = open('loss.obj','wb')
pickle.dump(loss, fileo)


#####################################

# export scalar data to JSON for external processing
#writer = logging.summary()  write logger some how

#write embedding for tensorboard
writer.add_embedding(seq2seq_m.decoder.embedding.weight, metadata = output_vocab.itos)

#save weights for tensorboard
for name, param in seq2seq_m.named_parameters():
    writer.add_histogram(name, param.clone().cpu().data.numpy(), num_epochs)

#get accuracy and loss for plot
import re
rawFile=[]
with open(loggingFileName, "r") as f:
  for line in f:
    rawFile.append(line.strip())

accuracy = list(filter(lambda x: re.search("^.*Accuracy: ",x), rawFile))
accuracyValues = list(re.sub("^.*Accuracy: ","", x) for x in accuracy)

#write accuarcy for tensorboard
for i in range(num_epochs):
  writer.add_scalar('data/accuracy', float(accuracyValues[i]), i)

### loss
# NLLLoss
loss_n = list(filter(lambda x: re.search("^.*Perplexity: ",x), rawFile))
loss_m = list(re.sub("^.*Perplexity: ","", x) for x in loss_n)
lossValues = list(re.sub("[,].*$","", x) for x in loss_m)

#write loss for tensorboard
for i in range(num_epochs):
  writer.add_scalar('data/loss', float(lossValues[i]), i)

writer.export_scalars_to_json("./all_scalars.json")
writer.close()

e = int(time.time() - start_time)
print('ELAPSED TIME ~> {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
