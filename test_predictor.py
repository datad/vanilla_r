import os
import argparse
import logging

import torch
import torchtext

import seq2seq
from trainer import SupervisedTrainer
from evaluator import Predictor, Evaluator
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq, TopKDecoder
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.util.checkpoint import Checkpoint

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

import pickle
filei = open('input_vocab.obj', 'rb')
input_vocab = pickle.load(filei)

fileo = open('output_vocab.obj', 'rb')
output_vocab = pickle.load(fileo)

fileo = open('loss.obj', 'rb')
loss = pickle.load(fileo)

seq2seq_m = torch.load('model.pt')

################################
# testing
################################
train_path = 'save_train_data.csv'
dev_path = 'save_vali_data.csv'
test_path = 'save_test_data.csv'


src = SourceField()
tgt = TargetField()
max_len = 2300
batch_size = 32

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

evaluator = Evaluator(loss=loss, batch_size=batch_size)

loss1, accuracy1 = evaluator.evaluate(seq2seq_m, train)
print(" training set ")
print("loss: ", loss1)
print("accuracy: ", accuracy1)


loss1, accuracy1 = evaluator.evaluate(seq2seq_m, test)
print(" testing ")
print("loss: ", loss1)
print("accuracy: ", accuracy1)


loss1, accuracy1 = evaluator.evaluate(seq2seq_m, dev)
print(" evaluation set ")
print("loss: ", loss1)
print("accuracy: ", accuracy1)


#############


beam_search = Seq2seq(seq2seq_m.encoder, TopKDecoder(seq2seq_m.decoder, 3))
if torch.cuda.is_available():
    beam_search.cuda()
else:
    print(" error no cuda")

predictor = Predictor(beam_search, input_vocab, output_vocab)

####
from rouge import Rouge

print("**training rouge")
references = []
hypothesis = []
test_set = train
for i in range(len(test_set)):
    hypo = predictor.predict(test_set[i].src)
    reference = test_set[i].tgt
    references.append( str(reference) )
    hypothesis.append( str(hypo) )
rouge1 = Rouge()
scores = rouge1.get_scores(hypothesis, references, avg=True)
print(scores)
fileo = open('scores_tr.obj','wb')
pickle.dump(scores, fileo)
fileo = open('references_tr.obj','wb')
pickle.dump(references, fileo)
fileo = open('hypothesis_tr.obj','wb')
pickle.dump(hypothesis, fileo)


print("**testing rouge")
references = []
hypothesis = []
test_set = test
for i in range(len(test_set)):
    hypo = predictor.predict(test_set[i].src)
    reference = test_set[i].tgt
    references.append( str(reference) )
    hypothesis.append( str(hypo) )
rouge1 = Rouge()
scores = rouge1.get_scores(hypothesis, references, avg=True)
print(scores)
fileo = open('scores_te.obj','wb')
pickle.dump(scores, fileo)
fileo = open('references_te.obj','wb')
pickle.dump(references, fileo)
fileo = open('hypothesis_te.obj','wb')
pickle.dump(hypothesis, fileo)

print("**validation rouge")
references = []
hypothesis = []
test_set = dev
for i in range(len(test_set)):
    hypo = predictor.predict(test_set[i].src)
    reference = test_set[i].tgt
    references.append( str(reference) )
    hypothesis.append( str(hypo) )
rouge1 = Rouge()
scores = rouge1.get_scores(hypothesis, references, avg=True)
print(scores)
fileo = open('scores_va.obj','wb')
pickle.dump(scores, fileo)
fileo = open('references_va.obj','wb')
pickle.dump(references, fileo)
fileo = open('hypothesis_va.obj','wb')
pickle.dump(hypothesis, fileo)


#############
seq = train[1].src
print("testing ***** \n ", seq)
stPredict = predictor.predict(seq)
print("output ***** \n ")
print( ' '.join(stPredict) )
################
