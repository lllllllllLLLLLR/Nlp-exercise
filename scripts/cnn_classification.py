import pandas as pd
import numpy as np
import torch
import sys
from torch import nn
from torch.optim import Adam
from collections import Counter
from tqdm import tqdm
from torch.utils.data  import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import torch.nn.functional as F
import argparse

sequence_length = 128
max_vocab = 32000

parser = argparse.ArgumentParser()
parser.add_argument("--model",default='cnn',type=str,help="The kind of model (lstm or cnn -- default: lstm).",)
parser.add_argument("--train",default='',type=str,help="Training data in csv format",required=True)
parser.add_argument("--valid",default='',type=str,help="Validation (valid or dev) data in csv format",required=True)
parser.add_argument("--test",default='',type=str,help="Evaluation data in csv format",required=True)
parser.add_argument("--epochs",default=1,type=int,help="Number of epoch")
args = parser.parse_args()

train_file = args.train # train file in csv format
valid_file = args.test # dev/valid file in csv format
test_file = args.valid # test file in csv format
#mymodel = "lstm" # cnn
mymodel = args.model # cnn or lstm

epochs = args.epochs


class SentimentModelLSTM(nn.Module):
    def __init__(self, vocab_size, output_size, hidden_size=128, embedding_size=100, n_layers=2, dropout=0.2):
        super(SentimentModelLSTM, self).__init__()
        self.name = "lstm"

        # embedding layer is useful to map input into vector representation
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # LSTM layer preserved by PyTorch library
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=dropout, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # Linear layer for output
        self.fc = nn.Linear(hidden_size, output_size)

        # Sigmoid layer cz we will have binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # convert feature to long
        x = x.long()

        # map input to vector
        x = self.embedding(x)

        # pass forward to lstm
        o, _ =  self.lstm(x)

        # get last sequence output
        o = o[:, -1, :]

        # apply dropout and fully connected layer
        o = self.dropout(o)
        o = self.fc(o)

        # sigmoid
        o = self.sigmoid(o)

        return o



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SentimentModelCNN(nn.Module):

    def __init__(self, vocab_size,embedding_size,class_size, dropout=0.2):
        super(SentimentModelCNN, self).__init__()
        self.name = "cnn"

        V = vocab_size
        D = embedding_size
        C = class_size
        Ci = 1
        Co = 100
        Ks = [3,4,5]

        self.embed = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

        #if self.args.static:
            #self.embed.weight.requires_grad = False

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


def pad_features(reviews, pad_id, seq_length=128):
    # features = np.zeros((len(reviews), seq_length), dtype=int)
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)

    for i, row in enumerate(reviews):
        # if seq_length < len(row) then review will be trimmed
        features[i, :len(row)] = np.array(row)[:seq_length]

    return features

def encode_review(reviews, index, seq_length=128):

    # encode words
    reviews_enc = []
    for review in tqdm(reviews):
        l_reviews_enc = []
        for word in review.split():
            if word in index:
                l_reviews_enc.append(index[word])
            else:
                l_reviews_enc.append(1)
        reviews_enc.append(l_reviews_enc)
    #reviews_enc = [[index[word] for word in review.split()] for review in tqdm(reviews)]

    x = pad_features(reviews_enc, pad_id=index['<PAD>'], seq_length=seq_length)
    assert len(x) == len(reviews_enc)
    assert len(x[0]) == seq_length
    return x


def load_and_preprocess_data(filename_train,filename_valid,filename_test, seq_length=128, max_vocab=-1):
    print("loading files...")
    l_data_train = pd.read_csv(filename_train)
    l_data_valid = pd.read_csv(filename_valid)
    l_data_test = pd.read_csv(filename_test)

    # get all processed reviews
    reviews_train = l_data_train.review.values
    reviews_valid = l_data_valid.review.values
    reviews_test = l_data_test.review.values


    print("Merging files...")
    # merge into single variable, separated by whitespaces
    words = ' '.join(reviews_train) + " " + ' '.join(reviews_valid) + " " + ' '.join(reviews_test)
    # obtain list of words
    words = words.split()

    # build vocabulary
    print("Building vocab...")
    counter = Counter(words)
    vocab = sorted(counter, key=counter.get, reverse=True)
    if max_vocab != -1:
        vocab = vocab[:max_vocab]
    int2word = dict(enumerate(vocab, 2))
    int2word[0] = '<PAD>'
    int2word[1] = '<UNK>'
    word2int = {word: id for id, word in int2word.items()}


    ## encode words
    #reviews_enc = [[word2int[word] for word in review.split()] for review in tqdm(reviews)]


    #seq_length = 256
    print("Encoding reviews...")
    #train_x = pad_features(reviews_enc, pad_id=word2int['<PAD>'], seq_length=seq_length)
    l_train_x = encode_review(reviews_train,word2int,seq_length)
    l_valid_x = encode_review(reviews_valid,word2int,seq_length)
    l_test_x = encode_review(reviews_test,word2int,seq_length)

    # get labels as numpy
    l_train_y = l_data_train.label.to_numpy()
    l_valid_y = l_data_valid.label.to_numpy()
    l_test_y = l_data_test.label.to_numpy()
    return l_train_x,l_train_y,l_valid_x,l_valid_y,l_test_x,l_test_y,word2int

train_x,train_y,valid_x,valid_y,test_x,test_y,vocab_index = load_and_preprocess_data(train_file,valid_file,test_file,sequence_length,max_vocab)

# print out the shape
print('Feature Shapes:')
print('===============')
print('Train set: {}'.format(train_x.shape))
print('Validation set: {}'.format(valid_x.shape))
print('Test set: {}'.format(test_x.shape))



# define batch size
batch_size = 128

# create tensor datasets
trainset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
validset = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
testset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# create dataloaders
trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
valloader = DataLoader(validset, shuffle=True, batch_size=batch_size)
testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vocab_size = len(vocab_index)
print("Taille vocabulaire", vocab_size)
output_size = 1
embedding_size = 100
hidden_size = 128
n_layers = 1
dropout=0.25


lr = 1e-4
#dropout_keep_prob = 0.5
max_document_length = sequence_length  # each sentence has until 100 words
max_size = vocab_size # maximum vocabulary size
#seed = 1
num_classes = 2
#pool_size = 2
#n_filters = 128
#filter_sizes = [3, 8]

# model initialization
model = None
if mymodel == 'lstm':
    model = SentimentModelLSTM(vocab_size, output_size, hidden_size, embedding_size, n_layers, dropout)
if mymodel == 'cnn':
    model = SentimentModelCNN(vocab_size,embedding_size,num_classes)
#model = SentimentModelCNN(vocab_size, embedding_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes, sequence_length, dropout_keep_prob)
print(model)

# training config
#lr = 0.001
criterion = nn.BCELoss()  # we use BCELoss cz we have binary classification problem

optim = Adam(model.parameters(), lr=lr)
grad_clip = 5


print_every = 1
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'epochs': epochs
}
es_limit = 5


# train loop
model = model.to(device)

epochloop = tqdm(range(epochs), position=0, desc='Training', leave=True)

# early stop trigger
es_trigger = 0
val_loss_min = torch.inf

for e in epochloop:

    #################
    # training mode #
    #################

    model.train()

    train_loss = 0
    train_acc = 0

    for id, (feature, target) in enumerate(trainloader):
        # add epoch meta info
        epochloop.set_postfix_str(f'Training batch {id}/{len(trainloader)}')

        # move to device
        feature, target = feature.to(device), target.to(device)

        # reset optimizer
        optim.zero_grad()

        # forward pass
        out = model(feature)
        predicted = []
        out_probs = []
        loss = 0
        if model.name == 'cnn':
            predicted = torch.tensor([1 if i[0] < i[1] else 0 for i in out > 0.5], device=device)
            #(torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            out_probs = torch.tensor([i[1] if i[0] < i[1] else i[0] for i in out > 0.5], device=device)
            loss = F.cross_entropy(out, target, size_average=False)
        else:
        # acc
            predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
            loss = criterion(out.squeeze(), target.float())
        #print(predicted)
        #print(target)
        equals = predicted == target
        #acc = (torch.max(out, 1)[1].view(target.size()).data == target.data).sum()
        acc = torch.mean(equals.type(torch.FloatTensor))
        train_acc += acc.item()

        # loss
        #loss = criterion(predicted.squeeze(), target.float())
        #loss = F.cross_entropy(out, target, size_average=False)

        train_loss += loss.item()
        loss.backward()

        # clip grad
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # update optimizer
        optim.step()

        # free some memory
        del feature, target, predicted

    history['train_loss'].append(train_loss / len(trainloader))
    history['train_acc'].append(train_acc / len(trainloader))

    ####################
    # validation model #
    ####################

    model.eval()

    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for id, (feature, target) in enumerate(valloader):
            # add epoch meta info
            epochloop.set_postfix_str(f'Validation batch {id}/{len(valloader)}')

            # move to device
            feature, target = feature.to(device), target.to(device)

            # forward pass
            out = model(feature)
            predicted = []
            out_probs = []
            loss = 0
            if model.name == 'cnn':
                predicted = torch.tensor([1 if i[0] < i[1] else 0 for i in out > 0.5], device=device)
                #(torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                out_probs = torch.tensor([i[1] if i[0] < i[1] else i[0] for i in out > 0.5], device=device)
                loss = F.cross_entropy(out, target, size_average=False)
            else:
            # acc
                predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
                loss = criterion(out.squeeze(), target.float())
            #print(predicted)
            #print(target)
            equals = predicted == target

            # acc
            #predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
            #equals = predicted == target
            acc = torch.mean(equals.type(torch.FloatTensor))
            val_acc += acc.item()

            # loss
            #loss = criterion(out.squeeze(), target.float())
            #loss = F.cross_entropy(out, target, size_average=False)
            val_loss += loss.item()

            # free some memory
            del feature, target, predicted

        history['val_loss'].append(val_loss / len(valloader))
        history['val_acc'].append(val_acc / len(valloader))

    # reset model mode
    model.train()

    # add epoch meta info
    epochloop.set_postfix_str(f'Val Loss: {val_loss / len(valloader):.3f} | Val Acc: {val_acc / len(valloader):.3f}')

    # print epoch
    if (e+1) % print_every == 0:
        epochloop.write(f'Epoch {e+1}/{epochs} | Train Loss: {train_loss / len(trainloader):.3f} Train Acc: {train_acc / len(trainloader):.3f} | Val Loss: {val_loss / len(valloader):.3f} Val Acc: {val_acc / len(valloader):.3f}')
        epochloop.update()

    # save model if validation loss decrease
    if val_loss / len(valloader) <= val_loss_min:
        torch.save(model.state_dict(), train_file + '_sentiment_'+ mymodel + "." + str(epochs) + '.pt')
        val_loss_min = val_loss / len(valloader)
        es_trigger = 0
    else:
        epochloop.write(f'[WARNING] Validation loss did not improved ({val_loss_min:.3f} --> {val_loss / len(valloader):.3f})')
        es_trigger += 1

    # force early stop
    if es_trigger >= es_limit:
        epochloop.write(f'Early stopped at Epoch-{e+1}')
        # update epochs history
        history['epochs'] = e+1
        break

# test loop
model.eval()

# metrics
test_loss = 0
test_acc = 0

all_target = []
all_predicted = []

testloop = tqdm(testloader, leave=True, desc='Inference')
with torch.no_grad():
    for feature, target in testloop:
        feature, target = feature.to(device), target.to(device)

        out = model(feature)

        predicted = []
        out_probs = []
        if model.name == 'cnn':
            predicted = torch.tensor([1 if i[0] < i[1] else 0 for i in out > 0.5], device=device)
            #(torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
            out_probs = torch.tensor([i[1] if i[0] < i[1] else i[0] for i in out > 0.5], device=device)
            loss = F.cross_entropy(out, target, size_average=False)
        else:
        # acc
            predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
            loss = criterion(out.squeeze(), target.float())
        #print(predicted)
        #print(target)
        equals = predicted == target

        #predicted = torch.tensor([1 if i == True else 0 for i in out > 0.5], device=device)
        #equals = predicted == target
        acc = torch.mean(equals.type(torch.FloatTensor))
        test_acc += acc.item()

        #loss = criterion(out.squeeze(), target.float())
        #loss = F.cross_entropy(out, target, size_average=False)
        test_loss += loss.item()

        all_target.extend(target.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())

    print(f'Accuracy: {test_acc/len(testloader):.4f}, Loss: {test_loss/len(testloader):.4f}')


print(classification_report(all_predicted, all_target))





