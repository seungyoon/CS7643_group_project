import io, sys, time, random, math, string
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.legacy.data import TabularDataset, Field, BucketIterator
from transformers import VanillaTransformer, UniversalTransformer

import config

"""
Config - Copy, Reverse or Addition
"""
# overrite Transformer Type
args = len(sys.argv)
model_type = config.model_type
if args > 1:
    model_type = str(sys.argv[1])
# override data_size
data_size = config.data_size
if args > 2:
    data_size = str(sys.argv[2])
# override task
task = config.task
if args > 3:
    task = str(sys.argv[3])

train_csv = task + '-train.csv'
validation_csv = task + '-validation.csv'
test_csv = task + '-test.csv'
best_model_pt = 'TransformerModel-' + model_type + '-' + task + '.pt'
BATCH_SIZE = config.batch_size
if data_size == 'large':
    BATCH_SIZE = config.batch_size * 10

"""
Preparing Data
"""
tokenize = lambda x: x.split()
INPUT  = Field(sequential=True, tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True)
TARGET = Field(sequential=True, tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True)

datafields = [("input", INPUT), ("target", TARGET)]

trn, vld, tst = TabularDataset.splits(
        path="data/" + data_size,
        train=train_csv, validation=validation_csv, test=test_csv,
        format='csv',
        skip_header=True,
        fields=datafields)

print(f"Number of {data_size} training examples: {len(trn.examples)}")
print(f"Number of {data_size} validation examples: {len(vld.examples)}")
print(f"Number of {data_size} test examples: {len(tst.examples)}")

INPUT.build_vocab(trn)
TARGET.build_vocab(trn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iter, val_iter, test_iter = BucketIterator.splits((trn, vld, tst),
                                                          sort_key=lambda x: len(x.input), sort_within_batch=False,
                                                          batch_size=BATCH_SIZE, device=device)

"""
Build Transformer
https://colab.research.google.com/drive/1g4ZFCGegOmD-xXL-Ggu7K5LVoJeXYJ75#scrollTo=sWsVpbRMKiJc
"""
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, model_type, intoken, outtoken, hidden, enc_layers=3, dec_layers=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        nhead = hidden//64
        
        self.encoder = nn.Embedding(intoken, emsize)
        #self.pos_encoder = PositionalEncoding(emsize, dropout, config.max_len)

        self.decoder = nn.Embedding(outtoken, emsize)
        #self.pos_decoder = PositionalEncoding(emsize, dropout, config.max_len)

        if model_type == "Vanilla":
            self.transformer = VanillaTransformer(d_model=emsize, nhead=nhead, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=hidden*4, dropout=dropout, activation='relu')
        elif model_type == "Universal":
            self.transformer = UniversalTransformer(d_model=emsize, nhead=nhead, num_encoder_layers=enc_layers, num_decoder_layers=dec_layers, dim_feedforward=hidden*4, dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(emsize, outtoken)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg)).to(trg.device)

        src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        #src = self.pos_encoder(src)

        trg = self.decoder(trg)
        #trg = self.pos_decoder(trg)

        output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask, memory_mask=self.memory_mask,
                                  src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask, memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output




"""
Training the Transformer model
"""
ntokens = len(INPUT.vocab) # the size of vocabulary
emsize = config.encoder_embedding_size # embedding dimension
nhid = config.hidden_size # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = config.num_layer # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
#nhead = config.num_heads # the number of heads in the multiheadattention models
dropout = config.encoder_dropout # the dropout value

model = TransformerModel(model_type, ntokens, ntokens, nhid, enc_layers=nlayers, dec_layers=nlayers, dropout=dropout).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

print(model.apply(init_weights))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-09)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model, optimizer, criterion, iterator):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src, trg = batch.input, batch.target
        
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output.transpose(0, 1).transpose(1, 2), trg.transpose(0, 1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, criterion, iterator):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():    
        for i, batch in enumerate(iterator):
            src, trg = batch.input, batch.target

            output = model(src, trg)
            loss = criterion(output.transpose(0, 1).transpose(1, 2), trg.transpose(0, 1))
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


# a function that used to tell us how long an epoch takes.
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time  / 60)
    elapsed_secs = int(elapsed_time -  (elapsed_mins * 60))
    return  elapsed_mins, elapsed_secs


def train_transformer():
    best_valid_loss = float("inf")
    N_EPOCHS = config.num_epochs

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, optimizer, criterion, train_iter)
        valid_loss = evaluate(model, criterion, val_iter)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        if epoch == 0:
            print("\n------------ " + model_type + " " +  task + " " + data_size + " task ------------")
        print(f"Epoch: {epoch+1:02} | Time {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tValid Loss: {valid_loss:.3f}")
    
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_pt)
    
        scheduler.step()


def test():
    best_model = TransformerModel(ntokens, ntokens, nhid, enc_layers=nlayers, dec_layers=nlayers, dropout=dropout).to(device)
    best_model.load_state_dict(torch.load(best_model_pt))

    test_loss = evaluate(best_model, criterion, test_iter)
    #print(f"Test Loss : {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}")
    print(f"Test Loss : {test_loss:.3f}")


def validate(iterator):
    best_model = TransformerModel(ntokens, ntokens, nhid, enc_layers=nlayers, dec_layers=nlayers, dropout=dropout).to(device)
    best_model.load_state_dict(torch.load(best_model_pt))
    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):
            src, trg = batch.input, batch.target

            output = model(src, trg) # turn off teacher forcing.

            source = src.transpose(1,0)
            target = trg.transpose(1,0)
            output = output.transpose(1,0)

            np.set_printoptions(threshold=sys.maxsize)
            torch.set_printoptions(profile="full")

            break # run only for the first batch, ... for now

    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(profile="full")

    input = np.array([list(map(lambda x: INPUT.vocab.itos[x], source[i])) for i in range(source.shape[0])])
    #print("\nsource:\n", input)

    raw = np.array([list(map(lambda x: TARGET.vocab.itos[x], target[i])) for i in range(target.shape[0])])
    #print("\ntarget:\n", raw)

    token_trans = np.argmax(output.cpu().numpy(), axis = 2)
    predict = np.array([list(map(lambda x: TARGET.vocab.itos[x], token_trans[i])) for i in range(token_trans.shape[0])])
    #print("\nprediction:\n", predict)

    torch.set_printoptions(profile="default")

    # statistics
    num_examples = raw.shape[0]
    sequence_length = len(raw[0][1:-1])
    character_count = sequence_length * num_examples

    sequence_match = 0
    character_match = 0

    for i in range(num_examples):
        # sequence accuracy
        comparison = raw[i][1:-1] == predict[i][1:-1]
        if comparison.all():
            sequence_match += 1
            character_match += sequence_length
        else:
            character_match += np.count_nonzero(comparison)

    print("\n------------ " + model_type + " " + task + " " + data_size + " Task Result ------------")
    print(f"\tSequence  Accuracy: {sequence_match/num_examples:3.3f} | Number of Sequences : {num_examples:5d} |  Sequence Match : {sequence_match:5d}")
    print(f"\tCharacter Accuracy: {character_match/character_count:3.3f} | Number of Characters: {character_count:5d} |  Character Match: {character_match:5d}")
    return


train_transformer()
test()
validate(test_iter)
