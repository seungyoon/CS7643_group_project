import sys, time, random, math, string
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import TabularDataset, Field, BucketIterator

import config

"""
Config - Copy, Reverse or Addition
"""
task = config.task

train_csv = task + '-train.csv'
validation_csv = task + '-validation.csv'
test_csv = task + '-test.csv'
best_model_pt = 'Seq2SeqModel-LSTM-' + task + '.pt'

BATCH_SIZE = config.batch_size

# override data_size
args = len(sys.argv)
if args > 1:
    cmdargs = str(sys.argv)

    print(cmdargs)
    print(cmdargs[1])
    data_size = cmdargs[1]
    print("args:", data_size)
else:
    data_size = config.data_size
    print(data_size)
     


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

print(f"Number of training examples: {len(trn.examples)}")
print(f"Number of validation examples: {len(vld.examples)}")
print(f"Number of test examples: {len(tst.examples)}")

INPUT.build_vocab(trn)
TARGET.build_vocab(trn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter, valid_iter, test_iter = BucketIterator.splits((trn, vld, tst),
                                                          sort_key=lambda x: len(x.input), sort_within_batch=False,
                                                          batch_size=BATCH_SIZE, device=device)

"""
Building the Seq2Seq Model
"""

"""
Encoder
"""
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        # src : [sen_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        
        # embedded : [sen_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [sen_len, batch_size, hid_dim * n_directions]
        # hidden = [n_layers * n_direction, batch_size, hid_dim]
        # cell = [n_layers * n_direction, batch_size, hid_dim]
        return hidden, cell

"""
Decoder
"""
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, dropout=dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        # input = [batch_size]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        input = input.unsqueeze(0)
        # input : [1, ,batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq_len, batch_size, hid_dim * n_dir]
        # hidden = [n_layers * n_dir, batch_size, hid_dim]
        # cell = [n_layers * n_dir, batch_size, hid_dim]
        
        # seq_len and n_dir will always be 1 in the decoder
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch_size, output_dim]
        return prediction, hidden, cell


"""
Seq2Seq
"""
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            'hidden dimensions of encoder and decoder must be equal.'
        assert encoder.n_layers == decoder.n_layers, \
            'n_layers of encoder and decoder must be equal.'
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [sen_len, batch_size]
        # trg = [sen_len, batch_size]
        # teacher_forcing_ratio : the probability to use the teacher forcing.
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        # first input to the decoder is the <sos> token.
        input = trg[0, :]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states 
            # receive output tensor (predictions) and new hidden and cell states.
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # replace predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # decide if we are going to use teacher forcing or not.
            teacher_force = random.random() < teacher_forcing_ratio
            
            # get the highest predicted token from our predictions.
            top1 = output.argmax(1)
            # update input : use ground_truth when teacher_force 
            input = trg[t] if teacher_force else top1
            
        return outputs



"""
Training the Seq2Seq model
"""
# First initialize our model.
INPUT_DIM = len(INPUT.vocab)
OUTPUT_DIM = len(TARGET.vocab)
ENC_EMB_DIM = config.encoder_embedding_size
DEC_EMB_DIM = config.decoder_embedding_size
HID_DIM = config.hidden_size
N_LAYERS = config.num_layer
ENC_DROPOUT = config.encoder_dropout
DEC_DROPOUT = config.decoder_dropout

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(encoder, decoder, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
print(model.apply(init_weights))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src, trg = batch.input, batch.target
        
        optimizer.zero_grad()
        # trg = [sen_len, batch_size]
        # output = [trg_len, batch_size, output_dim]
        output = model(src, trg, 1) # turn on teacher forcing.
        output_dim = output.shape[-1]
        
        # transfrom our output : slice off the first column, and flatten the output into 2 dim.
        output = output[1:].view(-1, output_dim) 
        trg = trg[1:].view(-1)
        # trg = [(trg_len-1) * batch_size]
        # output = [(trg_len-1) * batch_size, output_dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch.input, batch.target

            output = model(src, trg, 0) # turn off teacher forcing.
            
            # trg = [sen_len, batch_size]
            # output = [sen_len, batch_size, output_dim]
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)


# a function that used to tell us how long an epoch takes.
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time  / 60)
    elapsed_secs = int(elapsed_time -  (elapsed_mins * 60))
    return  elapsed_mins, elapsed_secs


def train_seq2seq():
    N_EPOCHS = config.num_epochs
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
    
        start_time = time.time()
    
        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iter, criterion)
    
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_pt)

        if epoch == 0:
            print("\n------------ " + task + " " + config.data_size + " task ------------")
        print(f"Epoch: {epoch+1:02} | Time {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\tValid Loss: {valid_loss:.3f} | Valid PPL: {math.exp(valid_loss):7.3f}")


def test():
    best_model = Seq2Seq(encoder, decoder, device).to(device)
    best_model.load_state_dict(torch.load(best_model_pt))
    
    test_loss = evaluate(model, test_iter, criterion)
    
    print(f"Test Loss : {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f}")


def validate(iterator):
    best_model = Seq2Seq(encoder, decoder, device).to(device)
    best_model.load_state_dict(torch.load(best_model_pt))
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        
        for i, batch in enumerate(iterator):
            src, trg = batch.input, batch.target

            output = model(src, trg, 0) # turn off teacher forcing.

            source = src.transpose(1,0)
            target = trg.transpose(1,0)
            output = output.transpose(1,0)

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

    print("\n------------ " + task + " " + config.data_size + " Task Result ------------")
    print(f"\tSequence  Accuracy: {sequence_match/num_examples:3.3f} | Number of Sequences : {num_examples:5d} |  Sequence Match : {sequence_match:5d}")
    print(f"\tCharacter Accuracy: {character_match/character_count:3.3f} | Number of Characters: {character_count:5d} |  Character Match: {character_match:5d}")
    return


train_seq2seq()
test()
validate(test_iter)
