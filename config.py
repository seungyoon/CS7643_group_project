#task = 'add'
#task = 'reverse'
#task = 'copy'
task = 'badd'

data_size = 'small'

# LSTM
model = "LSTM"
batch_size = 128
encoder_embedding_size = 128
decoder_embedding_size = 128
hidden_size = 256
num_layer = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
num_epochs = 10

# Transformer : Attention is all you need
num_epochs = 10
num_layer = 6
num_heads = 8
encoder_dropout = 0.1
decoder_dropout = 0.1
encoder_embedding_size = 512
decoder_embedding_size = 512
max_len = 10
