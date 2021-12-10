## Dataset configuration
batch_size = 64
train_path = "../UIT-VSFC/train"
val_path = "../UIT-VSFC/dev"
test_path = "../UIT-VSFC/test"
specials = ["<pad>", "<sos>", "<eos>"]
word_embedding = None
tokenize_level = "syllable" # "word"

## model configuration
embedding_dim = 300
hidden_size = 512

## optimizer configuration
learning_rate = 5e-3


## training configuration
model_checkpoint = ""
epochs = 30