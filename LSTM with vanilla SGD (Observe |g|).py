import transformers
from LM_data_reader import Corpus
from train_utils import evaluate_on_dataset, detach
import torch
from torch import dropout_, nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import transformers
import time
import math

# Best Hyper parameters
word_level = True
seed = 1337
embed_size = 650
hidden_size = 650
num_layers = 2
embedding_dropout_rate = 0.5
hid_dropout_rate = 0.5
output_dropout_rate = 0.5
num_epochs = 80
batch_size = 20
seq_len = 35
lr = 5
momentum = 0
wdecay = 0
lr_gamma = 0.5
patience = 0
clip_grad = 0.25

# initialize seed
torch.manual_seed(seed)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("The device in use is: {}".format(device))

# load PTB data
PTBData = Corpus("datasets/penn-treebank/", word_level)
vocab_size = PTBData.dictionary.token_count
print("The vocabulary size of training data is: {}".format(vocab_size))

# RNN based language model
class RNNLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        hidden_size,
        num_layers,
        embed_drop,
        hid_drop,
        output_drop,
    ):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_dropout = nn.Dropout(embed_drop)
        self.rnn = nn.LSTM(
            embed_size, hidden_size, num_layers, dropout=hid_drop, batch_first=True,
        )
        self.output_dropout = nn.Dropout(output_drop)
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.embed.weight = self.decoder.weight

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        x = self.embed_dropout(x)

        # Forward propagate LSTM
        out, h = self.rnn(x, h)

        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out = self.output_dropout(out)

        # Decode hidden states of all time steps
        out = self.decoder(out)
        return out, h


# create LM model
model = RNNLM(
    vocab_size,
    embed_size,
    hidden_size,
    num_layers,
    embed_drop=embedding_dropout_rate,
    hid_drop=hid_dropout_rate,
    output_drop=output_dropout_rate,
).to(device)
print(
    "number of parameters: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(
#     model.parameters(), lr=lr, momentum=momentum, weight_decay=wdecay
# )
optimizer = torch.optim.SGD(
    model.parameters(), lr=lr, nesterov=False, momentum=momentum, weight_decay=wdecay
)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", lr_gamma, patience=patience, threshold=0
)

# Train the model
try:
    for epoch in range(num_epochs):
        # record training time
        epoch_start = time.time()

        # initialize RNN states
        states = (
            torch.zeros(num_layers, batch_size, hidden_size).to(device),
            torch.zeros(num_layers, batch_size, hidden_size).to(device),
        )

        # training iterations
        model.train()
        total_train_loss = 0
        n_train_loss = 0
        for inputs, targets in PTBData.generate_data(
            batch_size, seq_len, "train", device
        ):
            # Forward pass
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_grad)

            # before gradient clipping, find the norm of all gradients
            gradient_norm = 0
            for p in model.parameters():
                gradient_norm += p.grad.norm(2).item() ** 2
            gradient_norm = math.sqrt(gradient_norm)

            for p in model.parameters():
                p.grad /= gradient_norm

            # gradient_norm = 0
            # for p in model.parameters():
            #     gradient_norm += p.grad.norm(2).item() ** 2
            # gradient_norm = math.sqrt(gradient_norm)
            # print("currect gradient norm is: {}".format(gradient_norm))

            optimizer.step()

            # accumulate loss
            total_train_loss += loss.item()
            n_train_loss += 1
        total_train_loss /= n_train_loss

        # calculate model performance on validation set
        total_valid_loss = evaluate_on_dataset(
            PTBData, model, "valid", batch_size, seq_len, device
        )

        # schedule the learning rate
        scheduler.step(total_valid_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        # print training stats for the final batch of the training data
        print(
            "For Training, at Epoch [{}/{}], Training Time: {:.3f}, Learning rate: {:.4f}, Train vs Valid Loss: {:.4f} / {:.4f}, Train vs Valid Perplexity: {:5.2f} / {:5.2f}".format(
                epoch + 1,
                num_epochs,
                time.time() - epoch_start,
                cur_lr,
                total_train_loss,
                total_valid_loss,
                np.exp(total_train_loss),
                np.exp(total_valid_loss),
            )
        )
except KeyboardInterrupt:
    print("-" * 89)
    print("Exiting from training early")

# calculate model performance on test set
total_test_loss = 0
total_test_loss = evaluate_on_dataset(
    PTBData, model, "test", batch_size, seq_len, device
)

# print out the perplexity and loss on test data
print(
    "For test data, Loss: {:.4f}, Perplexity: {:5.2f}".format(
        total_test_loss, np.exp(total_test_loss)
    )
)

