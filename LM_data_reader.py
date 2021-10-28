import os
import torch

"""
Several Notes regarding PTB dataset:
1. Test data set has much lower perplexity than Valid dataset. 
    With a setup of LSTM with 2 layers, 650 nEmbedding and 650 nHidden with 0.5 dropout, 
    the perplexity was 127 and 117 respectively.
2. 
"""


class Corpus:
    """
    Allow automated data streaming from Penn TreeBank data from disk
    """

    def __init__(self, path, word_level):
        """
        path: string, the system path to the directory of Penn Tree Database
        word_level: bool, specifying whether the dataset should be interpreted as word level data. If set to false, char level interpretation will be taken 
        """
        self.dictionary = Dictionary()

        # word level LM data and char level LM data have different file names
        if word_level:
            path_prefix = "word."
        else:
            path_prefix = "char."

        # tokenize data and store in corresponding torch tensors
        self.train_data = self.tokenize(
            os.path.join(path, path_prefix + "train.txt"), True
        )  # only update dictionary with train data

        # store tokens in vocab.txt
        vocab_filepath = os.path.join(path, path_prefix + "vocab.txt")
        if not os.path.exists(vocab_filepath):
            with open(vocab_filepath, "w") as f:
                for token in self.dictionary.token2id:
                    f.write(token + "\n")
            print("Successfully generated vocabulary")

        # tokenize the validation and test data
        self.valid_data = self.tokenize(
            os.path.join(path, path_prefix + "valid.txt"), False
        )
        self.test_data = self.tokenize(
            os.path.join(path, path_prefix + "test.txt"), False
        )

    def tokenize(self, path, update_dictionary):
        """
        tokenize text data

        Input:
        path: string, the system path to the text file in Penn Tree Database
        update_dictionary: bool, if set to true, use this text data to update dictionary; otherwise, use current dictionary to tokenize, and set all unkown tokens as oov 

        Output:
        torch.Tensor: a 1D tensor of contiguous token id sequence
        """
        assert os.path.exists(path)

        # Add tokens to the dictionary
        if update_dictionary:
            with open(path, "r", encoding="utf8") as f:
                for line in f:
                    tokens = line.split() + ["<eos>"]
                    for token in tokens:
                        self.dictionary.add_token(token)

        # start tokenizing text data
        token_ids_list = []
        with open(path, "r", encoding="utf8") as f:
            for line in f:
                tokens = line.split() + ["<eos>"]
                token_ids = []
                for token in tokens:
                    token_ids.append(self.dictionary.get_token_id(token))
                token_ids_list.append(torch.tensor(token_ids).type(torch.int64))

        return torch.cat(token_ids_list)

    def generate_data(self, batch_size, seq_len, which_data, device):
        """
        A generator for train, test or valid purpose

        Input:
        batch_size: int: the number sequences in each batch
        seq_len: int: the length of the sequence
        which data: string: enumerates {"trian", "test", "valid"}
        device: torch.device: specifying where the data will be operated on

        Yield:
        input_data: torch.Tensor: batch of input sequences
        output_data: torch.Tensor: batch of output sequences
        """
        if which_data == "train":
            data = self.train_data
        elif which_data == "valid":
            data = self.valid_data
        elif which_data == "test":
            data = self.test_data
        else:
            raise ValueError(
                "Unknow data type, please choose from train, valid and test"
            )

        # align data with batch size
        n_batch = data.size(0) // batch_size
        data = data.narrow(0, 0, n_batch * batch_size)
        data = data.view(batch_size, -1)
        data = data.to(device)

        # generate data in batch
        for i in range(0, data.size(1) - seq_len, seq_len):
            input_data = data[:, i : i + seq_len].to(device)
            output_data = data[:, i + 1 : i + 1 + seq_len].to(device)
            yield input_data, output_data


class Dictionary:
    """
    Keep record of the vocabulary in use
    """

    def __init__(self):
        self.token2id = {}
        self.id2token = {}

        # initialize id 0 token to be <oov>, i.e. out of vocabulary
        self.token2id["<oov>"] = 0
        self.id2token[0] = "<oov>"

        self.token_count = 1

    def add_token(self, token):
        """
        Add token to dictionary

        token: string
        """
        assert token != self.id2token[0]  # token should not be <oov>
        if token not in self.token2id:
            self.token2id[token] = self.token_count
            self.id2token[self.token_count] = token
            self.token_count += 1

    def get_token_id(self, token):
        """
        Return the id of a token, if token is not in the dictionary, <oov> will be returned

        token: string
        """
        assert token != self.id2token[0]  # token should not be <oov>
        if token in self.token2id:
            return self.token2id[token]
        else:
            return 0  # return <oov> id

