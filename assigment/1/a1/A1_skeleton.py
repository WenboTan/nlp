
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    """Tokenize text into lowercase tokens using a simple approach."""
    # Split on whitespace and punctuation
    words = []
    current_word = ""
    for char in text:
        if char.isalnum() or char == "'":  # Keep alphanumeric chars and apostrophes
            current_word += char
        else:
            if current_word:
                words.append(current_word.lower())
                current_word = ""
            if not char.isspace():  # Add punctuation as separate tokens
                words.append(char.lower())
    if current_word:  # Add the last word if exists
        words.append(current_word.lower())
    return words

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file.

        Args:
             train_file:        The name of the file containing the training texts.
             tokenize_fun:      The function that maps a text to a list of string tokens.
             max_voc_size:      The maximally allowed size of the vocabulary.
             model_max_length:  Truncate texts longer than this length.
             pad_token:         The dummy string corresponding to padding.
             unk_token:         The dummy string corresponding to out-of-vocabulary tokens.
             bos_token:         The dummy string corresponding to the beginning of the text.
             eos_token:         The dummy string corresponding to the end the text.
    """
    # First count token frequencies in the training file
    counter = Counter()
    
    # Read and process training file
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                tokens = tokenize_fun(line.strip())
                counter.update(tokens)
    
    # Create vocabulary with special tokens first
    str_to_int = {
        pad_token: 0,  # Padding token
        bos_token: 1,  # Beginning of sequence token
        eos_token: 2,  # End of sequence token
        unk_token: 3,  # Unknown token
    }
    
    # Calculate how many regular tokens we can add
    remaining_space = max_voc_size - len(str_to_int) if max_voc_size else float('inf')
    
    # Add most common tokens up to the vocabulary size limit
    for token, _ in counter.most_common(int(remaining_space)):
        if token not in str_to_int:  # Avoid adding special tokens again
            str_to_int[token] = len(str_to_int)
    
    # Create reverse mapping
    int_to_str = {v: k for k, v in str_to_int.items()}
    
    # Create and return tokenizer instance
    return A1Tokenizer(
        str_to_int=str_to_int,
        int_to_str=int_to_str,
        tokenize_fun=tokenize_fun,
        model_max_length=model_max_length,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token
    )

class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, int_to_str, tokenize_fun, model_max_length=None,
                 pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
        """Initialize the tokenizer.
        
        Args:
            str_to_int: Dictionary mapping tokens to their integer IDs
            int_to_str: Dictionary mapping integer IDs back to tokens
            tokenize_fun: Function to split text into tokens
            model_max_length: Maximum sequence length (for truncation)
            pad_token: Token used for padding
            unk_token: Token used for unknown words
            bos_token: Token used for beginning of sequence
            eos_token: Token used for end of sequence
        """
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.tokenize_fun = tokenize_fun
        self.model_max_length = model_max_length
        
        # Store special token IDs
        self.pad_token_id = str_to_int[pad_token]
        self.unk_token_id = str_to_int[unk_token]
        self.bos_token_id = str_to_int[bos_token]
        self.eos_token_id = str_to_int[eos_token]

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """Tokenize the given texts and return a BatchEncoding containing the integer-encoded tokens."""
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')

        # Ensure texts is a list of strings
        if isinstance(texts[0], str):
            texts = [texts]  # Make it a batch of 1 if single text list provided
        
        # Tokenize and convert to IDs
        encoded_texts = []
        for text_list in texts:
            sequence = []
            for text in text_list:
                # Add BOS token
                tokens = [self.bos_token_id]
                # Tokenize and convert to IDs
                for token in self.tokenize_fun(text):
                    token_id = self.str_to_int.get(token, self.unk_token_id)
                    tokens.append(token_id)
                # Add EOS token
                tokens.append(self.eos_token_id)
                
                # Truncate if needed
                if truncation and self.model_max_length:
                    tokens = tokens[:self.model_max_length]
                
                sequence.extend(tokens)
            encoded_texts.append(sequence)

        # Find maximum sequence length for padding
        if padding:
            max_len = max(len(seq) for seq in encoded_texts)
            # Pad sequences
            for i in range(len(encoded_texts)):
                padding_length = max_len - len(encoded_texts[i])
                if padding_length > 0:
                    encoded_texts[i].extend([self.pad_token_id] * padding_length)
        
        # Create attention mask if needed
        attention_mask = None
        if padding:
            attention_mask = [[1] * len(seq) for seq in encoded_texts]
            for i, seq in enumerate(attention_mask):
                padding_length = max_len - len(seq)
                if padding_length > 0:
                    attention_mask[i].extend([0] * padding_length)
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            encoded_texts = torch.tensor(encoded_texts)
            if attention_mask is not None:
                attention_mask = torch.tensor(attention_mask)
        
        result = {'input_ids': encoded_texts}
        if attention_mask is not None:
            result['attention_mask'] = attention_mask
            
        return BatchEncoding(result)

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = ...
        self.rnn = ...
        self.unembedding = ...
        
    def forward(self, X):
        """The forward pass of the RNN-based language model.
        
           Args:
             X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
        """
        embedded = ...
        rnn_out, _ = ...
        out = ...
        return out


###
### Part 4. Training the language model.
###

## Hint: the following TrainingArguments hyperparameters may be relevant for your implementation:
#
# - optim:            What optimizer to use. You can assume that this is set to 'adamw_torch',
#                     meaning that we use the PyTorch AdamW optimizer.
# - eval_strategy:    You can assume that this is set to 'epoch', meaning that the model should
#                     be evaluated on the validation set after each epoch
# - use_cpu:          Force the trainer to use the CPU; otherwise, CUDA or MPS should be used.
#                     (In your code, you can just use the provided method select_device.)
# - learning_rate:    The optimizer's learning rate.
# - num_train_epochs: The number of epochs to use in the training loop.
# - per_device_train_batch_size: 
#                     The batch size to use while training.
# - per_device_eval_batch_size:
#                     The batch size to use while evaluating.
# - output_dir:       The directory where the trained model will be saved.

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        """Set up the trainer.
           
           Args:
             model:          The model to train.
             args:           The training parameters stored in a TrainingArguments object.
             train_dataset:  The dataset containing the training documents.
             eval_dataset:   The dataset containing the validation documents.
             eval_dataset:   The dataset containing the validation documents.
             tokenizer:      The tokenizer.
        """
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # TODO: Relevant arguments: at least args.learning_rate, but you can optionally also consider
        # other Adam-related hyperparameters here.
        optimizer = torch.optim.AdamW(...)

        # TODO: Relevant arguments: args.per_device_train_batch_size, args.per_device_eval_batch_size
        train_loader = DataLoader(...)
        val_loader = DataLoader(...)
        
        # TODO: Your work here is to implement the training loop.
        #       
        # for each training epoch (use args.num_train_epochs here):
        #   for each batch B in the training set:
        #
        #       PREPROCESSING AND FORWARD PASS:
        #       input_ids = apply your tokenizer to B
	    #       X = all columns in input_ids except the last one
	    #       Y = all columns in input_ids except the first one
	    #       put X and Y onto the GPU (or whatever device you use)
        #       apply the model to X
        #   	compute the loss for the model output and Y
        #
        #       BACKWARD PASS AND MODEL UPDATE:
        #       optimizer.zero_grad()
        #       loss.backward()
        #       optimizer.step()

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

    
