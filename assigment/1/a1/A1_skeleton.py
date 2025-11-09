import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

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

import nltk
def nltk_lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

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
    # First count token frequencies in the training file(s)
    counter = Counter()
    # Allow train_file to be a single path or a list/tuple of paths
    files = train_file if isinstance(train_file, (list, tuple)) else [train_file]

    for path in files:
        with open(path, 'r', encoding='utf-8') as f:
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
    
    # Calculate how many regular tokens we can add. If max_voc_size is None, include all tokens.
    if max_voc_size is None:
        # Add all tokens ordered by frequency
        most_common_tokens = [t for t, _ in counter.most_common()]
    else:
        remaining_space = max_voc_size - len(str_to_int)
        if remaining_space > 0:
            most_common_tokens = [t for t, _ in counter.most_common(remaining_space)]
        else:
            most_common_tokens = []

    # Add the selected tokens (skip tokens that collide with special tokens)
    for token in most_common_tokens:
        if token not in str_to_int:
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
        # Accept either:
        # - texts: list[str] -> batch of strings (each string is one sample)
        # - texts: list[list[str]] -> batch where each sample is a list of strings to be concatenated
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return BatchEncoding({'input_ids': []})

        # Determine whether each element is a str (single sample) or list (concatenation of pieces)
        is_batch_of_lists = isinstance(texts[0], (list, tuple))

        batch_samples = []
        if is_batch_of_lists:
            batch_samples = texts
        else:
            # list[str]
            batch_samples = [[t] for t in texts]

        # Tokenize and convert to IDs per sample
        encoded_texts = []
        orig_lengths = []
        for sample_texts in batch_samples:
            # sample_texts is a list of strings to concatenate for this sample
            tokens = [self.bos_token_id]
            for text in sample_texts:
                for tok in self.tokenize_fun(text):
                    token_id = self.str_to_int.get(tok, self.unk_token_id)
                    tokens.append(token_id)
            tokens.append(self.eos_token_id)

            # We'll apply truncation at the sample (sequence) level if requested
            if truncation and self.model_max_length is not None:
                tokens = tokens[:self.model_max_length]

            encoded_texts.append(tokens)
            orig_lengths.append(len(tokens))

        # Padding: if requested (or requested tensor output with variable lengths), pad to max length
        attention_mask = None
        if padding or (return_tensors == 'pt' and len(set(orig_lengths)) > 1):
            max_len = max(orig_lengths)
            padded = []
            attention_mask = []
            for seq, L in zip(encoded_texts, orig_lengths):
                pad_len = max_len - L
                if pad_len > 0:
                    padded_seq = seq + [self.pad_token_id] * pad_len
                else:
                    padded_seq = seq
                padded.append(padded_seq)
                attention_mask.append([1] * L + [0] * (max_len - L))
            encoded_texts = padded

        # Convert to tensors if requested
        if return_tensors == 'pt':
            if len(encoded_texts) == 0:
                input_ids = torch.empty((0, 0), dtype=torch.long)
            else:
                input_ids = torch.tensor(encoded_texts, dtype=torch.long)
            if attention_mask is not None:
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)

            result = {'input_ids': input_ids}
            if attention_mask is not None:
                result['attention_mask'] = attention_mask
            return BatchEncoding(result)

        # Return Python lists if tensors not requested
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
### Part 2. Loading texts and creating batches
###

def load_text_datasets(train_file, val_file):
    """Load train/val text files using the HuggingFace `datasets` library.

    This returns a dataset dict with keys 'train' and 'val', where each element
    is a dict with key 'text'. Empty lines are automatically filtered out.

    Args:
        train_file: path to training text file
        val_file: path to validation text file
    Returns:
        dataset: a DatasetDict (or None if `datasets` not available)
    """
    if load_dataset is None:
        raise RuntimeError('datasets library is not available in the environment')

    dataset = load_dataset('text', data_files={'train': train_file, 'val': val_file})
    # Filter out empty lines
    dataset = dataset.filter(lambda x: x['text'].strip() != '')
    return dataset


def subsample_dataset(dataset, n=1000):
    """Optionally reduce dataset size for faster development.

    Returns a shallow-wrapped dataset where each split is a torch.utils.data.Subset
    containing at most `n` items.
    """
    from torch.utils.data import Subset
    for sec in ['train', 'val']:
        size = len(dataset[sec])
        k = min(n, size)
        dataset[sec] = Subset(dataset[sec], range(k))
    return dataset


def collate_fn_tokenizer(batch, tokenizer, truncation=True, padding=True, return_tensors='pt'):
    """Collate function for DataLoader that tokenizes a list of dataset examples.

    `batch` is a list of dataset examples (dictionaries) that must contain the key 'text'.
    The function returns a BatchEncoding (matching HuggingFace style) where tensors are
    returned when return_tensors='pt'.
    """
    texts = [ex['text'] for ex in batch]
    # tokenizer will handle padding/truncation and return tensors if requested
    return tokenizer(texts, truncation=truncation, padding=padding, return_tensors=return_tensors)


def make_dataloader(dataset_split, tokenizer, batch_size=8, shuffle=True, num_workers=0, truncation=True, padding=True):
    """Create a PyTorch DataLoader from a dataset split.

    Args:
        dataset_split: a dataset split object (e.g., dataset['train'])
        tokenizer: an A1Tokenizer instance
        batch_size: batch size
        shuffle: whether to shuffle examples (useful for training)
        num_workers: DataLoader workers
    Returns:
        DataLoader that yields BatchEncoding objects
    """
    return DataLoader(dataset_split,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=lambda b: collate_fn_tokenizer(b, tokenizer, truncation=truncation, padding=padding, return_tensors='pt'))


def create_dataloaders(train_file, val_file, tokenizer, train_batch=16, eval_batch=32, subsample=None, num_workers=0):
    """Helper that loads datasets from files and returns train/val DataLoaders.

    Args:
        train_file, val_file: paths
        tokenizer: an A1Tokenizer instance
        train_batch, eval_batch: batch sizes
        subsample: if int, reduce both splits to at most this many examples for quick dev
    Returns:
        train_loader, val_loader
    """
    ds = load_text_datasets(train_file, val_file)
    if subsample is not None:
        ds = subsample_dataset(ds, subsample)

    train_loader = make_dataloader(ds['train'], tokenizer, batch_size=train_batch, shuffle=True, num_workers=num_workers)
    val_loader = make_dataloader(ds['val'], tokenizer, batch_size=eval_batch, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
   

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
        # Initialize layers using hyperparameters from config
        assert config.vocab_size is not None, 'config.vocab_size must be set'
        assert config.embedding_size is not None, 'config.embedding_size must be set'
        assert config.hidden_size is not None, 'config.hidden_size must be set'

        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        # Use LSTM as recommended; batch_first=True so inputs are (batch, seq, features)
        self.rnn = nn.LSTM(input_size=config.embedding_size,
                           hidden_size=config.hidden_size,
                           batch_first=True)
        # Linear layer to map RNN hidden states to vocabulary logits
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, X):
        """The forward pass of the RNN-based language model.
        
           Args:
             X:  The input tensor (2D), consisting of a batch of integer-encoded texts.
           Returns:
             The output tensor (3D), consisting of logits for all token positions for all vocabulary items.
        """
        # X is expected to be shape (batch_size, seq_len) with dtype long
        if X.dtype != torch.long and X.dtype != torch.int:
            X = X.long()

        embedded = self.embedding(X)                    # (B, T, E)
        rnn_out, _ = self.rnn(embedded)                 # rnn_out: (B, T, H)
        out = self.unembedding(rnn_out)                 # (B, T, V)
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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=getattr(args, 'learning_rate', 1e-3))

        # Create DataLoaders using helper from Part 2. Expect that self.train_dataset and self.eval_dataset
        # are iterable datasets returning dicts with key 'text'. Use tokenizer to collate.
        train_batch = getattr(args, 'per_device_train_batch_size', 8)
        eval_batch = getattr(args, 'per_device_eval_batch_size', 8)

        train_loader = make_dataloader(self.train_dataset, self.tokenizer, batch_size=train_batch, shuffle=True, num_workers=getattr(args, 'dataloader_num_workers', 0))
        val_loader = make_dataloader(self.eval_dataset, self.tokenizer, batch_size=eval_batch, shuffle=False, num_workers=getattr(args, 'dataloader_num_workers', 0))
        
        # Implement training loop
        num_epochs = getattr(args, 'num_train_epochs', 1)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            n_tokens = 0
            for batch in train_loader:
                # batch is a BatchEncoding produced by our collate_fn: contains 'input_ids' and 'attention_mask'
                input_ids = batch['input_ids']
                if isinstance(input_ids, torch.Tensor) is False:
                    # If collate returned lists (unlikely), convert to tensor
                    input_ids = torch.tensor(input_ids, dtype=torch.long)

                # Shifted language modeling: predict next token
                X = input_ids[:, :-1].to(device)
                Y = input_ids[:, 1:].to(device)

                logits = self.model(X)  # (B, T-1, V)

                V = logits.size(-1)
                logits_flat = logits.reshape(-1, V)
                targets_flat = Y.reshape(-1)

                loss = loss_func(logits_flat, targets_flat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_tokens = (targets_flat != self.tokenizer.pad_token_id).sum().item()
                running_loss += loss.item() * batch_tokens
                n_tokens += batch_tokens

            avg_loss = running_loss / max(1, n_tokens)
            print(f'Epoch {epoch+1}/{num_epochs} training loss per token: {avg_loss:.6f}')

            # Evaluation after each epoch if requested
            if getattr(args, 'eval_strategy', None) == 'epoch':
                self.model.eval()
                val_loss_acc = 0.0
                val_tokens = 0
                with torch.no_grad():
                    for vb in val_loader:
                        v_input_ids = vb['input_ids']
                        if isinstance(v_input_ids, torch.Tensor) is False:
                            v_input_ids = torch.tensor(v_input_ids, dtype=torch.long)
                        VX = v_input_ids[:, :-1].to(device)
                        VY = v_input_ids[:, 1:].to(device)
                        v_logits = self.model(VX)
                        V = v_logits.size(-1)
                        v_logits_flat = v_logits.reshape(-1, V)
                        v_targets_flat = VY.reshape(-1)
                        v_loss = loss_func(v_logits_flat, v_targets_flat)
                        n_tok = (v_targets_flat != self.tokenizer.pad_token_id).sum().item()
                        val_loss_acc += v_loss.item() * n_tok
                        val_tokens += n_tok

                val_avg = val_loss_acc / max(1, val_tokens)
                print(f'Validation loss per token: {val_avg:.6f}')

        print(f'Saving to {args.output_dir}.')
        os.makedirs(args.output_dir, exist_ok=True)
        self.model.save_pretrained(args.output_dir)

#############################################
# Optional analysis utilities (was previously executed at import time).
# Wrapped in a main guard so training/importing this module does NOT require
# matplotlib or pre-saved model artifacts. Run manually if you need them.
#############################################
if __name__ == '__main__' and False:  # Set to True to enable manual embedding analysis
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # Example: load tokenizer & model (adjust paths as needed)
    example_tokenizer = build_tokenizer('train.txt', max_voc_size=5000, model_max_length=128)
    example_model = A1RNNModel.from_pretrained('./a1_model_out_smoke')
    emb = example_model.embedding.weight.detach().cpu().numpy()  # (V, D)

    def nearest_neighbors(word, topk=10):
        if word not in example_tokenizer.str_to_int:
            print('word not in vocab:', word)
            return []
        idx = example_tokenizer.str_to_int[word]
        vec = emb[idx:idx+1]
        sims = cosine_similarity(vec, emb)[0]
        nn_ids = sims.argsort()[::-1][:topk]
        inv = example_tokenizer.int_to_str
        return [(inv[i], float(sims[i])) for i in nn_ids]

    # Simple PCA plot (optional)
    def plot_pca(sample_size=200):
        vocab_size = emb.shape[0]
        take = min(sample_size, vocab_size)
        sub = emb[:take]
        pca = PCA(n_components=2)
        pts = pca.fit_transform(sub)
        plt.scatter(pts[:,0], pts[:,1], s=5)
        plt.title('Embedding PCA (first {} tokens)'.format(take))
        plt.show()

    # Example usage
    print(nearest_neighbors('the'))
    # plot_pca()


