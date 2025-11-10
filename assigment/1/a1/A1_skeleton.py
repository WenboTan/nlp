# A1_skeleton.py
# Complete reference implementation for DAT450/DIT247 Assignment 1 (RNN LM)
# All comments are in English as requested.

import os
import time
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import nltk
from transformers import PreTrainedModel, PretrainedConfig
from transformers.tokenization_utils_base import BatchEncoding


############################################################
# Part 1: Tokenization
############################################################

def lowercase_tokenizer(text: str):
    """
    Basic word splitter with NLTK that lower-cases tokens.
    Includes robust downloads for punkt / punkt_tab if missing.
    """
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    # Some environments (newer NLTK) separate tables in punkt_tab
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            # Not all NLTK versions have this; ignore if unavailable.
            pass
    # Use word_tokenize (works reasonably on Wikipedia-like text)
    return [t.lower() for t in nltk.word_tokenize(text)]


class A1Tokenizer:
    """
    A minimal tokenizer similar to HuggingFace tokenizers:
    - Holds a vocabulary with 4 special tokens: PAD, UNK, BOS, EOS
    - __call__ supports truncation, padding, return_tensors='pt'
    - save / from_file for persistence
    """

    def __init__(self, *,
                 str_to_int, int_to_str,
                 pad_token, unk_token, bos_token, eos_token,
                 pad_token_id, unk_token_id, bos_token_id, eos_token_id,
                 tokenize_fun,
                 model_max_length=None):
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tokenize_fun = tokenize_fun
        self.model_max_length = model_max_length  # Used by truncation

    def __len__(self):
        """Return vocabulary size."""
        return len(self.str_to_int)

    def _encode_one(self, text: str, truncation: bool = False):
        toks = self.tokenize_fun(text)
        ids = [self.bos_token_id] + [self.str_to_int.get(t, self.unk_token_id) for t in toks] + [self.eos_token_id]
        if truncation and self.model_max_length is not None:
            ids = ids[:self.model_max_length]
        return ids

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        """
        Tokenize input.
        - texts can be: str | list[str] | dict with key 'text' | list[dict{'text': str}]
        - padding: right-pad with PAD to the longest sequence in batch
        - return_tensors: if 'pt', returns PyTorch tensors
        Returns a BatchEncoding with keys: input_ids, attention_mask
        """
        if return_tensors and return_tensors != 'pt':
            raise ValueError("return_tensors must be 'pt' or None")

        # Normalize inputs to list[str]
        if isinstance(texts, dict) and 'text' in texts:
            texts = texts['text']
        elif isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], dict) and 'text' in texts[0]:
            texts = [x['text'] for x in texts]
        elif isinstance(texts, str):
            texts = [texts]

        encoded = [self._encode_one(t, truncation=truncation) for t in texts]

        if padding:
            max_len = max(len(x) for x in encoded) if encoded else 0
            input_ids = []
            attention_mask = []
            for ids in encoded:
                pad_len = max_len - len(ids)
                input_ids.append(ids + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
        else:
            input_ids = encoded
            attention_mask = [[1] * len(x) for x in encoded]

        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})

    def save(self, filename: str):
        """Serialize tokenizer to disk."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename: str):
        """Load tokenizer from file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)


def build_tokenizer(train_file: str,
                    tokenize_fun=lowercase_tokenizer,
                    max_voc_size: int | None = None,
                    model_max_length: int | None = None,
                    pad_token: str = '<PAD>',
                    unk_token: str = '<UNK>',
                    bos_token: str = '<BOS>',
                    eos_token: str = '<EOS>') -> A1Tokenizer:
    """
    Build vocabulary from the training file (one paragraph per non-empty line).
    Keeps most frequent tokens up to max_voc_size (including the 4 specials).
    """
    counter = Counter()

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = tokenize_fun(line)
            counter.update(toks)

    special_tokens = [pad_token, unk_token, bos_token, eos_token]
    if max_voc_size is not None:
        keep_k = max(max_voc_size - len(special_tokens), 0)
        most_common = [w for w, _ in counter.most_common(keep_k)]
    else:
        most_common = list(counter.keys())

    vocab = special_tokens + most_common
    str_to_int = {s: i for i, s in enumerate(vocab)}
    int_to_str = {i: s for s, i in str_to_int.items()}

    return A1Tokenizer(
        str_to_int=str_to_int,
        int_to_str=int_to_str,
        pad_token=pad_token,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token_id=str_to_int[pad_token],
        unk_token_id=str_to_int[unk_token],
        bos_token_id=str_to_int[bos_token],
        eos_token_id=str_to_int[eos_token],
        tokenize_fun=tokenize_fun,
        model_max_length=model_max_length
    )


############################################################
# Part 2: Data utilities (collate fn for DataLoader)
############################################################

def text_collate_fn(batch):
    """
    Collate function that converts:
      - list[str] -> list[str]
      - list[{'text': str}] -> list[str]
    """
    if len(batch) == 0:
        return []
    if isinstance(batch[0], dict) and 'text' in batch[0]:
        return [x['text'] for x in batch]
    return batch


############################################################
# Part 3: RNN Language Model (HF-style)
############################################################

class A1RNNModelConfig(PretrainedConfig):
    """
    Configuration object for the RNN-based language model.
    """
    def __init__(self, vocab_size=None, embedding_size=None, hidden_size=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size


class A1RNNModel(PreTrainedModel):
    """
    A minimal HF-style RNN language model:
    Embedding -> GRU -> Linear (vocab logits)
    """
    config_class = A1RNNModelConfig

    def __init__(self, config: A1RNNModelConfig):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.GRU(
            input_size=config.embedding_size,
            hidden_size=config.hidden_size,
            batch_first=True
        )
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, X: torch.Tensor):
        """
        Args:
            X: [B, T] integer token IDs
        Returns:
            logits: [B, T, V] unnormalized scores over vocabulary
        """
        emb = self.embedding(X)          # [B, T, E]
        rnn_out, _ = self.rnn(emb)       # [B, T, H]
        logits = self.unembedding(rnn_out)  # [B, T, V]
        return logits


############################################################
# Part 4: Trainer (basic, single-GPU/CPU)
############################################################

class A1Trainer:
    """
    A minimal trainer similar to HuggingFace's Trainer:
    - AdamW optimizer
    - Padding-aware CE loss with ignore_index for PAD
    - Epoch-level evaluation on validation set
    """

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert (args.optim == 'adamw_torch'), "Only 'adamw_torch' is supported in this skeleton."
        assert (args.eval_strategy == 'epoch'), "Only eval at 'epoch' is supported in this skeleton."

    def select_device(self):
        if getattr(self.args, "use_cpu", False):
            return torch.device('cpu')
        if not getattr(self.args, "no_cuda", False) and torch.cuda.is_available():
            return torch.device('cuda')
        # For Apple Silicon / Metal
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    def train(self):
        """Run the training loop."""
        args = self.args
        device = self.select_device()
        print('Device:', device)
        self.model.to(device)

        pad_id = self.tokenizer.pad_token_id
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=text_collate_fn
        )
        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=text_collate_fn
        )

        for epoch in range(args.num_train_epochs):
            self.model.train()
            t0 = time.time()
            running_tokens = 0
            running_loss_weighted = 0.0

            for batch_texts in train_loader:
                enc = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )
                input_ids = enc['input_ids'].to(device)  # [B, T]

                if input_ids.size(1) < 2:
                    # Not enough length to build (X, Y) = shift by 1
                    continue

                # Autoregressive target: predict token t from context up to t-1
                X = input_ids[:, :-1].contiguous()   # [B, T-1]
                Y = input_ids[:, 1:].contiguous()    # [B, T-1]

                logits = self.model(X)               # [B, T-1, V]
                B, Tm1, V = logits.shape

                loss = loss_fn(
                    logits.reshape(B * Tm1, V),
                    Y.reshape(B * Tm1)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    valid_mask = (Y != pad_id)
                    n_valid = int(valid_mask.sum().item())
                    running_tokens += n_valid
                    running_loss_weighted += loss.item() * n_valid

            # Token-weighted average NLL and perplexity
            train_nll = running_loss_weighted / max(1, running_tokens)
            train_ppl = float(np.exp(train_nll))
            t1 = time.time()
            print(f"[Epoch {epoch+1}/{args.num_train_epochs}] "
                  f"Train NLL={train_nll:.4f} PPL={train_ppl:.2f}  ({t1 - t0:.1f}s)")

            # ----- Validation -----
            self.model.eval()
            val_tokens = 0
            val_loss_weighted = 0.0
            with torch.no_grad():
                for batch_texts in val_loader:
                    enc = self.tokenizer(
                        batch_texts,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    input_ids = enc['input_ids'].to(device)
                    if input_ids.size(1) < 2:
                        continue

                    X = input_ids[:, :-1].contiguous()
                    Y = input_ids[:, 1:].contiguous()

                    logits = self.model(X)
                    B, Tm1, V = logits.shape
                    loss = loss_fn(
                        logits.reshape(B * Tm1, V),
                        Y.reshape(B * Tm1)
                    )

                    valid_mask = (Y != pad_id)
                    n_valid = int(valid_mask.sum().item())
                    val_tokens += n_valid
                    val_loss_weighted += loss.item() * n_valid

            val_nll = val_loss_weighted / max(1, val_tokens)
            val_ppl = float(np.exp(val_nll))
            print(f"           Valid NLL={val_nll:.4f} PPL={val_ppl:.2f}")

        print(f"Saving to {args.output_dir}.")
        os.makedirs(args.output_dir, exist_ok=True)
        self.model.save_pretrained(args.output_dir)

############################################################
# Part 5: Evaluation utilities
############################################################

def compute_validation_perplexity(model, tokenizer, dataloader, device):
    """
    Compute token-weighted NLL and perplexity on validation loader.
    """
    model.eval()
    pad_id = tokenizer.pad_token_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    total_tokens = 0
    total_loss_weighted = 0.0
    with torch.no_grad():
        for batch_texts in dataloader:
            enc = tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(device)
            if input_ids.size(1) < 2:
                continue
            X = input_ids[:, :-1]
            Y = input_ids[:, 1:]
            logits = model(X)
            B, Tm1, V = logits.shape
            loss = loss_fn(logits.reshape(B * Tm1, V), Y.reshape(B * Tm1))
            valid_mask = (Y != pad_id)
            n_valid = int(valid_mask.sum().item())
            total_tokens += n_valid
            total_loss_weighted += loss.item() * n_valid
    nll = total_loss_weighted / max(1, total_tokens)
    ppl = float(np.exp(nll))
    return nll, ppl


def predict_next_word(model, tokenizer, text, topk=5, device=None):
    """
    Given a text prompt, return top-k next-token candidates (word, probability).
    This uses greedy top-k on the last real token (excluding EOS).
    """
    model.eval()
    enc = tokenizer(text, truncation=True, padding=False, return_tensors='pt')
    input_ids = enc['input_ids']
    if input_ids.size(1) < 1:
        return []
    if device is not None:
        input_ids = input_ids.to(device)
    with torch.no_grad():
        logits = model(input_ids)            # [1, T, V]
        # Use -2 position (last real token before EOS) for better predictions
        if logits.size(1) >= 2:
            last_logits = logits[0, -2, :]
        else:
            last_logits = logits[0, -1, :]
        # Convert to probabilities
        probs = torch.softmax(last_logits, dim=-1)
        scores, idxs = torch.topk(probs, k=topk)
    idxs = idxs.tolist()
    scores = scores.tolist()
    words = [tokenizer.int_to_str[i] for i in idxs]
    return list(zip(words, scores))


############################################################
# Example main (sanity check). Adjust paths for your env.
############################################################

if __name__ == "__main__":
    # ---- Paths (Minerva defaults; change if running locally) ----
    TRAIN_FILE = "/data/courses/2025_dat450_dit247/assignments/a1/train.txt"
    VAL_FILE   = "/data/courses/2025_dat450_dit247/assignments/a1/val.txt"
    TOKENIZER_FILE = "a1_tokenizer.pkl"

    # ---- Build or load tokenizer ----
    if os.path.exists(TOKENIZER_FILE):
        tokenizer = A1Tokenizer.from_file(TOKENIZER_FILE)
        print("Loaded tokenizer from disk.")
    else:
        tokenizer = build_tokenizer(
            TRAIN_FILE,
            tokenize_fun=lowercase_tokenizer,
            max_voc_size=30000,     # Adjust as needed / hyperparameter
            model_max_length=128    # Max sequence length used for truncation
        )
        tokenizer.save(TOKENIZER_FILE)
        print("Built and saved tokenizer.")

    # ---- Load datasets with HuggingFace Datasets ----
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install `datasets` (pip install datasets).")

    dataset = load_dataset('text', data_files={'train': TRAIN_FILE, 'val': VAL_FILE})
    # Remove empty lines
    dataset['train'] = dataset['train'].filter(lambda x: x['text'].strip() != '')
    dataset['val']   = dataset['val'].filter(lambda x: x['text'].strip() != '')

    # Optionally use small subsets during development for faster debugging
    # from torch.utils.data import Subset
    # dataset['train'] = Subset(dataset['train'], range(5000))
    # dataset['val']   = Subset(dataset['val'],   range(1000))

    # ---- TrainingArguments (minimal) ----
    class TrainingArguments:
        def __init__(self):
            self.optim = 'adamw_torch'
            self.eval_strategy = 'epoch'
            self.use_cpu = False            # Set True to force CPU
            self.no_cuda = False            # Set True to disable CUDA even if available
            self.learning_rate = 2e-3
            self.num_train_epochs = 1       # Increase for better results
            self.per_device_train_batch_size = 64
            self.per_device_eval_batch_size = 64
            self.output_dir = "trainer_output"

    args = TrainingArguments()

    # ---- Model config & instance ----
    config = A1RNNModelConfig(
        vocab_size=len(tokenizer),
        embedding_size=256,
        hidden_size=512
    )
    model = A1RNNModel(config)

    # ---- Trainer ----
    trainer = A1Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        tokenizer=tokenizer
    )
    trainer.train()

    # ---- Final validation perplexity (required for submission printout) ----
    device = trainer.select_device()
    val_loader = DataLoader(dataset['val'],
                            batch_size=args.per_device_eval_batch_size,
                            shuffle=False,
                            collate_fn=text_collate_fn)
    nll, ppl = compute_validation_perplexity(model, tokenizer, val_loader, device)
    print(f"[FINAL] Validation NLL={nll:.4f}  PPL={ppl:.2f}")

    # ---- Next-word prediction example (required for submission printout) ----
    prompt = "She lives in San"
    top = predict_next_word(model, tokenizer, prompt, topk=5, device=device)
    print(f'Next-word prediction for: "{prompt}"')
    for w, s in top:
        print(f"  {w:15s} logit={s:.3f}")
