import torch
from A1_skeleton import build_tokenizer

def test_tokenizer(train_file, val_file, max_voc_size=5000, model_max_length=512):
    print(f"Building tokenizer from {train_file} with max vocabulary size {max_voc_size}...")
    tokenizer = build_tokenizer(train_file, max_voc_size=max_voc_size, model_max_length=model_max_length)
    
    print(f"\nVocabulary size: {len(tokenizer)}")
    
    # Print some special tokens and their IDs
    special_tokens = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    print("\nSpecial tokens:")
    for token in special_tokens:
        token_id = tokenizer.str_to_int.get(token, tokenizer.unk_token_id)
        print(f"{token}: {token_id}")
    
    # Print some common words and their IDs
    print("\nSome common words:")
    common_words = ['the', 'and', 'in', 'of', 'to']
    for word in common_words:
        token_id = tokenizer.str_to_int.get(word, tokenizer.unk_token_id)
        print(f"{word}: {token_id}")
    
    # Test on validation file
    print(f"\nTesting on {val_file}...")
    with open(val_file, 'r', encoding='utf-8') as f:
        # Get first three non-empty paragraphs
        paragraphs = []
        for line in f:
            if line.strip():
                paragraphs.append(line.strip())
                if len(paragraphs) == 3:
                    break
    
    # Tokenize the paragraphs
    encoded = tokenizer(paragraphs, return_tensors='pt', padding=True, truncation=True)
    
    print("\nEncoded shape:", encoded['input_ids'].shape)
    print("First paragraph tokens:")
    first_paragraph_ids = encoded['input_ids'][0].tolist()
    tokens = [tokenizer.int_to_str[id] for id in first_paragraph_ids if id != tokenizer.pad_token_id]
    print(" ".join(tokens[:30]) + "..." if len(tokens) > 30 else " ".join(tokens))
    
    # Save tokenizer
    save_path = 'trained_tokenizer.pkl'
    tokenizer.save(save_path)
    print(f"\nTokenizer saved to {save_path}")
    
    return tokenizer

if __name__ == '__main__':
    train_file = 'train.txt'
    val_file = 'val.txt'
    tokenizer = test_tokenizer(train_file, val_file)