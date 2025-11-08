import nltk
nltk.download('punkt')  # Download required NLTK data

from A1_skeleton import build_tokenizer

# Create a small test file
with open('test.txt', 'w', encoding='utf-8') as f:
    f.write("This is a test.\nAnother test with more words.\nLet's see how it handles contractions!")

# Build tokenizer with small vocab size
tokenizer = build_tokenizer('test.txt', max_voc_size=10, model_max_length=10)

# Test tokenization
test_texts = [['This is a test.', 'Another test.']]
output = tokenizer(test_texts, return_tensors='pt', padding=True, truncation=True)

print("Vocabulary size:", len(tokenizer))
print("\nInput IDs:")
print(output['input_ids'])
if 'attention_mask' in output:
    print("\nAttention Mask:")
    print(output['attention_mask'])

# Test token-to-int and back conversion
print("\nVocabulary test:")
for token in ['test', 'unknown_word', '<PAD>', '<BOS>', '<EOS>', '<UNK>']:
    token_id = tokenizer.str_to_int.get(token, tokenizer.unk_token_id)
    back_to_token = tokenizer.int_to_str.get(token_id, '<MISSING>')
    print(f"{token} -> {token_id} -> {back_to_token}")