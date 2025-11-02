
# Understanding GPT Decoder-Only Architecture with Transformers

This notebook provides practical examples to understand how GPT's decoder-only architecture works using the Hugging Face transformers library.

## Installation
First, install the required packages:
```bash
pip install transformers torch
```

## 1. Basic Text Generation - Understanding Autoregressive Behavior

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # smallest GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# Input text
prompt = "The future of artificial intelligence is"

# Tokenize input
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print(f"Input: {prompt}")
print(f"Input tokens: {input_ids}")
print(f"Input shape: {input_ids.shape}")

# Generate text - this demonstrates autoregressive generation
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nGenerated text: {generated_text}")
```

**Key Insight**: GPT generates text one token at a time, using previously generated tokens as context. This is autoregressive generation - the hallmark of decoder-only models.

---

## 2. Step-by-Step Token Prediction - Visualizing the Decoder Process

```python
import torch.nn.functional as F

# Input prompt
prompt = "Machine learning is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

print(f"Starting prompt: {prompt}\n")
print("=" * 60)

# Generate tokens one by one to visualize the process
current_ids = input_ids.clone()
num_tokens_to_generate = 10

for i in range(num_tokens_to_generate):
    # Get model predictions for next token
    with torch.no_grad():
        outputs = model(current_ids)
        # Get logits for the last token position
        next_token_logits = outputs.logits[0, -1, :]
    
    # Get probabilities using softmax
    probs = F.softmax(next_token_logits, dim=-1)
    
    # Get top 5 most probable next tokens
    top_probs, top_indices = torch.topk(probs, 5)
    
    print(f"\nStep {i+1}:")
    print(f"Current text: {tokenizer.decode(current_ids[0])}")
    print(f"Top 5 predictions for next token:")
    for prob, idx in zip(top_probs, top_indices):
        token = tokenizer.decode([idx])
        print(f"  '{token}' -> {prob.item():.4f}")
    
    # Sample next token (using greedy decoding for consistency)
    next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
    
    # Append to current sequence
    current_ids = torch.cat([current_ids, next_token_id], dim=-1)
    
    selected_token = tokenizer.decode(next_token_id[0])
    print(f"Selected: '{selected_token}'")

print("\n" + "=" * 60)
print(f"Final generated text: {tokenizer.decode(current_ids[0])}")
```

**Key Insight**: At each step, GPT looks at ALL previous tokens and predicts the next one. The decoder processes the entire sequence each time, but uses masked attention to prevent looking ahead.

---

## 3. Understanding Masked Self-Attention

```python
import torch

# Create a simple example to understand masking
sequence_length = 5

# Create attention mask for decoder (causal/autoregressive mask)
# This prevents positions from attending to future positions
attention_mask = torch.tril(torch.ones(sequence_length, sequence_length))

print("Causal Attention Mask (Decoder-Only):")
print("1 = can attend, 0 = cannot attend (masked)\n")
print(attention_mask)
print("\nExplanation:")
print("- Row 0 (token 0): can only see token 0 (itself)")
print("- Row 1 (token 1): can see tokens 0, 1")
print("- Row 2 (token 2): can see tokens 0, 1, 2")
print("- Row 3 (token 3): can see tokens 0, 1, 2, 3")
print("- Row 4 (token 4): can see tokens 0, 1, 2, 3, 4")
print("\nThis is why it's called 'decoder-only' - each position")
print("can only attend to previous positions (and itself)")
```

**Key Insight**: The triangular mask ensures that when predicting token N, the model can only see tokens 0 to N-1. This is crucial for autoregressive generation.

---

## 4. Extracting and Analyzing Model Outputs

```python
# Get detailed model outputs including hidden states
prompt = "Python programming is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True, output_attentions=True)

# Extract components
logits = outputs.logits  # Predictions for next token
hidden_states = outputs.hidden_states  # Hidden states from all layers
attentions = outputs.attentions  # Attention weights from all layers

print(f"Input: {prompt}")
print(f"Input shape: {input_ids.shape}")
print(f"\nModel has {len(hidden_states)} layers (including embedding)")
print(f"Logits shape: {logits.shape}")  # (batch_size, sequence_length, vocab_size)
print(f"  - Batch size: {logits.shape[0]}")
print(f"  - Sequence length: {logits.shape[1]}")
print(f"  - Vocabulary size: {logits.shape[2]}")

print(f"\nHidden state for last layer shape: {hidden_states[-1].shape}")
print(f"Attention weights for layer 0 shape: {attentions[0].shape}")
print(f"  - Batch size: {attentions[0].shape[0]}")
print(f"  - Number of attention heads: {attentions[0].shape[1]}")
print(f"  - Sequence length: {attentions[0].shape[2]}")
print(f"  - Sequence length: {attentions[0].shape[3]}")

# Predict next token
next_token_logits = logits[0, -1, :]
next_token_id = torch.argmax(next_token_logits).item()
next_token = tokenizer.decode([next_token_id])

print(f"\nPredicted next token: '{next_token}'")
```

**Key Insight**: GPT processes the entire input sequence through multiple decoder layers, producing hidden states and attention patterns. The final layer's output is converted to vocabulary probabilities.

---

## 5. Comparing Different Decoding Strategies

```python
prompt = "Once upon a time"

print(f"Prompt: {prompt}\n")
print("=" * 80)

# 1. Greedy Decoding - always pick highest probability token
print("\n1. GREEDY DECODING (deterministic):")
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=30, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 2. Sampling with Temperature
print("\n2. SAMPLING with temperature=0.7 (randomness):")
output = model.generate(input_ids, max_length=30, do_sample=True, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 3. Top-k Sampling
print("\n3. TOP-K SAMPLING (k=50):")
output = model.generate(input_ids, max_length=30, do_sample=True, top_k=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 4. Top-p (Nucleus) Sampling
print("\n4. TOP-P SAMPLING (p=0.9):")
output = model.generate(input_ids, max_length=30, do_sample=True, top_p=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# 5. Beam Search
print("\n5. BEAM SEARCH (num_beams=5):")
output = model.generate(input_ids, max_length=30, num_beams=5, early_stopping=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Key Insight**: Decoder-only models support various decoding strategies, all based on the same principle: use previous tokens to predict the next one.

---

## 6. Understanding Context Window

```python
# GPT-2 has a maximum context length of 1024 tokens
max_length = model.config.n_positions
print(f"GPT-2 maximum context length: {max_length} tokens\n")

# Create a long prompt
long_text = "AI " * 600  # This will exceed the context window
input_ids = tokenizer.encode(long_text, return_tensors="pt")

print(f"Long text token count: {input_ids.shape[1]} tokens")

if input_ids.shape[1] > max_length:
    print(f"Text exceeds max length! Truncating to last {max_length} tokens...")
    input_ids = input_ids[:, -max_length:]
    print(f"Truncated token count: {input_ids.shape[1]} tokens")

# Generate with truncated input
with torch.no_grad():
    output = model.generate(input_ids, max_length=input_ids.shape[1] + 20)

print(f"\nGenerated: {tokenizer.decode(output[0][-50:], skip_special_tokens=True)}")
```

**Key Insight**: Decoder-only models have a fixed context window. They can only attend to a limited number of previous tokens.

---

## 7. Visualizing Attention Patterns

```python
import numpy as np

# Generate text with attention outputs
prompt = "The cat sat on"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(input_ids, output_attentions=True)

# Get attention from the first layer, first head
attention = outputs.attentions[0][0, 0].numpy()  # Shape: (seq_len, seq_len)

tokens = [tokenizer.decode([id]) for id in input_ids[0]]

print("Attention Pattern (Layer 0, Head 0):")
print("Rows = Query tokens, Columns = Key tokens\n")
print("Tokens:", tokens)
print("\nAttention weights (each row sums to 1.0):")
print("=" * 60)

for i, token in enumerate(tokens):
    print(f"\n{token:>10} -> ", end="")
    for j in range(len(tokens)):
        if j <= i:  # Due to causal masking
            print(f"{attention[i, j]:.3f} ", end="")
        else:
            print("0.000 ", end="")  # Masked positions

print("\n\n" + "=" * 60)
print("Notice: Upper triangle is zeros (causal masking in decoder)")
print("Each token only attends to previous tokens and itself")
```

**Key Insight**: The attention visualization shows how the causal mask prevents future token attention, which is the defining characteristic of decoder-only architecture.

---

## 8. Using Different GPT Model Sizes

```python
# Compare different GPT-2 model sizes
models_info = {
    "gpt2": "124M parameters",
    "gpt2-medium": "355M parameters", 
    "gpt2-large": "774M parameters",
    "gpt2-xl": "1.5B parameters"
}

prompt = "Artificial intelligence will"

print("Comparing GPT-2 model sizes:\n")
print("=" * 80)

# Only load and test the smallest model to save memory
# Uncomment others if you have sufficient RAM
for model_name in ["gpt2"]:  # Can add others: "gpt2-medium", etc.
    print(f"\nModel: {model_name} ({models_info[model_name]})")
    
    tokenizer_temp = GPT2Tokenizer.from_pretrained(model_name)
    model_temp = GPT2LMHeadModel.from_pretrained(model_name)
    model_temp.eval()
    
    input_ids = tokenizer_temp.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model_temp.generate(
            input_ids, 
            max_length=40,
            do_sample=True,
            temperature=0.7
        )
    
    generated = tokenizer_temp.decode(output[0], skip_special_tokens=True)
    print(f"Generated: {generated}")
    
    print(f"Config - Layers: {model_temp.config.n_layer}, "
          f"Hidden size: {model_temp.config.n_embd}, "
          f"Attention heads: {model_temp.config.n_head}")

print("\n" + "=" * 80)
print("Note: All GPT-2 variants use the same decoder-only architecture,")
print("just with different sizes (more layers, larger hidden dimensions)")
```

**Key Insight**: All GPT models, regardless of size, use the same decoder-only architecture. Larger models have more layers and larger hidden dimensions.

---

## 9. Conditional Generation - Showing Decoder Flexibility

```python
# Despite being decoder-only, GPT can handle various tasks
# through prompt engineering

tasks = [
    ("Translation", "Translate English to French:\nEnglish: Hello, how are you?\nFrench:"),
    ("Question Answering", "Q: What is the capital of France?\nA:"),
    ("Summarization", "Summarize this: Python is a high-level programming language known for its simplicity.\nSummary:"),
    ("Completion", "The three laws of robotics are")
]

print("Decoder-only models can handle various tasks through prompting:\n")
print("=" * 80)

for task_name, prompt in tasks:
    print(f"\n{task_name.upper()}:")
    print(f"Prompt: {prompt}")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 30,
            temperature=0.7,
            do_sample=True
        )
    
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Output: {result}")
    print("-" * 80)
```

**Key Insight**: Decoder-only models are surprisingly versatile. They can perform many tasks (translation, QA, summarization) by treating everything as text generation.

---

## 10. Understanding Token Embeddings

```python
# Access the token embedding layer
embedding_layer = model.transformer.wte  # Word Token Embeddings

# Get embeddings for a few tokens
tokens = ["cat", "dog", "computer", "programming"]
print("Token Embeddings:\n")

for token in tokens:
    token_id = tokenizer.encode(token, add_special_tokens=False)[0]
    embedding = embedding_layer.weight[token_id].detach()
    
    print(f"Token: '{token}' (ID: {token_id})")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 dimensions: {embedding[:10].numpy()}\n")

# Compute similarity between embeddings
def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

cat_emb = embedding_layer.weight[tokenizer.encode("cat", add_special_tokens=False)[0]]
dog_emb = embedding_layer.weight[tokenizer.encode("dog", add_special_tokens=False)[0]]
computer_emb = embedding_layer.weight[tokenizer.encode("computer", add_special_tokens=False)[0]]

print("Embedding Similarities:")
print(f"cat <-> dog: {cosine_similarity(cat_emb, dog_emb):.4f}")
print(f"cat <-> computer: {cosine_similarity(cat_emb, computer_emb):.4f}")
print(f"dog <-> computer: {cosine_similarity(dog_emb, computer_emb):.4f}")
```

**Key Insight**: The decoder starts by converting tokens to dense embeddings. Similar words have similar embeddings.

---

## Summary

This notebook demonstrated:
1. **Autoregressive Generation**: GPT generates one token at a time
2. **Masked Self-Attention**: Prevents looking at future tokens
3. **Decoder-Only Architecture**: No encoder component needed
4. **Flexibility**: Can handle various tasks through prompting
5. **Context Window**: Limited by maximum sequence length
6. **Multiple Decoding Strategies**: Different ways to sample the next token

All GPT models share this decoder-only architecture, making them efficient and powerful for text generation tasks!
