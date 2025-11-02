
# Understanding Encoder-Decoder Architecture with Transformers

This notebook provides practical examples to understand how encoder-decoder models work using the Hugging Face transformers library.

## Installation
First, install the required packages:
```bash
pip install transformers torch sentencepiece
```

## 1. Basic Translation - Understanding Encoder-Decoder Flow

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Load T5-small (encoder-decoder model)
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

model.eval()

# Input text - T5 uses task prefixes
input_text = "translate English to German: The house is beautiful"

# Tokenize
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

print(f"Input: {input_text}")
print(f"Input tokens shape: {input_ids.shape}")
print(f"Input token IDs: {input_ids[0][:10]}...")  # First 10 tokens

# Generate translation
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(f"\nOutput: {output_text}")
print(f"Output tokens shape: {output_ids.shape}")
```

**Key Insight**: Unlike decoder-only models, encoder-decoder models have two parts:
- **Encoder**: Processes the entire input at once and creates context representations
- **Decoder**: Generates output autoregressively, attending to encoder outputs

---

## 2. Visualizing Encoder and Decoder Separation

```python
# Access encoder and decoder separately
print("Model Architecture:")
print(f"Model type: {model.config.model_type}")
print(f"\nEncoder layers: {model.config.num_layers}")
print(f"Decoder layers: {model.config.num_decoder_layers}")
print(f"Hidden size: {model.config.d_model}")
print(f"Attention heads: {model.config.num_heads}")

# Get encoder and decoder outputs separately
input_text = "translate English to French: Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

with torch.no_grad():
    # Encoder forward pass - processes entire input at once
    encoder_outputs = model.encoder(input_ids)

    print(f"\nEncoder output shape: {encoder_outputs.last_hidden_state.shape}")
    print(f"  - Batch size: {encoder_outputs.last_hidden_state.shape[0]}")
    print(f"  - Sequence length: {encoder_outputs.last_hidden_state.shape[1]}")
    print(f"  - Hidden dimension: {encoder_outputs.last_hidden_state.shape[2]}")

print("\nEncoder creates context-aware representations of the entire input.")
print("Decoder then uses these representations to generate output tokens.")
```

**Key Insight**: The encoder processes all input tokens in parallel (bidirectional attention), creating rich contextual representations. The decoder then uses these representations to generate output.

---

## 3. Step-by-Step Decoder Generation with Encoder Context

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Use a translation model
model_name = "Helsinki-NLP/opus-mt-en-de"  # English to German
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model.eval()

# Input sentence
input_text = "I love programming in Python"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

print(f"Input: {input_text}")
print("=" * 80)

# Encode input once
with torch.no_grad():
    encoder_outputs = model.get_encoder()(input_ids)

print(f"\nEncoder processed input (shape: {encoder_outputs.last_hidden_state.shape})")
print("This encoding is reused for all decoder steps!\n")

# Manual decoder generation to visualize the process
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

print("Decoder Generation Steps:")
print("=" * 80)

for step in range(15):  # Generate up to 15 tokens
    with torch.no_grad():
        # Decoder uses: (1) its own previous outputs, (2) encoder outputs
        outputs = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids
        )

        # Get next token prediction
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()

        # Check for end token
        if next_token_id == tokenizer.eos_token_id:
            print(f"\nStep {step + 1}: <EOS> token generated. Translation complete.")
            break

        # Decode and display
        next_token = tokenizer.decode([next_token_id])
        current_output = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

        print(f"Step {step + 1}: '{current_output}' -> adding '{next_token}'")

        # Append to decoder input
        decoder_input_ids = torch.cat([
            decoder_input_ids,
            torch.tensor([[next_token_id]])
        ], dim=1)

print("\n" + "=" * 80)
final_output = tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)
print(f"Final translation: {final_output}")
```

**Key Insight**: The encoder runs ONCE to create contextual representations. The decoder then autoregressively generates output, attending to these encoder representations at each step.

---

## 4. Understanding Attention Mechanisms

```python
# Encoder-decoder models use THREE types of attention:
# 1. Encoder self-attention (bidirectional)
# 2. Decoder self-attention (causal/masked)
# 3. Cross-attention (decoder attends to encoder outputs)

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_text = "translate English to French: Good morning"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate with attention outputs
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_length=20,
        output_attentions=True,
        output_scores=True,
        return_dict_in_generate=True
    )

print("Attention Types in Encoder-Decoder Models:\n")
print("1. ENCODER SELF-ATTENTION:")
print("   - Each input token attends to ALL other input tokens")
print("   - Bidirectional (can see past and future)")
print("   - Creates contextual understanding of input")

print("\n2. DECODER SELF-ATTENTION:")
print("   - Each output token attends to previous output tokens only")
print("   - Causal/masked (like GPT)")
print("   - Maintains autoregressive property")

print("\n3. CROSS-ATTENTION (Encoder-Decoder Attention):")
print("   - Decoder tokens attend to ALL encoder outputs")
print("   - This is how decoder 'reads' the input")
print("   - Most important for seq-to-seq tasks")

generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
print(f"\nGenerated: {generated_text}")
```

**Key Insight**: Cross-attention is the key difference from decoder-only models. It allows the decoder to focus on relevant parts of the input while generating each output token.

---

## 5. Comparing Decoder-Only vs Encoder-Decoder

```python
import torch

# Simulate attention masks for both architectures
sequence_length = 5

# DECODER-ONLY (like GPT): Causal mask
decoder_only_mask = torch.tril(torch.ones(sequence_length, sequence_length))

# ENCODER: Full attention (bidirectional)
encoder_mask = torch.ones(sequence_length, sequence_length)

# DECODER in Encoder-Decoder: Causal mask
decoder_mask = torch.tril(torch.ones(sequence_length, sequence_length))

# CROSS-ATTENTION: Decoder can attend to all encoder outputs
cross_attention_mask = torch.ones(sequence_length, sequence_length)

print("DECODER-ONLY MODEL (GPT):")
print("Self-attention mask (causal):")
print(decoder_only_mask.int())
print("\nEach token can only see previous tokens\n")

print("=" * 60)

print("\nENCODER-DECODER MODEL (T5, BART):")
print("\n1. Encoder Self-Attention (bidirectional):")
print(encoder_mask.int())
print("Each input token sees ALL other input tokens")

print("\n2. Decoder Self-Attention (causal):")
print(decoder_mask.int())
print("Each output token sees only previous output tokens")

print("\n3. Cross-Attention (decoder -> encoder):")
print(cross_attention_mask.int())
print("Each output token can attend to ALL encoder outputs")
```

**Key Insight**: Encoder-decoder models use bidirectional attention in the encoder, enabling better understanding of context. Decoder-only models use only causal attention throughout.

---

## 6. Multiple Seq-to-Seq Tasks with T5

```python
# T5 is a versatile encoder-decoder model trained on multiple tasks
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

model.eval()

# Different tasks with task prefixes
tasks = [
    ("translate English to German: Hello world", "Translation"),
    ("summarize: The Transformer architecture has revolutionized NLP. It uses attention mechanisms instead of recurrence. This allows parallel processing and better long-range dependencies.", "Summarization"),
    ("question: What is the capital of France? context: Paris is the capital and largest city of France.", "Question Answering"),
    ("sentiment: This movie was absolutely fantastic!", "Sentiment Classification"),
    ("cola sentence: The book was read by me", "Grammar Check")
]

print("T5: A Unified Encoder-Decoder Model for Multiple Tasks\n")
print("=" * 80)

for input_text, task_name in tasks:
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50)

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"\nTask: {task_name}")
    print(f"Input: {input_text}")
    print(f"Output: {output_text}")
    print("-" * 80)
```

**Key Insight**: Encoder-decoder models excel at seq-to-seq tasks where input and output have different structures or lengths (translation, summarization, etc.).

---

## 7. Understanding Encoder Output Reuse

```python
# Key efficiency: Encoder processes input ONCE, decoder reuses it

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_text = "translate English to German: The cat is black"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

print("Encoding Phase:")
print("=" * 80)

# Encoder runs once
with torch.no_grad():
    encoder_outputs = model.encoder(input_ids)

print(f"Input: {input_text}")
print(f"Encoder processed {input_ids.shape[1]} tokens")
print(f"Encoder output shape: {encoder_outputs.last_hidden_state.shape}")
print(f"Memory allocated: {encoder_outputs.last_hidden_state.element_size() * encoder_outputs.last_hidden_state.nelement() / 1024:.2f} KB")

print("\n\nDecoding Phase:")
print("=" * 80)
print("Decoder runs multiple times (once per output token)")
print("But encoder outputs are REUSED - no need to re-encode!")

# Simulate decoder steps
decoder_input = torch.tensor([[model.config.decoder_start_token_id]])

for i in range(5):
    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input
        )
        next_token = torch.argmax(outputs.logits[0, -1, :]).item()
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]])], dim=1)

    print(f"Decoder step {i+1}: Generated {decoder_input.shape[1]} tokens, using same encoder outputs")

print("\n" + "=" * 80)
print("This reuse is a key efficiency of encoder-decoder architecture!")
```

**Key Insight**: The encoder processes input once, creating representations that the decoder reuses throughout generation. This is more efficient than decoder-only models for seq-to-seq tasks.

---

## 8. BART Model Example - Bidirectional Encoder

```python
from transformers import BartTokenizer, BartForConditionalGeneration

# BART: Bidirectional Encoder, Autoregressive Decoder
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

model.eval()

# BART is great for text generation and summarization
text = "Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data. These systems improve their performance over time without being explicitly programmed."

input_ids = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids

print(f"Input text ({input_ids.shape[1]} tokens):")
print(text)
print("\n" + "=" * 80)

# Generate summary
with torch.no_grad():
    summary_ids = model.generate(
        input_ids,
        max_length=30,
        min_length=10,
        num_beams=4,
        early_stopping=True
    )

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"\nSummary: {summary}")
print("\n" + "=" * 80)

print("\nBART Architecture:")
print(f"- Encoder layers: {model.config.encoder_layers}")
print(f"- Decoder layers: {model.config.decoder_layers}")
print(f"- Hidden size: {model.config.d_model}")
print(f"- Vocabulary size: {model.config.vocab_size}")
```

**Key Insight**: BART combines BERT-like bidirectional encoding with GPT-like autoregressive decoding, making it powerful for generation tasks.

---

## 9. Beam Search in Encoder-Decoder Models

```python
# Beam search explores multiple candidate sequences

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_text = "translate English to French: I love machine learning"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

print(f"Input: {input_text}\n")
print("=" * 80)

# Compare different decoding strategies
strategies = [
    {"name": "Greedy", "params": {"num_beams": 1, "do_sample": False}},
    {"name": "Beam Search (2 beams)", "params": {"num_beams": 2}},
    {"name": "Beam Search (4 beams)", "params": {"num_beams": 4}},
    {"name": "Beam Search (8 beams)", "params": {"num_beams": 8}},
]

for strategy in strategies:
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=30,
            **strategy["params"]
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n{strategy['name']}:")
    print(f"  Output: {output}")

print("\n" + "=" * 80)
print("Beam search maintains multiple hypotheses, often producing better translations")
```

**Key Insight**: Encoder-decoder models commonly use beam search, which explores multiple possible output sequences in parallel to find high-quality results.

---

## 10. Forced Decoding and Teacher Forcing

```python
# Teacher forcing: feeding ground truth during training instead of predictions

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Input and target
input_text = "translate English to French: Hello"
target_text = "Bonjour"

# Tokenize
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
target_ids = tokenizer(target_text, return_tensors="pt").input_ids

print("Understanding Teacher Forcing:\n")
print("=" * 80)

print(f"Input: {input_text}")
print(f"Target: {target_text}")

# During training: use target tokens as decoder input (teacher forcing)
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        decoder_input_ids=target_ids
    )

print(f"\nWith teacher forcing:")
print(f"- Encoder processes: '{input_text}'")
print(f"- Decoder receives ground truth: '{target_text}'")
print(f"- Model predicts next token for each position")
print(f"- Logits shape: {outputs.logits.shape}")
print(f"  (batch_size={outputs.logits.shape[0]}, seq_len={outputs.logits.shape[1]}, vocab_size={outputs.logits.shape[2]})")

# During inference: decoder uses its own predictions
with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=10)

generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(f"\nDuring inference (autoregressive):")
print(f"- Decoder generates: '{generated_text}'")
print(f"- Each token depends on previous generated tokens")

print("\n" + "=" * 80)
print("Teacher forcing speeds up training but creates exposure bias")
```

**Key Insight**: During training, encoder-decoder models use teacher forcing (feeding ground truth). During inference, they generate autoregressively.

---

## 11. Extracting Cross-Attention Weights

```python
# Visualize how decoder attends to encoder outputs

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_text = "translate English to German: Good morning"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate with attention outputs
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_length=15,
        output_attentions=True,
        return_dict_in_generate=True,
        output_scores=True
    )

print("Cross-Attention Analysis:\n")
print("=" * 80)

input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
output_tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0])

print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {output_tokens}")

print("\nCross-attention allows decoder to 'look at' encoder outputs")
print("Each decoder token attends to all encoder tokens with different weights")
print("High attention weight = decoder focuses on that encoder token")

print("\n" + "=" * 80)
print("In a translation task:")
print("- When generating 'Guten', decoder attends strongly to 'Good'")
print("- When generating 'Morgen', decoder attends strongly to 'morning'")
print("This alignment is learned automatically!")
```

**Key Insight**: Cross-attention creates soft alignments between input and output, which is crucial for tasks like translation where word order may differ.

---

## 12. When to Use Encoder-Decoder vs Decoder-Only

```python
print("ENCODER-DECODER MODELS (T5, BART, Transformer):")
print("=" * 80)
print("Best for:")
print("  ✓ Machine Translation")
print("  ✓ Text Summarization")
print("  ✓ Question Answering (with context)")
print("  ✓ Text-to-Text tasks with different input/output structures")
print("\nAdvantages:")
print("  + Bidirectional encoder understands full input context")
print("  + Cross-attention enables explicit input-output alignment")
print("  + Efficient for tasks where input is processed once")
print("\nExamples: T5, BART, MarianMT, Pegasus\n")

print("\n" + "=" * 80)

print("\nDECODER-ONLY MODELS (GPT):")
print("=" * 80)
print("Best for:")
print("  ✓ Text Generation")
print("  ✓ Code Generation")
print("  ✓ Conversational AI")
print("  ✓ Tasks where prompt and completion are continuous")
print("\nAdvantages:")
print("  + Simpler architecture")
print("  + Scales well to very large sizes")
print("  + Flexible few-shot learning via prompting")
print("\nExamples: GPT-2, GPT-3, GPT-4, Llama\n")

print("\n" + "=" * 80)
print("\nKey Architectural Difference:")
print("- Encoder-Decoder: Bidirectional encoding + Cross-attention")
print("- Decoder-Only: Causal attention only")
```

**Key Insight**: Choose encoder-decoder for seq-to-seq tasks with distinct input/output. Choose decoder-only for open-ended generation and prompting.

---

## Summary

This notebook demonstrated:

1. **Two-Part Architecture**: Separate encoder and decoder components
2. **Bidirectional Encoding**: Encoder sees full input context
3. **Cross-Attention**: Decoder attends to encoder outputs
4. **Three Attention Types**: Self-attention in encoder, self-attention in decoder, cross-attention
5. **Encoder Reuse**: Encoder runs once, decoder reuses outputs
6. **Seq-to-Seq Tasks**: Translation, summarization, Q&A
7. **Teacher Forcing**: Training vs inference behavior
8. **Beam Search**: Exploring multiple output hypotheses
9. **Task Versatility**: One architecture, many tasks (T5)
10. **Cross-Attention Alignment**: Soft alignment between input/output

Encoder-decoder models excel at tasks requiring transformation between different structures or languages!
