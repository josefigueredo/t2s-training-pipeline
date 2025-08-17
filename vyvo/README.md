# VyvoTTS with SNAC Codec Training

This folder contains scripts for training and inference using VyvoTTS, a modern language model approach to TTS that uses SNAC (Scalable Neural Audio Codec) for high-quality audio generation.

## Overview

VyvoTTS represents a paradigm shift in TTS:

1. **Language Model Approach**: Treats TTS as a text-to-audio-token sequence task
2. **SNAC Codec**: Hierarchical neural audio codec for efficient audio representation
3. **LoRA Fine-tuning**: Parameter-efficient adaptation of pre-trained models

## VyvoTTS Architecture

```mermaid
graph TB
    subgraph "SNAC Encoding"
        A[Audio WAV<br/>24kHz] --> B[SNAC Encoder]
        B --> C[Layer 1<br/>75Hz, 4096 codes]
        B --> D[Layer 2<br/>150Hz, 4096 codes]
        B --> E[Layer 3<br/>300Hz, 4096 codes]
        C --> F[7-way Packing]
        D --> F
        E --> F
        F --> G[Token Sequence]
    end
    
    subgraph "Language Model Training"
        H[Text Input] --> I[Tokenizer]
        I --> J[Text Tokens]
        G --> K[Audio Tokens<br/>+AUDIO_BASE offset]
        J --> L[Combined Sequence]
        K --> L
        L --> M[Qwen3-0.6B<br/>4-bit Quantized]
        M --> N[LoRA Adapters<br/>Rank 16]
    end
    
    subgraph "Inference"
        O[Input Text] --> P[Generate Tokens]
        P --> Q[Unpack 7-way]
        Q --> R[SNAC Decoder]
        R --> S[Audio Output]
    end
    
    style A fill:#e3f2fd
    style G fill:#fff9c4
    style S fill:#c8e6c9
```

## Training Pipeline Sequence

```mermaid
sequenceDiagram
    participant User
    participant PreProc as precompute_snac.py
    participant FineTune as vyvo_finetune.py
    participant Infer as vyvo_infer.py
    participant SNAC as SNAC Codec
    participant Model as Qwen3 Model
    participant GPU
    
    User->>PreProc: Start preprocessing
    PreProc->>PreProc: Load metadata.csv
    
    loop For each audio file
        PreProc->>PreProc: Load WAV (24kHz)
        PreProc->>SNAC: Encode to 3 layers
        SNAC->>SNAC: Generate hierarchical codes
        SNAC->>PreProc: Return token arrays
        PreProc->>PreProc: 7-way packing
        PreProc->>PreProc: Save to train_snac.jsonl
    end
    
    User->>FineTune: Start fine-tuning
    FineTune->>Model: Load base model (4-bit)
    FineTune->>Model: Add LoRA adapters
    FineTune->>GPU: Allocate memory (~6GB)
    
    loop Training Loop (4000 steps)
        FineTune->>FineTune: Load batch
        FineTune->>Model: Forward pass
        Model->>GPU: Compute loss
        GPU->>Model: Backward pass
        Model->>Model: Update LoRA weights
        
        alt Every 500 steps
            Model->>FineTune: Save checkpoint
        end
    end
    
    FineTune->>FineTune: Save vyvo_qwen3_lora/
    
    User->>Infer: Generate speech "Hello"
    Infer->>Model: Load model + LoRA
    Infer->>Model: Tokenize text
    Model->>Model: Generate audio tokens
    Infer->>Infer: Unpack 7-way format
    Infer->>SNAC: Decode to audio
    SNAC->>User: output.wav
```

## SNAC Codec Processing

```mermaid
graph LR
    subgraph "Encoding Process"
        A[Audio Input] --> B[Convolutional Encoder]
        B --> C[Residual VQ Layer 1<br/>4096 codes @ 75Hz]
        B --> D[Residual VQ Layer 2<br/>4096 codes @ 150Hz]
        B --> E[Residual VQ Layer 3<br/>4096 codes @ 300Hz]
    end
    
    subgraph "7-Way Packing"
        C --> F[L1: c1, c2, c3...]
        D --> G[L2: c1, c2, c3, c4, c5, c6...]
        E --> H[L3: c1, c2, c3, ... c12...]
        F --> I[Interleaved:<br/>L1_c1, L2_c1, L3_c1,<br/>L3_c2, L2_c2, L3_c3,<br/>L3_c4...]
    end
    
    subgraph "Decoding Process"
        I --> J[Unpack Layers]
        J --> K[Reconstruct Hierarchical Codes]
        K --> L[Convolutional Decoder]
        L --> M[Audio Output]
    end
    
    style A fill:#e1f5fe
    style I fill:#fff3e0
    style M fill:#e8f5e9
```

## Token Sequence Format

```mermaid
graph LR
    A["[SOH]"] --> B[Text Tokens]
    B --> C["[EOT]"]
    C --> D["[EOH]"]
    D --> E["[SOS_AUDIO]"]
    E --> F[Audio Tokens<br/>+12000 offset]
    F --> G["[EOS_AUDIO]"]
    
    style A fill:#ffebee
    style C fill:#ffebee
    style D fill:#ffebee
    style E fill:#e3f2fd
    style G fill:#e3f2fd
    style B fill:#e8f5e9
    style F fill:#fff9c4
```

## Files

- `precompute_snac.py` - Convert audio to SNAC token sequences
- `vyvo_finetune.py` - Fine-tune VyvoTTS with LoRA adapters
- `vyvo_infer.py` - Generate speech from text using trained model

## Setup

Install dependencies:

```bash
uv sync
```

Ensure you have recorded audio data in the `data/` folder with `metadata.csv`.

## Workflow

### 1. Precompute SNAC Codes

Convert your WAV files to SNAC token sequences:

```bash
uv run precompute_snac.py
```

**Input**: `data/metadata.csv` and `data/wavs/*.wav`

**Output**: `data/train_snac.jsonl` with format:

```json
{"text": "Hello world", "codes": [12045, 8192, 16384, ...]}
```

**Process**:

- Loads 24kHz mono WAV files
- Encodes to 3-layer SNAC representation
- Packs to 7-way token stream for language modeling

### 2. Fine-tune VyvoTTS

Adapt the pre-trained VyvoTTS model to your voice:

```bash
uv run vyvo_finetune.py
```

**Base Model**: `Vyvo/VyvoTTS-v0-Qwen3-0.6B`

**Method**: LoRA (Low-Rank Adaptation) fine-tuning

**Output**: `vyvo_qwen3_lora/` directory with adapters

**Training Details**:

- 4000 training steps
- 4-bit quantization for memory efficiency
- LoRA rank 16, alpha 32
- Learning rate 2e-4

### 3. Generate Speech

Use the fine-tuned model for inference:

```bash
uv run vyvo_infer.py "Hello, this is a test." output.wav
```

**Models Required**:

- Base: `Vyvo/VyvoTTS-v0-Qwen3-0.6B`
- SNAC: `hubertsiuzdak/snac_24khz`
- Fine-tuned adapters (if available)

## LoRA Adaptation Flow

```mermaid
graph TD
    subgraph "Base Model"
        A[Qwen3-0.6B<br/>Frozen Weights] --> B[Q, K, V Matrices]
    end
    
    subgraph "LoRA Adapters"
        B --> C[Low-Rank Decomposition]
        C --> D[Matrix A<br/>hidden_dim x rank]
        C --> E[Matrix B<br/>rank x hidden_dim]
        D --> F[LoRA Delta]
        E --> F
    end
    
    subgraph "Fine-tuning"
        F --> G[Scale by alpha/rank]
        G --> H[Add to Base Weights]
        H --> I[Updated Attention]
    end
    
    subgraph "Memory Savings"
        J["4-bit Quantization<br/>75% reduction"] --> K["LoRA Only<br/>~0.5% params"]
        K --> L["Total: ~1.5GB model<br/>+ 6MB adapters"]
    end
    
    style A fill:#e3f2fd
    style F fill:#fff9c4
    style L fill:#c8e6c9
```

## Technical Details

### SNAC Codec

SNAC uses a hierarchical approach with 3 layers:

- **Layer 1**: 4096 codes, frame rate 75Hz
- **Layer 2**: 4096 codes, frame rate 150Hz  
- **Layer 3**: 4096 codes, frame rate 300Hz

The 7-way packing scheme interleaves codes for language model training.

### Token Format

The language model processes sequences like:

```plaintext
[SOH] text tokens [EOT] [EOH] [SOS_AUDIO] audio tokens [EOS_AUDIO]
```

Where audio tokens are SNAC codes offset by `AUDIO_BASE`.

### Memory Optimization

For 8GB GPU constraints:

- **4-bit quantization**: Reduces model memory by ~75%
- **LoRA adapters**: Only fine-tune small parameter subset
- **Float16 precision**: Halves activation memory

## Model Inference

Load and use trained models:

```python
import torch
from unsloth import FastLanguageModel
from snac import SNAC

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    "vyvo_qwen3_lora",  # Your fine-tuned adapters
    dtype=torch.float16,
    load_in_4bit=True
)

# Load SNAC decoder
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")

# Generate audio tokens
tokens = model.generate(...)

# Decode to audio
audio = snac.decode(codes)
```

## Performance Tips

### Training Optimization

1. **Batch Size**: Start with 1, increase if memory allows
2. **Gradient Accumulation**: Simulate larger batches
3. **Checkpoint Frequency**: Save every 500 steps
4. **Mixed Precision**: Use float16 for speed

### Inference Optimization

1. **Temperature**: 0.6-0.8 for natural speech
2. **Top-p**: 0.95 for diversity control
3. **Repetition Penalty**: 1.1 to avoid loops
4. **Max Tokens**: 800 for typical sentences

## Troubleshooting

**CUDA OOM during training**:

- Enable 4-bit quantization
- Reduce LoRA rank to 8
- Use gradient checkpointing

**Poor audio quality**:

- Check SNAC encoding quality
- Verify 24kHz input format
- Increase training steps

**Repetitive output**:

- Adjust repetition penalty
- Increase temperature
- Check training data diversity

**Slow inference**:

- Use smaller max_tokens
- Enable use_cache=True
- Consider model quantization

## Advanced Usage

### Custom LoRA Configuration

Modify `vyvo_finetune.py` for different LoRA settings:

```python
peft_cfg = LoraConfig(
    r=32,              # Higher rank = more parameters
    lora_alpha=64,     # Scaling factor
    lora_dropout=0.1,  # Regularization
    target_modules=[...] # Which layers to adapt
)
```

### Multi-GPU Training

For faster training with multiple GPUs:

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```
