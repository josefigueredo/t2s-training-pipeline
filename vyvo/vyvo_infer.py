# scripts/vyvo_infer.py
"""
Generate speech WAV from text using Vyvo/VyvoTTS-v0-Qwen3-0.6B with Unsloth + SNAC.

Usage:
  uv run scripts/vyvo_infer.py "Hello Jose, testing VyvoTTS." out.wav

Notes:
  - Uses float16 on GPU to fit 8 GB. For very tight memory, set load_in_4bit=True.
  - Expects SNAC 24 kHz decoder.
"""
import sys
import torch
import soundfile as sf
from unsloth import FastLanguageModel
from snac import SNAC

TEXT = sys.argv[1] if len(sys.argv) > 1 else "Hello from VyvoTTS."
OUT = sys.argv[2] if len(sys.argv) > 2 else "out.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load base LM
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Vyvo/VyvoTTS-v0-Qwen3-0.6B",
    max_seq_length=4096,
    dtype=torch.float16 if DEVICE == "cuda" else None,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)
model.to(DEVICE)

# SNAC vocoder (24 kHz)
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(DEVICE).eval()

# Token constants (from model card)
TOKENISER_LEN = 151669
SOT, EOT = 151643, 151645
SOH, EOH = TOKENISER_LEN + 3, TOKENISER_LEN + 4
PAD = TOKENISER_LEN + 7
AUDIO_BASE = TOKENISER_LEN + 10
EOS_AUDIO = TOKENISER_LEN + 2
SOS_AUDIO = TOKENISER_LEN + 1


def build_ids(prompt: str):
    start = torch.tensor([[SOH]], dtype=torch.long)
    end = torch.tensor([[EOT, EOH]], dtype=torch.long)
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    mod = torch.cat([start, ids, end], dim=1)
    # left-pad with PAD to keep attention mask simple
    padlen = mod.shape[1]
    padded = torch.cat([torch.full((1, padlen), PAD, dtype=torch.long), mod], dim=1)
    attn = torch.cat(
        [
            torch.zeros((1, padlen), dtype=torch.long),
            torch.ones((1, mod.shape[1]), dtype=torch.long),
        ],
        dim=1,
    )
    return padded.to(DEVICE), attn.to(DEVICE)


inp, attn = build_ids(TEXT)

with torch.no_grad():
    gen = model.generate(
        input_ids=inp,
        attention_mask=attn,
        max_new_tokens=800,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=EOS_AUDIO,
        use_cache=True,
    )

# Extract audio token codes after SOS_AUDIO
tok = gen
hit = (tok == SOS_AUDIO).nonzero(as_tuple=True)
if len(hit[1]) > 0:
    idx = hit[1][-1].item()
    tok = tok[:, idx + 1 :]

# Remove EOS_AUDIO, convert to code indices
tok = tok[tok != EOS_AUDIO]
codes = (tok - AUDIO_BASE).tolist()

# Unpack to SNAC's 3-layer format (7-way packing per frame)
layer1, layer2, layer3 = [], [], []
for i in range((len(codes) + 1) // 7):
    j = 7 * i
    if j + 6 >= len(codes):
        break
    layer1.append(codes[j + 0])
    layer2.append(codes[j + 1] - 4096)
    layer3.append(codes[j + 2] - 2 * 4096)
    layer3.append(codes[j + 3] - 3 * 4096)
    layer2.append(codes[j + 4] - 4 * 4096)
    layer3.append(codes[j + 5] - 5 * 4096)
    layer3.append(codes[j + 6] - 6 * 4096)

codes_t = [
    torch.tensor(layer1, device=DEVICE).unsqueeze(0),
    torch.tensor(layer2, device=DEVICE).unsqueeze(0),
    torch.tensor(layer3, device=DEVICE).unsqueeze(0),
]

with torch.no_grad():
    audio = snac.decode(codes_t).detach().squeeze().float().cpu().numpy()

sf.write(OUT, audio, 24000)
print(f"✅ Wrote {OUT}")
