# %% [markdown]
# 🧩 Install minimal deps (fast on Colab T4)
!pip -q install "transformers>=4.41.0" "datasets>=2.19.0" "accelerate>=0.31.0" peft trl gradio pillow einops --upgrade

import os, random, re, math
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from PIL import Image

import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# BLOCK 2

# %% [markdown]
# ⚙️ Config (adjust here if needed)

# Small open VLM that works with TRL/PEFT on consumer GPUs:
MODEL_ID = "HuggingFaceTB/SmolVLM-1.7B-Instruct"  # swap if needed
OUTPUT_DIR = "smolvlm-era-vqa"
SEED = 42

# Training knobs (safe for a T4)
EPOCHS = 3
PER_DEVICE_BATCH = 2           # small; we use grad accumulation
GRAD_ACCUM = 8                 # effective batch = 16
LR = 1e-4                      # LoRA params only
BF16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8  # bfloat16 ok on A100/L4 etc.
FP16 = not BF16 and torch.cuda.is_available()  # fallback

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 🧠 Load model + processor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32),
    device_map="auto"
)

# Disable KV cache during training, makes life easier for TRL
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

print("Loaded:", MODEL_ID)

# BLOCK 3
# %% [markdown]
# 📚 Load dataset & convert to VQA-style conversations

raw = load_dataset("biglam/dating-historical-color-images")  # single 'train' split
feat = raw["train"].features
label_names = feat["label"].names  # e.g. ['1930s','1940s','1950s','1960s','1970s']

# Stratified 80/10/10 split
tmp = raw["train"].train_test_split(test_size=0.2, stratify_by_column="label", seed=SEED)
valtest = tmp["test"].train_test_split(test_size=0.5, stratify_by_column="label", seed=SEED)
ds = DatasetDict({"train": tmp["train"], "validation": valtest["train"], "test": valtest["test"]})

ERA_LABELS = label_names
LABEL_LIST = ", ".join(ERA_LABELS)

TEMPLATES = [
    f"Which decade is this image from? Answer with one of: {LABEL_LIST}.",
    f"What decade best fits this photo? Choose only from: {LABEL_LIST}.",
    f"Identify the decade of this image. Options: {LABEL_LIST}.",
]
SYSTEM_MSG = ("You are a visual historian. "
              "Look at the image and answer with exactly one label from the options provided.")

def to_messages(example, k_paraphrases=2):
    label_text = ERA_LABELS[example["label"]]
    # Pick up to k templates per example (randomized for variety)
    ks = min(k_paraphrases, len(TEMPLATES))
    chosen = random.sample(TEMPLATES, k=ks)
    convs = []
    for tpl in chosen:
        convs.append({
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
                {"role": "user",   "content": [{"type": "image", "image": example["image"]},
                                               {"type": "text",  "text": tpl}]},
                {"role": "assistant","content":[{"type": "text", "text": label_text}]}
            ]
        })
    return {"conversations": convs}

def expand_split(split, k=2):
    mapped = split.map(lambda ex: to_messages(ex, k_paraphrases=k), remove_columns=split.column_names)
    # flatten conversations
    flat = []
    for row in mapped:
        flat.extend(row["conversations"])
    return Dataset.from_list(flat)

train_messages = expand_split(ds["train"], k=2)
val_messages   = expand_split(ds["validation"], k=1)
test_messages  = expand_split(ds["test"], k=1)

print(train_messages[:1])
print("Sizes:", len(train_messages), len(val_messages), len(test_messages))

# BLOCK 4
# %% [markdown]
# 🧩 Apply LoRA (QLoRA-ish params) and wire up the trainer

peft_config = LoraConfig(
    r=8, lora_alpha=8, lora_dropout=0.1,
    target_modules=["down_proj","o_proj","k_proj","q_proj","gate_proj","up_proj","v_proj"],
    use_dora=True,                    # more stable updates
    init_lora_weights="gaussian",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- collator: turns "messages" into model-ready tensors (input_ids, pixel values, labels) ---
image_token_id = processor.tokenizer.convert_tokens_to_ids(
    processor.tokenizer.image_token if hasattr(processor.tokenizer, "image_token") else "<image>"
)

def collate_fn(examples: List[Dict[str, Any]]):
    texts = []
    imgs_nested = []
    for ex in examples:
        msgs = ex["messages"]
        user = next(m for m in msgs if m["role"] == "user")
        img = user["content"][0]["image"]
        if isinstance(img, Image.Image):
            if img.mode != "RGB": img = img.convert("RGB")
        imgs_nested.append([img])  # nested list matches many VLM processors

        sys_txt = next((m["content"][0]["text"] for m in msgs if m["role"] == "system"), "")
        usr_txt = next(c["text"] for c in user["content"] if c["type"] == "text")
        asst_txt = next(m["content"][0]["text"] for m in msgs if m["role"] == "assistant")

        # Simple serialized dialogue; processor handles special tokens
        texts.append(f"{sys_txt}\nUser: {usr_txt}\nAssistant: {asst_txt}")

    batch = processor(text=texts, images=imgs_nested, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    # ignore loss on padding + image tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100
    if isinstance(image_token_id, int):
        labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return {k: v.to(model.device) for k, v in batch.items()}

# Training args
args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    bf16=BF16,
    fp16=FP16,
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    report_to=[],
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_messages,
    eval_dataset=val_messages,
    data_collator=collate_fn,
    peft_config=None,                        # already applied
    processing_class=processor.tokenizer,    # needed by TRL for text ops
)

trainer.train()


# BLOCK 5 
# %% [markdown]
# 📊 Constrained decoding + accuracy on a test sample

VALID = [s.lower() for s in ERA_LABELS]
def normalize_to_label(s: str) -> str:
    s = s.strip().lower()
    for lbl in VALID:
        if s.startswith(lbl) or lbl in s:
            return lbl
    toks = set(re.findall(r"[a-z0-9]+", s))
    best, score = "unknown", -1
    for lbl in VALID:
        ov = len(toks & set(lbl.split()))
        if ov > score: best, score = lbl, ov
    return best

@torch.no_grad()
def generate_decade(img, question_template=None, max_new_tokens=4):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.mode != "RGB": img = img.convert("RGB")
    q = question_template or TEMPLATES[0]
    inputs = processor(text=q, images=[[img]], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    txt = processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return normalize_to_label(txt)

from tqdm.auto import tqdm
subset = test_messages.select(range(min(100, len(test_messages))))
correct = 0
for ex in tqdm(subset):
    user = next(m for m in ex["messages"] if m["role"] == "user")
    img = user["content"][0]["image"]
    gold = next(m["content"][0]["text"] for m in ex["messages"] if m["role"] == "assistant").lower()
    pred = generate_decade(img)
    correct += int(pred == gold)

acc = correct / len(subset)
print(f"Sample test accuracy (n={len(subset)}): {acc:.3f}")


# BLOCK 6
# %% [markdown]
# 🧪 Try it live (upload an image)

import gradio as gr

def predict_ui(img):
    img = Image.fromarray(img) if not isinstance(img, Image.Image) else img
    pred = generate_decade(img)
    pretty = next((lbl for lbl in ERA_LABELS if lbl.lower() == pred), "unknown")
    return pretty

demo = gr.Interface(
    fn=predict_ui,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Decade VQA (SmolVLM + LoRA)",
    description=f"Answers with one of: {', '.join(ERA_LABELS)}"
)
demo.launch(share=False)


# here are two drop-in Colab cells for a zero-shot baseline. Put them right after Cell 3 (dataset mapping) and before the LoRA training cell. They reload a fresh base VLM, run constrained VQA, and report accuracy on a test sample.

# %% [markdown]
# 🧪 Zero-shot baseline (no training) — load a fresh base VLM

ZS_MODEL_ID = MODEL_ID  # reuse the same model id as config; swap here if you want to compare models

base_processor = AutoProcessor.from_pretrained(ZS_MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    ZS_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if BF16 else (torch.float16 if FP16 else torch.float32),
    device_map="auto"
)
if hasattr(base_model.config, "use_cache"):
    base_model.config.use_cache = True  # fine for inference
base_model.eval()

# Constrain free-text to a valid label
VALID = [s.lower() for s in ERA_LABELS]

def zs_normalize_to_label(s: str) -> str:
    s = s.strip().lower()
    for lbl in VALID:
        if s.startswith(lbl) or lbl in s:
            return lbl
    # simple token-overlap fallback
    import re
    toks = set(re.findall(r"[a-z0-9]+", s))
    best, score = "unknown", -1
    for lbl in VALID:
        ov = len(toks & set(lbl.split()))
        if ov > score:
            best, score = lbl, ov
    return best

@torch.no_grad()
def zs_generate_decade(img, question_template=None, max_new_tokens=4):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    q = question_template or TEMPLATES[0]   # uses the templates defined in Cell 3
    inputs = base_processor(text=q, images=[[img]], return_tensors="pt").to(base_model.device)
    out = base_model.generate(**inputs, max_new_tokens=max_new_tokens)
    txt = base_processor.tokenizer.decode(out[0], skip_special_tokens=True)
    return zs_normalize_to_label(txt)


# %% [markdown]
# 📊 Zero-shot accuracy (and robustness to prompt paraphrases)

from tqdm.auto import tqdm

def eval_zero_shot(samples, use_all_prompts=False):
    correct = 0
    for ex in tqdm(samples):
        user = next(m for m in ex["messages"] if m["role"] == "user")
        img = user["content"][0]["image"]
        gold = next(m["content"][0]["text"] for m in ex["messages"] if m["role"] == "assistant").lower()
        if use_all_prompts:
            # vote across all question templates, pick majority (ties → first)
            preds = [zs_generate_decade(img, t) for t in TEMPLATES]
            pred = max(set(preds), key=preds.count)
        else:
            pred = zs_generate_decade(img)
        correct += int(pred == gold)
    return correct / len(samples)

# Evaluate on a manageable subset first (speeds up on T4)
subset = test_messages.select(range(min(100, len(test_messages))))
acc_zs = eval_zero_shot(subset, use_all_prompts=False)
print(f"Zero-shot sample accuracy (n={len(subset)}): {acc_zs:.3f}")

# Optional: paraphrase-robust accuracy (slower)
acc_zs_robust = eval_zero_shot(subset, use_all_prompts=True)
print(f"Zero-shot (prompt-ensemble) accuracy: {acc_zs_robust:.3f}")
