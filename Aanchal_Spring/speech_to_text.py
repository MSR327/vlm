import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BATCH_SIZE = 20
EPOCHS = 40

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

LATEST_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
BEST_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")

torch.backends.cudnn.benchmark = True

# MODELS
wav_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").to(device)

# Freeze LLM
for p in llm.parameters():
    p.requires_grad = False

# Partial Wav2Vec (only last layers)
for name, param in wav_model.named_parameters():
    if "encoder.layers.10" in name or "encoder.layers.11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

dtype = llm.dtype

# PROJECTOR
class Projector(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=llm.config.hidden_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1024),
            torch.nn.GELU(), # GELU is smoother than ReLU for LLMs
            torch.nn.LayerNorm(1024), # CRITICAL: Keeps weights stable
            torch.nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.net(x)

projector = Projector().to(device).to(dtype)

# AUDIO CACHE
audio_cache = {}

def load_audio(path):
    try:
        waveform, sr = torchaudio.load(path)

        if waveform is None or waveform.numel() == 0:
            return None

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        waveform = waveform.squeeze(0)

        # filter very short audio
        if waveform.shape[0] < 1000:
            return None

        return waveform

    except Exception:
        return None

# DATA
def load_dataset(metadata_csv, audio_folder2):
    data = []
    df = pd.read_csv(metadata_csv)
    for _, row in df.iterrows():
        audio_path = os.path.join(audio_folder2, row["audio_filename"])
        if os.path.exists(audio_path):
            data.append({
                "audio": audio_path,
                "text": f"ADAS_CMD: {row['sentence']}"
            })
    return data

# CHECKPOINT
def save_checkpoint(epoch, optimizer, best_loss):
    torch.save({
        "epoch": epoch,
        "projector": projector.state_dict(),
        "wav_model": wav_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss
    }, LATEST_PATH)

def save_best(best_loss):
    torch.save({
        "projector": projector.state_dict(),
        "wav_model": wav_model.state_dict(),
        "best_loss": best_loss
    }, BEST_PATH)

def load_checkpoint(optimizer):
    if os.path.exists(LATEST_PATH):
        checkpoint = torch.load(LATEST_PATH, map_location=device)
        projector.load_state_dict(checkpoint["projector"])
        wav_model.load_state_dict(checkpoint["wav_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Resuming from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"] + 1, checkpoint["best_loss"]
    return 0, float("inf")

# FORWARD (BATCH)
def forward_batch(batch):
    waveforms = []
    valid_texts = []

    for x in batch:
        w = load_audio(x["audio"])
        if w is None:
            continue
        waveforms.append(w.numpy()) # Convert to numpy for the processor
        valid_texts.append(x["text"])

    if len(waveforms) == 0:
        return None

    current_batch_size = len(waveforms)
    inputs = wav_processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True).to(device)

    audio_out = wav_model(**inputs)
    audio_emb = audio_out.last_hidden_state.mean(dim=1)
    proj = projector(audio_emb.to(dtype)).unsqueeze(1).repeat(1, 4, 1)
    proj = proj 

    # Tokenize the structured output
    tokens = tokenizer(valid_texts, return_tensors="pt", padding=True).to(device)
    text_embeds = llm.get_input_embeddings()(tokens.input_ids)

    # Inputs = [Audio Projector] + [Structured Text Embeds]
    inputs_embeds = torch.cat([proj, text_embeds], dim=1)

    # Labels: Mask the Audio Projector part (-100) and only train on the Text part
    pad_labels = torch.full((current_batch_size, 4), -100).to(device)
    labels = torch.cat([pad_labels, tokens.input_ids], dim=1)

    # Ensure PAD tokens in text don't contribute to loss
    labels[labels == tokenizer.pad_token_id] = -100

    audio_mask = torch.ones((current_batch_size, 4)).to(device)
    attention_mask = torch.cat([audio_mask, tokens.attention_mask], dim=1)

    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels
    )

    return outputs.loss
# GENERATE
def generate_samples(data, num_samples=5):
    print("\n Structured ADAS Predictions:\n")
    samples = np.random.choice(data, min(num_samples, len(data)), replace=False)
    
    for i in range(num_samples):
        sample = samples[i]
        waveform = load_audio(sample["audio"])
        
        inputs = wav_processor(waveform, sampling_rate=16000, return_tensors="pt").to(device)

        with torch.no_grad():
            audio_out = wav_model(**inputs).last_hidden_state
            audio_emb = audio_out.mean(dim=1) 
            
            # 3. MATCH TRAINING: 4 tokens and 0.1 scale
            proj = projector(audio_emb.to(dtype)).unsqueeze(1).repeat(1, 4, 1)
            proj = proj 
            # We use an empty prompt or a trigger word like "Action:"
            # because the audio embeddings should now lead directly to the INTENT
            # 4. Use the NEW Unique Trigger "ADAS_CMD:" 
            prompt = tokenizer("ADAS_CMD:", return_tensors="pt").to(device)
            prompt_embeds = llm.get_input_embeddings()(prompt.input_ids)
            
            inputs_embeds = torch.cat([proj, prompt_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2]).to(device)

            outputs = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=25, # Increased slightly for structured format
                do_sample=False,
                repetition_penalty=1.5, # Encourage more diverse outputs
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        clean_pred = pred.replace("ADAS_CMD:", "").strip()
        
        print(f"Ground Truth : {sample['text'].replace('ADAS_CMD:', '').strip()}")
        print(f"Model Predict: {clean_pred}")
        print("-" * 50)
# TRAIN
def train(data):
    optimizer = torch.optim.Adam(
        list(projector.parameters()) + list(wav_model.parameters()),
        lr=5e-6
    )

    scaler = torch.cuda.amp.GradScaler()

    start_epoch, best_loss = load_checkpoint(optimizer)

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0

        for i in tqdm(range(0, len(data), BATCH_SIZE)):
            batch = data[i:i+BATCH_SIZE]

            try:
                with torch.cuda.amp.autocast():
                    loss = forward_batch(batch)
                if loss is None:    
                    continue

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                
                if(i!=0 and  i% 1000 == 0):
                    print(f"Batch {i} Loss: {total_loss/i:.4f}")

            except Exception as e:
                print("Error:", e)
                continue

        avg_loss = total_loss / len(data)
        print(f"\nEpoch {epoch+1} Loss: {avg_loss}")

        # Save latest
        save_checkpoint(epoch, optimizer, best_loss)

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_best(best_loss)
            print("Saved BEST model")

        # Preview outputs
        generate_samples(data)

# RUN
if __name__ == "__main__":
    data = load_dataset(
        "dataset2/final_speech_dataset.csv",
        "dataset2/audio_files"
    )

    print("Total samples:", len(data))

    tokenizer.pad_token = tokenizer.eos_token
    llm.config.pad_token_id = llm.config.eos_token_id

    train(data)