import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm
from jiwer import wer, cer
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIG
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BEST_PATH = "checkpoints/best.pt"

OUTPUT_DIR = "test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRED_FILE = os.path.join(OUTPUT_DIR, "predictions.txt")
METRIC_FILE = os.path.join(OUTPUT_DIR, "metrics.txt")

# MODELS
wav_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").to(device)

tokenizer.pad_token = tokenizer.eos_token
llm.config.pad_token_id = llm.config.eos_token_id

# PROJECTOR
class Projector(torch.nn.Module):
    def __init__(self, in_dim=768, out_dim=llm.config.hidden_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1024),
            torch.nn.GELU(),
            torch.nn.LayerNorm(1024),
            torch.nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        return self.net(x)

projector = Projector().to(device)

# LOAD CHECKPOINT
checkpoint = torch.load(BEST_PATH, map_location=device)

projector.load_state_dict(checkpoint["projector"])
wav_model.load_state_dict(checkpoint["wav_model"])

projector.eval()
wav_model.eval()
llm.eval()

print("Model loaded")

# AUDIO
def load_audio(path):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    return waveform.squeeze(0)

# DATA
def load_dataset(csv_path, audio_folder):
    df = pd.read_csv(csv_path)
    data = []

    for _, row in df.iterrows():
        path = os.path.join(audio_folder, row["audio_filename"])
        if os.path.exists(path):
            data.append({
                "audio": path,
                "text": row["sentence"]
            })

    return data

# INFERENCE
def predict(audio_path):
    waveform = load_audio(audio_path)

    inputs = wav_processor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_out = wav_model(**inputs).last_hidden_state
        audio_emb = audio_out.mean(dim=1)

        proj = projector(audio_emb).unsqueeze(1).repeat(1, 4, 1)

        prompt = tokenizer("ADAS_CMD:", return_tensors="pt").to(device)
        prompt_embeds = llm.get_input_embeddings()(prompt.input_ids)

        inputs_embeds = torch.cat([proj, prompt_embeds], dim=1)

        attention_mask = torch.ones(inputs_embeds.shape[:2]).to(device)

        outputs = llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=12,
            do_sample=False,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # CLEAN OUTPUT
    pred = pred.replace("ADAS_CMD:", "").strip()
    pred = pred.split("?")[0]
    pred = pred.split(".")[0]
    pred = pred.split("\n")[0]

    return pred.lower()

# TEST
def evaluate(data):
    preds = []
    gts = []

    with open(PRED_FILE, "w") as f:

        for sample in tqdm(data):
            try:
                pred = predict(sample["audio"])
                gt = sample["text"].lower()

                preds.append(pred)
                gts.append(gt)

                f.write(f"GT : {gt}\n")
                f.write(f"PR : {pred}\n")
                f.write("-" * 40 + "\n")

            except Exception as e:
                continue

    # METRICS
    final_wer = wer(gts, preds)
    final_cer = cer(gts, preds)

    with open(METRIC_FILE, "w") as f:
        f.write(f"WER: {final_wer:.4f}\n")
        f.write(f"CER: {final_cer:.4f}\n")

    print("\n FINAL RESULTS:")
    print(f"WER: {final_wer:.4f}")
    print(f"CER: {final_cer:.4f}")

# RUN
if __name__ == "__main__":
    data = load_dataset(
        "dataset2/final_speech_dataset.csv",
        "dataset2/audio_files"
    )

    print("Total samples:", len(data))

    evaluate(data[:5000])