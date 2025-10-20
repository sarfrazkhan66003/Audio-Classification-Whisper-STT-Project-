from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import soundfile as sf

# ------------------------------
# 1. Load pre-trained Whisper model and processor
# ------------------------------
model_name = "openai/whisper-small"
print("⏳ Loading Whisper model...")

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

model.config.forced_decoder_ids = None  # Disable forced language
model.eval()

print("✅ Whisper model loaded successfully!")

# ------------------------------
# 2. Load an audio file (16kHz, mono)
# ------------------------------
AUDIO_FILE = r"C:\Users\DELL\Desktop\Sarfu\PW Data Science\PW Project DS\Audio-classification\sample.wav"  # 👈 replace with your file
audio, sr = librosa.load(AUDIO_FILE, sr=16000)

# ------------------------------
# ------------------------------
input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

# ------------------------------
# 4. Generate text (transcribe)
# ------------------------------
print("🎧 Transcribing audio...")
with torch.no_grad():
    predicted_ids = model.generate(input_features)

# ------------------------------
# 5. Decode text output
# ------------------------------
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print("🗣️ Transcription Result:")
print(transcription)
