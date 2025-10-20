Audio Classification & Whisper STT
Speech-to-Text + Audio Classification pipeline using Hugging Face Whisper (openai/whisper-small) for transcription and a PyTorch-based classifier for audio classification.
This repository contains code to record or load audio, transcribe with Whisper, extract features, train/evaluate an audio classifier, and export predictions + visual outputs.

✨ Features
🎙️ Record or load WAV audio (16 kHz recommended)
📝 Transcribe audio using openai/whisper-small (Hugging Face)
🔊 Preprocess audio (resampling, mono, trimming, normalization)
📊 Extract features (log-Mel spectrograms / MFCCs / raw waveform)
🧩 Trainable classifier (PyTorch/Scikit-learn) for audio labels (eg. speech commands, emotion, environmental sounds)
🧭 Inference pipeline: record → transcribe → classify → save results (text + heatmap + JSON)
📈 Evaluation metrics: Accuracy, Precision, Recall, F1, Confusion Matrix
🖼️ Output visualizations: spectrograms, class probability bar charts, Grad-CAM / saliency (if using CNN)
💾 Save & load models (Hugging Face & local save_pretrained / torch.save)
✅ Designed to run locally (VS Code / conda/pip venv) — GPU optional for faster training/inference

📁 Repository Structure (suggested)
audio-classification/
├─ data/                       # raw & processed audio datasets
├─ notebooks/                  # experiments & EDA notebooks
├─ models/                     # saved models (whisper + classifier)
├─ src/
│  ├─ record.py                # record audio and save sample.wav
│  ├─ transcribe.py            # load whisper model + transcribe
│  ├─ preprocess.py            # resample, trim, normalize, augment
│  ├─ features.py              # compute spectrograms, MFCCs, etc.
│  ├─ train.py                 # training loop for classifier
│  ├─ infer.py                 # inference wrapper (stt + classifier)
│  └─ utils.py                 # helper functions (IO, plots)
├─ whisper_stt.py              # simple run script: record -> transcribe
├─ requirements.txt            # pip install -r requirements.txt
├─ README.md
└─ ABOUT.md


🛠️ Installation (Quick Start)
Use a fresh environment (conda or venv) to avoid TF conflicts.
# using conda (recommended)
conda create -n audio_env python=3.10 -y
conda activate audio_env

# install essentials
pip install torch torchvision torchaudio
pip install transformers librosa soundfile sounddevice scipy scikit-learn matplotlib seaborn tqdm

# optional (for faster inference)
pip install accelerate

# verify
python -c "import transformers, torch, librosa; print('OK')"

Train classifier from dataset
python src/train.py \
  --data_dir data/train \
  --epochs 30 \
  --batch_size 32 \
  --save_path models/my_classifier.pth

🔬 About the Model & Algorithm
Whisper for STT (openai/whisper-small) 🗣️
  Model type: Encoder-decoder transformer trained by OpenAI, available on Hugging Face.
  Role: Converts raw audio → token IDs → decoded human-readable text.
  Why Whisper? Robust off-the-shelf ASR across many languages and noisy conditions.
  Usage in repo: We use WhisperProcessor + WhisperForConditionalGeneration from transformers (PyTorch backend).
  Config note: model.config.forced_decoder_ids = None disables forced language tokens so model detects language automatically.

Classifier (Audio Labeling) 🏷️
  Input features: log-Mel spectrograms (recommended), MFCCs or raw waveform features.
  Model options: simple MLP / CNN (e.g., 1D-CNN or 2D-CNN on spectrograms) or pre-trained audio models (e.g., PANNs).
  Training algorithm: supervised learning with cross-entropy loss, Adam optimizer, learning rate scheduling, early stopping.
  Output: class probabilities (softmax), predicted label, confidence score.


Preprocessing Steps (detailed)

Load audio — use librosa.load(path, sr=16000, mono=True) (16 kHz mono).
Trim silence — optional librosa.effects.trim or energy-based threshold.
Normalize — zero-mean / unit-variance or peak normalization.
Feature extraction — compute log-Mel spectrogram:
  window size: 1024 (or 25 ms), hop length: 256 (or 10 ms), number of Mel bins: 80
  apply librosa.feature.melspectrogram -> librosa.power_to_db
Data augmentation (training): time-stretching, pitch shift, additive noise, SpecAugment (time/freq masking)
Batching & padding — pad features to fixed length or use collate_fn to handle variable lengths.

📥 Input & 📤 Output (Detailed)
Input
Audio file: WAV (PCM) recommended, 16 kHz, mono. Example: sample.wav
Dataset folder: data/train/<class_name>/xxxx.wav
Optional: CSV/annotations file mapping audio filenames → labels / metadata

Output
Transcription text (string): produced by Whisper e.g. "hello world this is a test"
Classification results (JSON):
{
  "file": "sample.wav",
  "transcription": "hello world this is a test",
  "predicted_label": "speech_command",
  "confidence": 0.92,
  "probabilities": {
    "speech_command": 0.92,
    "music": 0.05,
    "noise": 0.03
  }
}


Visual outputs:
sample_spectrogram.png — mel-spectrogram visualization
sample_confidence_bar.png — bar chart of class probabilities
saliency_map.png (optional) — model attention/saliency visualization

🧪 Evaluation & Metrics
Accuracy — (TP+TN)/Total for balanced tasks
Precision / Recall / F1-score — per-class and macro/micro averages
Confusion Matrix — helps spot common misclassifications
ROC / AUC — for binary or multi-label scenarios
Cross-validation — k-fold recommended for small datasets

🧾 About This Project (ABOUT.md) — Short Version
Purpose: Create a reproducible pipeline for transcribing audio & performing audio classification. Ideal for speech command recognition, sound-event detection, or building audio-based features for larger systems.
Audience: ML practitioners, students, hobbyists who want a simple end-to-end audio ML pipeline with Whisper for STT + custom classifier.
Outcomes: A working system that records audio, transcribes with Whisper, extracts features, classifies audio, and saves human-friendly outputs and visualizations.
