# Real-Time Human Footstep Detection

A lightweight, CPU-friendly prototype: train a small CNN on Log-Mel
spectrograms of WAV footstep clips, then run continuous binary
detection (FOOTSTEP DETECTED / NO FOOTSTEP) on live microphone audio.

## 1. Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Linux only** — sounddevice needs PortAudio at the OS level:
```bash
sudo apt install portaudio19-dev
```
macOS/Windows: PortAudio ships bundled with the `sounddevice` wheel, no
extra step needed.

## 2. Dataset layout

```
dataset/
├── footstep/
│   ├── clip001.wav
│   ├── clip002.wav
│   └── ...
└── no_footstep/
    ├── bg001.wav
    └── ...
```
Any sample rate / mono or stereo / any length is fine — preprocessing
handles resampling to 16kHz, mono conversion, and fixed-length
windowing automatically.

### If you don't have `no_footstep` data yet

You need negative examples to train a binary classifier. Fastest ways
to get them, roughly in order of realism/effort tradeoff:

1. **Record it yourself** (best — matches your real deployment mic/room):
   5–10 minutes of typical ambient sound where the system will run:
   silence, talking, typing, chair movement, door sounds, traffic,
   HVAC hum, non-footstep object drops. Chop into clips with any audio
   editor or with `preprocessing/audio.py`'s segmenting logic.
2. **Public datasets** — pull background/non-footstep clips from
   [ESC-50](https://github.com/karoldvl/ESC-50) or
   [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
   (both free, small, quick to download). Use categories unrelated to
   footsteps (engine, wind, dog bark, etc.) as negatives.
3. **Silence/noise floor** — a few minutes of just room-tone /
   microphone noise-floor recording is a legitimate (if weak) negative
   class in a pinch; combine with (1) or (2) for better coverage.

More negative diversity = fewer live false positives, so don't skip
this even under time pressure.

## 3. Train

```bash
python train.py --dataset_root dataset --epochs 30
```
Key flags (all optional, shown with defaults):
```bash
python train.py \
  --dataset_root dataset \
  --sr 16000 \
  --duration_sec 1.0 \
  --hop_sec 0.5 \
  --n_fft 1024 --hop_length 320 --win_length 1024 --n_mels 64 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --patience 6 \
  --checkpoint checkpoints/best_model.pt
```
This prints dataset size + class distribution before training starts,
trains with class-weighted `CrossEntropyLoss` (handles imbalance
automatically), applies early stopping on validation F1, and saves the
best-F1 checkpoint. The checkpoint is self-contained: model weights +
class names + sample rate + mel params + duration, so nothing about
preprocessing has to be hardcoded elsewhere.

## 4. Evaluate on held-out test set

```bash
python evaluate.py --dataset_root dataset --checkpoint checkpoints/best_model.pt
```
Reports accuracy, precision, recall, F1, a confusion matrix, and
average per-sample inference time (plus a CPU utilization snapshot if
`psutil` is installed).

## 5. Run live detection

```bash
python infer_live.py --checkpoint checkpoints/best_model.pt
```
Sample output:
```
[12:41:03.120] FOOTSTEP DETECTED | confidence=94.2%
[12:41:03.520] NO FOOTSTEP | confidence=91.1%
```
Tunable flags:
```bash
python infer_live.py \
  --checkpoint checkpoints/best_model.pt \
  --infer_interval_sec 0.3 \
  --prob_threshold 0.80 \
  --smoothing_window 3 \
  --smoothing_min_hits 2
```
- `prob_threshold`: minimum footstep probability for a single window
  to count as a "hit."
- `smoothing_window` / `smoothing_min_hits`: majority-vote smoothing —
  default requires 2 of the last 3 predictions to hit before declaring
  FOOTSTEP DETECTED, which kills single-frame flicker.
- If the wrong microphone is picked up, list devices with
  `python -c "import sounddevice as sd; print(sd.query_devices())"`
  and pass the index via `--device_index`.

## Project structure

```
footstep_detection/
├── dataset/                  # your WAV data (footstep/, no_footstep/)
├── models/
│   └── cnn.py                 # CNN architecture
├── preprocessing/
│   ├── audio.py                # load/mono/resample/normalize/fixed-length
│   └── features.py             # Log-Mel spectrogram extraction
├── dataset.py                 # PyTorch Dataset, file-level split, augmentation
├── train.py                   # training loop + checkpoint saving
├── evaluate.py                 # test-set metrics + confusion matrix
├── infer_live.py               # microphone -> ring buffer -> live inference
├── checkpoints/                # saved model weights land here
└── requirements.txt
```

## Design notes

- **Anti-leakage split**: splitting happens at the *source file* level
  first (70/15/15 by recording), then all sliding-window segments from
  a file inherit that file's split. No chunk from the same recording
  ever appears in two splits.
- **Identical preprocessing train vs. live**: both paths call the same
  `preprocessing/audio.py` and `preprocessing/features.py` functions,
  and live inference reads all mel/sample-rate/duration parameters
  straight from the checkpoint — there's no way for them to drift out
  of sync.
- **Augmentation** is applied only to the training split (random gain,
  ±0.1s time shift, light additive noise) — validation/test/live all
  see clean, deterministic preprocessing.
- **Low latency**: 1s context window, 0.25–0.5s inference cadence,
  `torch.no_grad()`, no disk I/O in the live loop, GAP-based CNN head
  keeps the model tiny (~50–100K params) and fast enough for real-time
  CPU inference.

## Optional next steps (after the basic prototype works)

- **Transfer learning with PANNs** (Pretrained Audio Neural Networks)
  or YAMNet embeddings instead of training the CNN from scratch — big
  accuracy gains with small datasets, at the cost of a heavier model
  and slower CPU inference.
- **Noise robustness**: train with more diverse real-world background
  noise (SNR-controlled mixing), add SpecAugment (frequency/time
  masking) once the basic pipeline is validated.
- **Better event detection**: replace fixed-window classification with
  onset detection (spectral flux / energy peak picking) to localize
  the exact footstep moment rather than just "present in this window."
- **ONNX export** (`torch.onnx.export`) + `onnxruntime` for faster,
  smaller-footprint CPU inference, especially useful for embedded
  deployment.
- **Streamlit GUI**: wrap `infer_live.py`'s detection loop in a small
  Streamlit app with a live waveform plot and a probability gauge
  instead of terminal output.
