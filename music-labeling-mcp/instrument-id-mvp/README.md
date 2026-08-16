# instrument-id-mvp

Multi-label instrument identification on the [IRMAS dataset](https://zenodo.org/record/1290750) (11 classes).  
Compares a CNN baseline against a domain-aware HarmonicCNN that exploits overtone structure.

## After cloning — what to add manually

Two directories are excluded from the repo and must be added locally:

### 1. Dataset (`data/`)

Download the IRMAS dataset automatically:

```bash
python src/download_data.py --max-per-class 400            # training split (~350 MB)
python src/download_data.py --max-per-class 400 --include-test  # + testing data
```

Or place the raw zips yourself:

```
data/IRMAS/
├── IRMAS-TrainingData.zip
├── IRMAS-TestingData-Part1.zip
├── IRMAS-TestingData-Part2.zip
└── IRMAS-TestingData-Part3.zip
```

Then extract so the layout matches:

```
data/IRMAS/
├── training/IRMAS-TrainingData/{cel,cla,flu,gac,gel,org,pia,sax,tru,vio,voi}/
└── testing/{IRMAS-TestingData-Part1,.../}
```

### 2. Checkpoints (`checkpoints/`)

Either train from scratch (see Quick start below), or copy the `.pt` files from shared storage:

```
checkpoints/
├── InstrumentCNN_seed{0..4}_ep30.pt   ← CNN baseline (5 seeds)
└── HarmonicCNN_seed{0..4}_ep30.pt     ← HarmonicCNN  (5 seeds)
```

---

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download data (adjust --max-per-class as needed)
python src/download_data.py --max-per-class 400 --include-test

# Train both models (5 seeds × 30 epochs each, saves to checkpoints/)
# Fastest on Apple Silicon MPS; runs on CPU too
jupyter notebook cnn_vs_harmonic.ipynb
```

## Models

| Model | Architecture | Val F1-macro | Test F1-macro |
|---|---|---|---|
| `InstrumentCNN` | 3-block CNN + GlobalAvgPool + MLP | 0.632 ± 0.050 | 0.367 |
| `HarmonicCNN` | Same CNN over mel stacked at 0/1/2/3 octave shifts (4 channels) | 0.684 ± 0.018 | 0.422 |

HarmonicCNN stacks copies of the mel spectrogram shifted up by 1, 2, and 3 octaves as extra input channels, letting the CNN see harmonic overtone patterns aligned in frequency. Biggest gains on `vio` (+0.198), `tru` (+0.143), `sax` (+0.088).

## Project structure

```
instrument-id-mvp/
├── src/
│   ├── model.py                    — InstrumentCNN, HarmonicCNN
│   ├── dataset.py                  — IRMASDataset, train/val split
│   ├── features.py                 — log-mel spectrogram + sliding-window util
│   ├── train.py                    — single-model training loop
│   ├── evaluate.py                 — metrics on val or test set
│   ├── download_data.py            — download & extract IRMAS subsets
│   ├── compare.py                  — multi-seed comparison helpers
│   ├── plot.py                     — plotting utilities
│   └── visualize_architectures.py  — architecture diagram generation
├── cnn_vs_harmonic.ipynb           — main comparison notebook (train + eval)
├── pipeline.ipynb                  — end-to-end pipeline walkthrough
├── spectrogram_explorer.ipynb      — spectrogram visualisation
├── fourier_demo.ipynb              — Fourier / mel background
├── run_windowed_eval.py            — sliding-window inference script
├── plots/                          — saved figures (committed)
├── requirements.txt
└── .gitignore
```

Not committed (add manually after clone):

```
data/       ← IRMAS audio files (~350 MB training + ~230 MB testing)
checkpoints/ ← trained .pt files (~5 MB each, 10 total)
```

## Instrument classes

`cel` cello · `cla` clarinet · `flu` flute · `gac` acoustic guitar ·
`gel` electric guitar · `org` organ · `pia` piano · `sax` saxophone ·
`tru` trumpet · `vio` violin · `voi` human voice

## Device

Automatically selects Apple Silicon MPS → CPU. No GPU required.
