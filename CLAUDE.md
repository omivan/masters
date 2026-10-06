# CLAUDE.md — Master's Thesis

Магістерська дипломна робота: автоматичне розпізнавання музичних інструментів у аудіозаписах.

## Структура проєкту

```
masters/
├── code/                     — Python-код (ML-система розпізнавання інструментів)
│   └── instrument-id-mvp/    — основний MVP
│       ├── src/              — модулі (features, dataset, model, train, evaluate)
│       ├── data/IRMAS/       — датасет (11 класів інструментів)
│       └── *.ipynb           — jupyter notebooks (FFT demo, spectrogram, CNN vs HarmonicCNN)
└── thesis/                   — LaTeX-код дипломної роботи
    ├── main.tex              — головний файл
    ├── Chapters/             — розділи (.tex)
    ├── Images/               — зображення
    └── output/main.pdf       — скомпільований PDF
```

> Примітка: якщо папка ще називається `master-paper` — не перейменована через блокування IDE, перейменуй вручну після закриття PyCharm.

---

## Python-код (code/)

### Архітектура моделі

**Модель:** multi-label класифікація (кілька інструментів одночасно), датасет IRMAS (11 класів).

**11 класів:** `cel` cello · `cla` clarinet · `flu` flute · `gac` acoustic guitar · `gel` electric guitar · `org` organ · `pia` piano · `sax` saxophone · `tru` trumpet · `vio` violin · `voi` human voice

Пайплайн: log-mel spectrogram [128×130] → InstrumentCNNBiLSTM (2-layer CNN → BiLSTM(64) → Linear)

### Команди

```bash
cd code/instrument-id-mvp
source venv/bin/activate

# Дані
python src/download_data.py --max-per-class 50
python src/download_data.py --max-per-class 50 --include-test

# Тренування
python src/train.py --epochs 5

# Оцінка
python src/evaluate.py
python src/evaluate.py --test-dir data/IRMAS/testing

# Notebooks (через термінал, не PyCharm)
venv/bin/jupyter notebook spectrogram_explorer.ipynb
venv/bin/jupyter notebook fourier_demo.ipynb
venv/bin/jupyter notebook cnn_vs_harmonic.ipynb
```

### Середовище

- **Python:** `instrument-id-mvp/venv/bin/python` (Python 3.12.13, pyenv)
- **НЕ використовувати** `/usr/bin/python3` — системний Python 3.9.6 macOS, зламаний pip
- **PyCharm kernel error** `IllegalStateException: Kernel process exited` — PyCharm підхоплює системний Python. Рішення: запускати notebooks через термінал
- **kernel.json** виправлено: `instrument-id-mvp/venv/share/jupyter/kernels/python3/kernel.json`

---

## LaTeX-код (thesis/)

### Компіляція

```bash
cd thesis
make          # або xelatex main.tex
# PDF → thesis/output/main.pdf
```

### Структура розділів

- `Chapters/a1_abstract.tex` — анотація
- `Chapters/a3_introduction.tex` — вступ
- `Chapters/c1_chapter_01.tex` … `c4_chapter_04.tex` — основні розділи
- `Chapters/w1_conclusions.tex` — висновки
- `thesis.bib` — список літератури

---

## Концептуальні домовленості

- **Амплітуда у waveform** — нормалізоване [-1, 1], оцифрована напруга з мікрофона. До нормалізації — 16-бітне ціле [-32768, 32767].
- **dB у спектрі** — `20 * log10(amp / amp_max)`. 0 dB = найсильніша частота, -20 dB = в 10 разів слабша.
- **threshold=0.3 у cnn_vs_harmonic** — поріг sigmoid для multi-label. Нижче 0.5 бо рідкісні інструменти мають низьку впевненість моделі.

---

## Відкриті TODO

- [ ] Підібрати `threshold` перебором на validation set замість magic number 0.3
- [ ] Датасет IRMAS 2012 — невеликий (6000 кліпів). Розглянути аугментацію або додатковий датасет
- [ ] Порівняти HarmonicCNN vs базовий CNN за F1 — основний дослідницький внесок роботи
- [ ] Актуальність роботи: "порівняння архітектур з domain knowledge vs без", не "нова система розпізнавання"
- [ ] Налаштувати PyCharm interpreter: Settings → Project → Python Interpreter → `code/instrument-id-mvp/venv/bin/python`
