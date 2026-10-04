# CLAUDE.md — music-labeling-mcp

Контекст магістерської дипломної роботи: автоматичне розпізнавання музичних інструментів у аудіозаписах.

---

## Архітектура проєкту

```
music-labeling-mcp/
└── instrument-id-mvp/
    ├── src/
    │   ├── features.py       — log-mel spectrogram [128×130] з .npy кешем
    │   ├── dataset.py        — IRMASDataset, train/val split
    │   ├── model.py          — InstrumentCNNBiLSTM (2-layer CNN → BiLSTM(64) → Linear)
    │   ├── train.py          — тренувальний цикл, зберігає model.pt
    │   └── evaluate.py       — F1-macro/micro + per-class звіт
    ├── data/IRMAS/           — датасет (11 класів інструментів)
    ├── fourier_demo.ipynb    — інтерактивне пояснення FFT
    ├── spectrogram_explorer.ipynb — піано → waveform → FFT → mel spectrogram
    └── cnn_vs_harmonic.ipynb — порівняння CNN vs HarmonicCNN
```

**Модель:** multi-label класифікація (кілька інструментів одночасно), датасет IRMAS (11 класів).

**11 класів:** `cel` cello · `cla` clarinet · `flu` flute · `gac` acoustic guitar · `gel` electric guitar · `org` organ · `pia` piano · `sax` saxophone · `tru` trumpet · `vio` violin · `voi` human voice

---

## Команди

```bash
cd instrument-id-mvp
source venv/bin/activate

# Дані
python src/download_data.py --max-per-class 50
python src/download_data.py --max-per-class 50 --include-test

# Тренування
python src/train.py --epochs 5

# Оцінка
python src/evaluate.py
python src/evaluate.py --test-dir data/IRMAS/testing

# Запуск notebooks (через термінал, не PyCharm)
venv/bin/jupyter notebook spectrogram_explorer.ipynb
venv/bin/jupyter notebook fourier_demo.ipynb
venv/bin/jupyter notebook cnn_vs_harmonic.ipynb
```

---

## Зміни зроблені в цій розмові

### fourier_demo.ipynb — додана секція 3 (механізм FFT)

Між секцією "Сума синусоїд" і "FFT результат" додано 3 нові клітинки:

1. **Markdown — теорія DFT:**
   - Пояснення через кореляцію: множимо сигнал на тест-синусоїду і підсумовуємо
   - Таблиця: збіг частот → добутки завжди позитивні → велика сума; не збіг → скасовуються → нуль
   - Формула: $X[k] = \sum x[n] \cdot e^{-j2\pi kn/N}$, розкладена на Re (cos) та Im (sin) кореляції
   - Амплітуда = $\sqrt{Re^2 + Im^2}$, фаза = atan2(Im, Re)

2. **Код — Демо 3a (кореляція з однією частотою):**
   - Перевірка 440 Гц (є в сигналі) vs 600 Гц (немає)
   - 4 панелі: сигнал + тест-хвиля / поточочний добуток для кожної частоти
   - Візуально показує чому збіг → велика сума, не збіг → нуль

3. **Код — Демо 3b ("ручний DFT"):**
   - Python-цикл: для кожної f від 50 до 2200 Гц рахує `re = dot(sig, cos)`, `im = dot(sig, sin)`
   - Порівнює результат із `numpy.fft.rfft` — графіки ідентичні
   - Виводить порівняння O(N²) ручного DFT vs O(N log N) FFT

---

## Концептуальні домовленості (пояснення)

- **Амплітуда у waveform** — нормалізоване число [-1, 1], що представляє оцифровану напругу з мікрофона (пропорційна відхиленню тиску повітря). До нормалізації — 16-бітне ціле [-32768, 32767].
- **dB у спектрі** — логарифмічна шкала відносного розміру: `20 * log10(amp / amp_max)`. 0 dB = найсильніша частота, -20 dB = в 10 разів слабша. Використовується для діапазону, не тільки для гучності.
- **threshold=0.3 у cnn_vs_harmonic** — поріг sigmoid для multi-label: якщо ймовірність ≥ 0.3 → клас активний. Нижче 0.5 бо рідкісні інструменти мають низьку впевненість моделі. Правильно підбирати перебором на validation set.

---

## Середовище

- **Python:** `instrument-id-mvp/venv/bin/python` (Python 3.12.13, pyenv)
- **НЕ використовувати** `/usr/bin/python3` — це системний Python 3.9.6 macOS, у нього зламаний pip і відсутні модулі стандартної бібліотеки в контексті PyCharm
- **PyCharm kernel error** `IllegalStateException: Kernel process exited` — PyCharm підхоплює системний Python замість venv. Рішення: запускати notebooks через термінал командою `venv/bin/jupyter notebook <file>.ipynb`
- **kernel.json** виправлено: `instrument-id-mvp/venv/share/jupyter/kernels/python3/kernel.json` — замінено `"python"` на абсолютний шлях до venv python

---

## Відкриті TODO

- [ ] Підібрати `threshold` перебором на validation set замість magic number 0.3
- [ ] Датасет IRMAS 2012 — невеликий (6000 кліпів). Розглянути аугментацію або додатковий датасет
- [ ] Порівняти HarmonicCNN vs базовий CNN за F1 — основний дослідницький внесок роботи
- [ ] Актуальність роботи краще формулювати як "порівняння архітектур з domain knowledge vs без", а не "нова система розпізнавання" (конкурує з MERT/MusicGen)
- [ ] Налаштувати PyCharm interpreter правильно: Settings → Project → Python Interpreter → вказати `instrument-id-mvp/venv/bin/python`
