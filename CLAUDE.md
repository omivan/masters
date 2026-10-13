# CLAUDE.md — Master's Thesis

Магістерська дипломна робота: автоматичне розпізнавання музичних інструментів у аудіозаписах.
Дослідницьке питання: **архітектури з domain knowledge (harmonic stacking) vs без** — не "нова система розпізнавання".
Детальний журнал експериментів — `CONTEXT.md` (тримати в синхроні з цим файлом).

## Структура проєкту

```
masters/
├── CONTEXT.md                — журнал експериментів і результатів
├── code/instrument-id-mvp/
│   ├── src/                  — код (див. нижче)
│   ├── checkpoints/          — *_best.pt (state_dict + model_cls + config_name + f1_macro)
│   ├── data/IRMAS/training/IRMAS-TrainingData/<cls>/   — 6705 моно-кліпів по 3 с
│   ├── data/IRMAS/testing/   — 2874 поліфонічні кліпи (multi-label)
│   └── *.ipynb               — notebooks (FFT demo, spectrogram, CNN vs HarmonicCNN)
└── thesis/                   — LaTeX (main.tex, Chapters/, Images/, output/main.pdf)
```

---

## Python-код

**Задача:** тренування — single-label (переважаючий інструмент), тест — multi-label поліфонія. Метрика — **test macro F1**.

**11 класів:** `cel` cello · `cla` clarinet · `flu` flute · `gac` acoustic guitar · `gel` electric guitar · `org` organ · `pia` piano · `sax` saxophone · `tru` trumpet · `vio` violin · `voi` human voice

**Фічі (`src/features.py`):** sr=22050, n_fft=1024, hop=512, 128 mel → log-mel `[128, 130]` (3 с), кеш `.npy` поруч з wav. Є також CQT (24 bins/oct, 144 біни) і HPSS — поки не використовуються в найкращих моделях.

**Harmonic stacking (у `forward()` моделі):**
- старий (`HarmonicCNN`…`DeepHarmonicCNN`): mel зсунута на фіксовані 0/12/24/36 бінів + Δ + Δ² → 6 каналів. **Неточний** (див. пастки).
- новий `ExactHarmonicStack`: канал h на біні k = log-mel на частоті h·f_k (лінійна інтерполяція між mel-бінами). `DeepExactHarmonicCNN` (гармоніки 1–5), абляції `DeepExactOctaveCNN` (1,2,4,8) і `DeepMelCNN` (без гармонік).

**Inference на тесті (`evaluate_test_sliding.py`):** ковзне вікно 3 с, hop 50%, S2-агрегація (сума sigmoid → нормалізація на max → threshold 0.3).

### Поточні результати (test macro F1)

| Модель | Test macro |
|---|---|
| **DeepHarmonicCNN + CosineLR** (найкраща наша) | **0.490** |
| Han2017CNN, 1-с вікна (наше відтворення) | 0.501 |
| Han et al. 2017 (стаття) — **ціль** | 0.503 |
| HarmonicHan2017CNN_3sec / Han2017CNN_3sec | 0.477 / 0.435 |
| BasicCNN | 0.443 |

Не допомогло: аугментація pitch±2/SpecAugment, SE-блоки, 60 епох, 100% даних без val, DeepHarmonic на 1-с вікнах (0.424). Повна таблиця — `CONTEXT.md`.

### Модулі `src/`

- `model.py` — BasicCNN, HarmonicCNN, DeltaHarmonicCNN, DeepHarmonicCNN, SEDeepHarmonicCNN, Han2017CNN, HarmonicHan2017CNN
- `dataset.py` — `IRMASDataset`, `train_val_split` (стратифікований 80/20, за замовчуванням групує за записом — `recording_id()`), `_scan_testing_dir`
- `combo_exp.py` — тренування конфігів (`--only` точна назва або підрядок, `--split recording|clip`, `--epochs 30`, `--seeds 5`) + підбір threshold на val. Чекпоінти з `--split recording` мають суфікс `_rec`
- `han_exp.py` — Han-style: випадкові 1-с кропи, 1-с ковзне вікно на тесті
- `train_full.py` — тренування на 100% train
- `evaluate_test_sliding.py` — оцінка `checkpoints/*_best.pt` на тесті (`--threshold`, `--only` фільтр за назвою файлу)

### Команди (Windows, CUDA)

```bash
cd code/instrument-id-mvp
venv/Scripts/python src/combo_exp.py --only DeepHarmonicCNN --epochs 30
venv/Scripts/python src/han_exp.py
venv/Scripts/python src/evaluate_test_sliding.py --only _rec_best
```

`--eval-test` зберігає всі сіди (`*_seed{n}.pt`) і після тренування дає test mean ± std (пороги 0.3/0.4/0.5).
Тренувальні mel-и вантажаться цілком у GPU (`preload_mels` + `GPULoader`, ~400 МБ), тестові — кешуються в RAM процесу.
Типовий час: 1 конфіг × 5 сідів × 30 епох ≈ **3.3 хв**, тест 5 сідів ≈ **1 хв**.
Прогрес: `~/.claude/train_progress.json` (status line) або `venv/Scripts/python src/progress.py`.

Довгі запуски — у фоні з логом у `logs/*_out.txt` / `*_err.txt`. **Обовʼязково `PYTHONIOENCODING=utf-8`** при перенаправленні виводу, інакше `UnicodeEncodeError` на символах `─` (cp1252).

---

## Відомі проблеми / пастки

- **Val leakage (виправлено, `--split recording`):** кожен запис IRMAS розрізаний на ~3 кліпи (`[cel][cla]0001__1.wav`, `__2`, `__3`). Старий поділ по кліпах мав 977 записів одночасно в train і val → val F1 (~0.80) завищений. Усі результати в `CONTEXT.md` до цього виправлення отримані на старому поділі.
- **Старий harmonic stacking неточний:** октава на mel-шкалі — не константа (55→110 Hz ≈ 3 біни, 880→1760 ≈ 21, 3.5→7 kHz ≈ 28). Перевірка на синусоїдах: фіксований зсув 12 бінів читає від −3 до −20 (тиша) замість піку гармоніки ≈ 11; `ExactHarmonicStack` — 9–10.8. Також 0/1/2/3 октави = гармоніки 1, 2, 4, 8 — без непарних 3f, 5f.
- Val — моно, тест — поліфонія: threshold, підібраний на val, може не переноситись на тест.
- Threshold і вибір моделі **не** підбирати на тест-сеті.

---

## LaTeX (thesis/)

```bash
cd thesis && make    # або xelatex main.tex → output/main.pdf
```

Розділи: `a3_introduction` (вступ: мета/об'єкт/предмет/методи з ФПЗ), `c1` огляд існуючих рішень, `c2` математичне забезпечення, `c3` програмне забезпечення (загально), `c4` валідація (дані, методика, метрики, результати, абляції). Кожен розділ закінчується «Висновки до розділу».
Бібліографія — ручний `thebibliography` у `Chapters/w2_bibliography.tex`, впорядкований за першою появою `\cite` (не `thesis.bib`). Ще не оновлені: `a1_abstract` (стара версія про CNN+BiLSTM), `w1_conclusions`.
Збірка: `xelatex -output-directory=output main.tex` двічі (TikZ потребує `positioning`, підключено в `01_packages.tex`).
У тексті дисертації, ФПЗ і матриці **не використовувати внутрішні назви моделей** (DeepMel…, LogPons, PolyMix) і уникати англіцизмів.

---

## Концептуальні домовленості

- **Амплітуда у waveform** — нормалізоване [-1, 1] (до нормалізації — int16 [-32768, 32767]).
- **dB у спектрі** — `20 * log10(amp / amp_max)`, 0 dB = найсильніша частота.
- **threshold=0.3** — поріг після S2-нормалізації; підібраний на val перебором [0.15..0.50].

---

## Відкриті TODO

- [ ] Дійти до рівня Han (0.503) з 3-с harmonic-моделлю
- [x] Виправити val split (групування за записом)
- [x] Точний harmonic stacking (`ExactHarmonicStack`) — чекає результатів тренування
- [ ] Перевірити розбіжність у CONTEXT.md: "Han з 3-sec вікнами 0.485" vs таблиця 0.435
- [ ] Прибрати `*_out.txt`/`*_err.txt` у `logs/`, закомітити видалені експериментальні скрипти
