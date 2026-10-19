# Assignment 1 - Task 3: Text Classification

Generative AI & LLM (Fall 2026)

This repository contains the PyTorch implementation and experiment results for **Task 3 only**: classification of published and generated news articles. The original model uses `nn.EmbeddingBag` followed by a linear classifier; the second model uses `nn.Embedding`, `nn.LSTM`, padding-aware max pooling, dropout and a linear output layer.

## Files and directory structure

```text
Assignment1Task3/
├── txt_classifier.py
├── requirements.txt
├── tsk3_execution.py
├── README.md
├── Report.pdf
├── data/
│   ├── txt_train.csv
│   ├── txt_val.csv
│   └── txt_test.csv
└── checkpoints/                # training outputs kept on Google Drive
```

**Model checkpoint folder (Google Drive):**
https://drive.google.com/drive/folders/1Az7Jv03aZqb2gsqUduyvbe676Sk-cfyq?usp=sharing

The folder contains the saved baseline, SGD, Adam and LSTM `.pth` checkpoints. Model checkpoints are hosted on Drive rather than committed to GitHub.

## Dataset

The provided dataset contains 358 training, 76 validation, and 74 test articles. Each split contains equal numbers of `real` (label 0) and `fake` / generated (label 1) articles. The CSV columns are `article`, `source`, `label`, `label_idx`.

The model uses **only** `article` as input text, with `label_idx` as the target. The provided data splits are not changed.

## Run in Google Colab

1. Copy `txt_classifier.py`, `requirements.txt`, and the `data/` folder into `MyDrive/Assignment1Task3/`.
2. Open Google Colab. To use a GPU for the LSTM, choose **Runtime -> Change runtime type -> T4 GPU** if available.
3. Mount Google Drive and change to your assignment folder:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Assignment1Task3
```

4. Colab generally includes `torch`, `pandas`, and `matplotlib` already. Check installed packages:

```python
import torch, pandas, matplotlib
print(torch.__version__, torch.cuda.is_available())
```

If a package is missing, install it in a Colab cell with `%pip install <package>`. **Avoid forcing older pinned PyTorch wheels from `requirements.txt` onto a newer Colab Python environment**; use Colab's compatible preinstalled PyTorch where possible. For a local Python 3.10-3.12 environment, `python -m pip install -r requirements.txt` may be used.

5. Create output folders:

```python
!mkdir -p checkpoints results
```

All training and implementation commands below are invoked from Colab using `!python ...`. Source code remains in `.py` files; a notebook is not part of the submission.

## Reproduce Task 3 experiments

Commands below assume the completed `txt_classifier.py` supports `--model`, `--optimizer`, `--learning_rate`, `--hidden_dim`, and the argument names in the provided starter file.

### 3.1 - Article-length histogram

```bash
python txt_classifier.py --batch_size 1 --max_len -1 --length_histogram
```

Expected saved figure: `article_length_histogram.png`. This measures training-article token lengths before padding or truncation.

### 3.2 - Runtime and variable-length batching

```bash
python txt_classifier.py --batch_size 1  --max_len -1   --n_epochs 5 --device cpu --output checkpoints/run1.pth
python txt_classifier.py --batch_size 1  --max_len 1024 --n_epochs 5 --device cpu --output checkpoints/run2.pth
python txt_classifier.py --batch_size 32 --max_len 1024 --n_epochs 5 --device cpu --output checkpoints/run3.pth
python txt_classifier.py --batch_size 32 --max_len -1   --n_epochs 5 --device cpu
```

The fourth command **intentionally raises a `RuntimeError`** in the supplied `PadSequence` collator because `torch.stack` cannot stack unequal-length token tensors when padding/truncation are disabled. Do not change that batching behavior for this experiment.

Observed training/evaluation runtimes: 7.390 s, 7.007 s, and 4.340 s for the first three configurations respectively. Actual runtimes will depend on hardware and system load.

### 3.3 - Original model, SGD versus Adam

```bash
python txt_classifier.py --model baseline --optimizer sgd  --learning_rate 5     --batch_size 32 --max_len 1024 --n_epochs 5 --device cpu --output checkpoints/sgd_model.pth
python txt_classifier.py --model baseline --optimizer adam --learning_rate 0.001 --batch_size 32 --max_len 1024 --n_epochs 5 --device cpu --output checkpoints/adam_model.pth
```

| Optimizer | Validation accuracy | Test accuracy |
|---|---:|---:|
| SGD, learning rate 5 | 85.53% | 83.78% |
| Adam, learning rate 0.001 | 85.53% | 91.89% |

### 3.4 - LSTM model with Adam

```bash
python txt_classifier.py --model lstm --optimizer adam --learning_rate 0.001 --batch_size 32 --max_len 1024 --embed_dim 64 --hidden_dim 64 --n_epochs 5 --device cuda --output checkpoints/lstm_model.pth
```

If no GPU is available, use `--device cpu` instead. Observed LSTM final validation accuracy was **89.47%** and test accuracy was **91.89%**.

### 3.5 - Article pair observations

The report discusses two real/generated article pairs (the panda flight story and the Buffalo Grove scholarship story) from the provided text data.

## Deliverables

- `txt_classifier.py` - completed source file, including histogram, padding/truncation, optimizer switch and LSTM model.
- `Report.pdf` - figures, actual experiment results, comparisons and observations.
- `README.md` - setup, execution instructions and checkpoint link.
- `requirements.txt` and the dataset files needed to reproduce the experiments.

