# 2026-DL-SEM-III-IMDB-Sentiment-Analysis
IMDB-Sentiment-Analysis using LSTMs and RNNs.
# IMDB Sentiment Analysis using RNN and LSTM

Binary sentiment classification (positive / negative) of IMDB movie reviews, built with PyTorch. I compare a vanilla RNN against an LSTM on the same preprocessing pipeline.

## Dataset

[IMDB Large Movie Review Dataset](https://huggingface.co/datasets/stanfordnlp/imdb) loaded from Hugging Face: 25,000 training and 25,000 test reviews, evenly split between positive and negative.

## Preprocessing

1. Remove HTML tags (`<br />`)
2. Lowercase and strip punctuation
3. Sentence and word tokenization (NLTK)
4. POS-tagging, then WordNet lemmatization
5. Stopword removal
6. Vocabulary of the 20,000 most frequent words, plus `<pad>` and `<unk>`
7. Dynamic padding per batch, with packed sequences for the RNN

## Models

Both models share the same structure:

`Embedding (128) -> RNN / LSTM (hidden 128) -> Dropout (0.3) -> Linear -> Sigmoid`

| Setting | Value |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam (lr = 5e-4, weight decay = 1e-5) |
| Batch size | 32 |
| Max epochs | 20 |
| Gradient clipping | max norm 1.0 |

## Results

| Model | Train Accuracy | Test Accuracy | Notes |
|---|---|---|---|
| RNN | 90.27% | 81.75% | Early stopping at epoch 11 |
| LSTM | 98.75% | 87.06% | 2.69M parameters |

The LSTM beats the RNN by about 5 points, which fits with LSTMs handling long-range dependencies in long reviews better. It also overfits more (a gap of about 12 points between train and test), so more regularization would be a natural next step.

Training curves and sample predictions are in the notebooks.

## Repository Structure

```
├── README.md
├── RNN_IMDB.ipynb
├── LSTM_IMDB.ipynb
└── results/
    ├── RNN_IMDB.md
    └── LSTM_IMDB.md
```

## How to Run

```bash
pip install torch datasets nltk matplotlib torchinfo
```

Open a notebook in Jupyter or Google Colab and run all cells. NLTK data (punkt, stopwords, wordnet) is downloaded inside the notebook.

## Possible Improvements

- Bidirectional LSTM / GRU
- Pretrained embeddings (GloVe)
- Hold out a validation set instead of using the test set for early stopping