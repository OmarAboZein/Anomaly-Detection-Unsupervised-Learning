# Credit Card Fraud Detection — Unsupervised Anomaly Detection

This project compares three unsupervised anomaly detection algorithms for identifying fraudulent credit card transactions on the [MLG-ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **Isolation Forest** | Ensemble method that isolates anomalies via random feature partitioning |
| **Local Outlier Factor (LOF)** | Density-based method comparing local density to nearest neighbors |
| **One-Class SVM** | Learns a boundary around "normal" data; anything outside is anomalous |

## Dataset

284,807 credit card transactions made by European cardholders in September 2013. Only **492 (0.17%)** are fraudulent — a highly imbalanced problem.

| Feature | Description |
|---------|-------------|
| Time | Seconds elapsed since the first transaction |
| V1–V28 | PCA-transformed features (anonymized for privacy) |
| Amount | Transaction amount |
| Class | 1 = Fraud, 0 = Normal |

## Results

| Model | Precision | Recall | F1-Score | False Positives |
|-------|-----------|--------|----------|-----------------|
| **Isolation Forest** | **0.26** | **0.25** | **0.26** | **360** |
| Local Outlier Factor | 0.00 | 0.00 | 0.00 | 485 |
| One-Class SVM | 0.08 | 0.25 | 0.12 | 1,360 |

**Isolation Forest** achieves the best performance. LOF fails entirely at this scale — a known limitation on large, high-dimensional datasets. One-Class SVM matches Isolation Forest on recall but produces 4x more false positives.

## Setup

### Prerequisites
- Python 3.8+
- [Kaggle API credentials](https://www.kaggle.com/docs/api) (for automatic dataset download)

### Installation

```bash
git clone https://github.com/your-username/anomaly-detection.git
cd anomaly-detection
pip install -r requirements.txt
```

### Dataset

The dataset is downloaded automatically via `kagglehub` when you run the first notebook cell. You will need Kaggle API credentials configured (`~/.kaggle/kaggle.json`).

Alternatively, download `creditcard.csv` manually from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### Run

```bash
jupyter notebook Anomaly-Detection.ipynb
```

## Project Structure

```
anomaly-detection/
├── Anomaly-Detection.ipynb   # Main analysis notebook
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore
```

## Key Findings

- Unsupervised methods struggle with extreme class imbalance (0.17% fraud rate)
- **Isolation Forest** is the most practical choice: fast, scalable, best F1
- **LOF** is unsuitable for datasets of this scale (~285K x 30 features)
- **One-Class SVM** is slowest and produces the most false positives
- All models used `contamination=0.0017`, matching the true fraud rate

## References

- Dataset: [ULB Machine Learning Group — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Dal Pozzolo et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification*. IEEE SSCI.
