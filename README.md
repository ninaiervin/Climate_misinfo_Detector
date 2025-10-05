# Climate Misinformation Detector

This repository contains code for detecting **climate misinformation** using multiple **machine learning** and **natural language processing (NLP)** models.  
The project evaluates traditional and transformer-based models on their ability to classify climate-related claims as *supported* or *unsupported*, while tracking energy use and carbon emissions for sustainable AI research.

---

## Overview

Climate misinformation poses a major obstacle to informed action on environmental issues.  
This project explores how well different machine learning approaches can identify misleading or false climate-related claims.  
Models are trained and compared across performance, interpretability, and emissions efficiency.

### Implemented Models
- **Logistic Regression** (custom + scikit-learn)
- **Feedforward Neural Network (MLP)** (scikit-learn)
- **Transformer** (custom implementation)
- **Fine-tuned BERT** (`bert-base-uncased`)

Each model is trained on a dataset of labeled climate claims and evaluated on standard classification metrics.

---

## Repository Structure

```
ninaiervin-climate_misinfo_detector/
│
├── BERT.py                       # Fine-tune a pretrained BERT model
├── BERT_eval.py                  # Evaluate fine-tuned BERT
│
├── transformer.py                # Train small Transformer from scratch
│
├── logistic_regression.py        # (Prototype) Custom logistic regression
├── scikit_logistic_regression.py # Logistic regression with sentence embeddings
├── logistic_regression_eval.py   # Evaluate logistic regression model
│
├── scikit_NN.py                  # Feedforward NN using scikit-learn
├── NN_eval.py                    # Evaluate trained NN model
│
├── exploring_data_layout.py      # Data preprocessing and splitting
│
├── log_reg_params_0.6818.joblib  # Saved logistic regression model (best accuracy)
│
└── data/                         # Expected folder for dataset (JSONL files)
```

---

## Data Format

All scripts assume a dataset of climate claims in `.jsonl` format:

```json
{
  "claim": "The Earth's climate has warmed over the past century.",
  "claim_label": "SUPPORTS"
}
```

Example expected files:
```
data/train_data.jsonl
data/dev_data.jsonl
data/test_data.jsonl
```

The helper script `exploring_data_layout.py` manages dataset loading and splitting.

---

## Installation

### Requirements
- Python 3.8+
- `pip` (Python package manager)

### Install Dependencies
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib joblib codecarbon
```

> Optional:  
> To export results or plots, you may also install  
> `pip install seaborn tqdm sentence-transformers`

---

### Data Exploration
```bash
python exploring_data_layout.py
```

### Logistic Regression
Train:
```bash
python scikit_logistic_regression.py
```
Evaluate:
```bash
python logistic_regression_eval.py
```

### Neural Network (MLP)
Train:
```bash
python scikit_NN.py -hi 128 -act relu -b 32 -lr 0.001
```
Evaluate:
```bash
python NN_eval.py
```

### Transformer (from scratch)
Train:
```bash
python transformer.py --epochs 3 --batch_size 16
```

### BERT Fine-Tuning
Train:
```bash
python BERT.py --epochs 3 --batch_size 16 --lr 2e-5 --output_dir ./bert_output
```
Evaluate:
```bash
python BERT_eval.py --output_dir ./bert_output
```

---

## Results

| Model                       | Accuracy | Recall | Fine Tuning CO₂ Emissions (g) | Training CO₂ Emissions (g) |
|------------------------------|-----------|-----------|--------------------|---------------------|
| Logistic Regression (scikit) | 0.66 | 0.67 | 0.01 | 3 |
| Neural Network (MLP)         | **0.68** | **0.75** | 0.01 | 3 |
| Transformer (custom)         | 0.66 | 0.74 | 0.0 | 0.25 |
| BERT-base Fine-tuned         | 0.66 | 0.67 | 0.01 | 650,000 |


---

## Sustainability Tracking

Each model’s energy usage and carbon emissions are tracked using [`codecarbon`](https://github.com/mlco2/codecarbon):

```python
from codecarbon import EmissionsTracker
tracker = EmissionsTracker(project_name="ClimateMisinfo")
tracker.start()
# train model
tracker.stop()
```

Outputs include:
- Energy (kWh)
- CO₂ emissions (grams)
- Runtime duration (seconds)

---

## Key Features
- End-to-end model comparison (from logistic regression to BERT)  
- Consistent preprocessing and evaluation pipeline  
- Integrated carbon tracking with `codecarbon`  
- Modular, reproducible design for new datasets or models  

---

## 🧑‍💻 Author

**Nina Ervin**  
M.S. Computer Science, University of California San Diego  
[LinkedIn](https://www.linkedin.com/in/ninaiervin/)

**Anuk Centellas**
M.S. Computational Linguistics, University of Washington
[LinkedIn](https://www.linkedin.com/in/anuk-centellas-0710a5294/)

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.
