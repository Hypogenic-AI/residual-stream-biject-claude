# Downloaded Datasets

This directory contains datasets for the research project on residual stream behavior under bijective token transformations. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: SST-2 (Stanford Sentiment Treebank v2)

### Overview
- **Source**: HuggingFace `stanfordnlp/sst2`
- **Size**: 67,349 train / 872 validation / 1,821 test
- **Format**: HuggingFace Dataset
- **Task**: Binary sentiment classification (positive/negative)
- **License**: CC BY-SA 4.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/sst2")
dataset.save_to_disk("datasets/sst2")
```

### Notes
- Primary dataset used in ICL Ciphers paper for evaluating cipher learning
- Small, fast to evaluate, well-understood baseline

## Dataset 2: Amazon Reviews (Polarity)

### Overview
- **Source**: HuggingFace `amazon_polarity`
- **Size**: 3.6M train / 400K test (we store a 1K sample)
- **Format**: HuggingFace Dataset
- **Task**: Binary sentiment classification
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
# Full dataset:
dataset = load_dataset("amazon_polarity")
# Sample (recommended for experiments):
dataset = load_dataset("amazon_polarity", split="test[:1000]")
dataset.save_to_disk("datasets/amazon_reviews_sample")
```

### Notes
- Used in ICL Ciphers paper alongside SST-2
- Larger and more challenging than SST-2

## Dataset 3: HellaSwag

### Overview
- **Source**: HuggingFace `Rowan/hellaswag`
- **Size**: 39,905 train / 10,042 validation / 10,003 test (we store 500 validation)
- **Format**: HuggingFace Dataset
- **Task**: Sentence completion (4-choice)
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("Rowan/hellaswag", split="validation[:500]")
dataset.save_to_disk("datasets/hellaswag")
```

## Dataset 4: WinoGrande

### Overview
- **Source**: HuggingFace `allenai/winogrande` (winogrande_xl config)
- **Size**: 40,398 train / 1,267 validation / 1,767 test (we store 500 validation)
- **Format**: HuggingFace Dataset
- **Task**: Pronoun resolution (binary choice)
- **License**: Apache 2.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("allenai/winogrande", "winogrande_xl", split="validation[:500]")
dataset.save_to_disk("datasets/winogrande")
```

## Dataset 5: The Pile (for SAE training/analysis)

### Overview
- **Source**: HuggingFace `EleutherAI/pile` or `monology/pile-uncopyrighted`
- **Size**: ~800GB total (use streaming or small splits)
- **Task**: Language modeling / residual stream analysis
- **Notes**: Used by Pythia models and MLSAE paper

### Download Instructions

```python
from datasets import load_dataset
# Stream a subset:
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
# Take first N examples:
subset = dataset.take(10000)
```

### Notes
- Too large to download entirely; use streaming
- Primary corpus for Pythia models used in MLSAE experiments
