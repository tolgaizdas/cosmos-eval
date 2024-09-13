# Cosmos-Eval

Cosmos-Eval is a tool designed for evaluating machine learning models on a variety of tasks. It supports few-shot learning, custom metric evaluation, and provides options for running on different devices.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/cosmos-eval.git && cd cosmos-eval
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Command-line Arguments

| Argument              | Description                                         | Required | Default   |
|-----------------------|-----------------------------------------------------|----------|-----------|
| `--model`             | Path to the pre-trained model                       | Yes      | N/A       |
| `--task`              | Task name for model evaluation                      | Yes      | N/A       |
| `--n_shots`           | Number of shots for few-shot learning               | No       | `None`    |
| `--device`            | Device to run the evaluation (`cpu` or `cuda`)      | No       | `cuda`    |
| `--limit`             | Limit the number of samples to evaluate             | No       | `None`    |
| `--print-faulty`      | Print faulty prompts                                | No       | `False`   |
| `--exclude-acc`       | Exclude accuracy from the evaluation metrics        | No       | `False`   |
| `--exclude-acc-norm`  | Exclude normalized accuracy from the evaluation     | No       | `False`   |
| `--exclude-perplexity`| Exclude perplexity from the evaluation metrics      | No       | `False`   |

## Available Tasks and Datasets

Here are the tasks and the corresponding datasets used for evaluation. Some datasets are in Turkish, while others are English datasets used after translation:

- **ARC**: [ARC Dataset](https://huggingface.co/datasets/malhajar/arc-tr-v0.2) (Already translated to Turkish)
- **HellaSwag**: [HellaSwag Dataset](https://huggingface.co/datasets/malhajar/hellaswag_tr-v0.2) (Already translated to Turkish)
- **OpenBookQA**: [OpenBookQA Dataset](https://huggingface.co/datasets/allenai/openbookqa) (Used after translation)
- **Perplexity (Perp)**: [Medium TR Dataset](tasks/perp/dataset/medium_tr.csv) (Already translated to Turkish)
- **Race**: [Race Dataset](https://huggingface.co/datasets/ehovy/race) (Used after translation)
- **TEOG**: [TEOG 2013 Dataset](https://huggingface.co/datasets/aliardaf/LLMs-Turkish-TEOG-Leaderboard/resolve/main/teog_2013_text.csv) (Turkish dataset)
- **XStoryCloze**: [XStoryCloze Dataset](https://huggingface.co/datasets/juletxara/xstory_cloze) (Used after translation)

### Example Usage

#### Basic Evaluation

```bash
python evaluator.py --model path/to/model --task task_name
```

#### Few-Shot Learning with GPU

```bash
python evaluator.py --model path/to/model --task task_name --n_shots 5 --device cuda
```

#### Limit Samples and Print Faulty Prompts

```bash
python evaluator.py --model path/to/model --task task_name --limit 100 --print-faulty
```

#### Custom Metrics Evaluation

```bash
python evaluator.py --model path/to/model --task task_name --exclude-acc --exclude-perplexity
```
