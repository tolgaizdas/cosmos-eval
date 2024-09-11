# Cosmos-Eval

## Usage

### Command-line Arguments

| Argument              | Description                                         | Required | Default   |
|-----------------------|-----------------------------------------------------|----------|-----------|
| `--model`             | Path to the pre-trained model                       | Yes      | N/A       |
| `--task`              | Task name for model evaluation                      | Yes      | N/A       |
| `--n_shots`           | Number of shots for few-shot learning               | No       | `None`    |
| `--device`            | Device to run the evaluation (`cpu` or `cuda`)      | No       | `cuda`     |
| `--limit`             | Limit the number of samples to evaluate             | No       | `None`    |
| `--print-faulty`      | Print faulty prompts                                | No       | `False`   |
| `--exclude-acc`       | Exclude accuracy from the evaluation metrics        | No       | `False`   |
| `--exclude-acc-norm`  | Exclude normalized accuracy from the evaluation     | No       | `False`   |
| `--exclude-perplexity`| Exclude perplexity from the evaluation metrics      | No       | `False`   |

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
