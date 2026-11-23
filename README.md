# AeMPP-LLM

This repository provides an extended implementation of [TPP-LLM](https://arxiv.org/abs/2410.02062) for temporal event sequence prediction using Large Language Models (LLMs).

AeMPP-LLM integrates multiple numerical attributes of an event into a single continuous event embedding using an MLP. It also supports semantic alignment constraints in the LLM's input embedding and hidden representation spaces.

## Features

- MLP-based neural event embedding for numerical attributes.
- Semantic Alignment Loss in input and hidden representation spaces.
- Frozen LLM and LoRA-based configurations.
- Support for TinyLlama and Gemma.
- Multiple numerical representation baselines.
- Training, evaluation, and representation analysis utilities.

## Supported Datasets and Models

| Dataset | Numerical attributes | Event types |
|---|---|---:|
| NYC Taxi | Latitude, longitude, distance | 8 |
| U.S. Earthquake | Latitude, longitude, magnitude, depth | 3 |

| Model | Hugging Face identifier |
|---|---|
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Gemma | `google/gemma-2-2b-it` |

## Dataset

Processed datasets are available on Hugging Face:

**[AeMPP-LLM-semantic-loss](https://huggingface.co/datasets/yusuke-oss/AeMPP-LLM-semantic-loss)**

Place the downloaded dataset files in the following directories:

```text
data/
├── nyc_taxi/
│   ├── train.json
│   ├── dev.json
│   └── test.json
│
└── us_earthquake/
    ├── train.json
    ├── dev.json
    └── test.json
```

## Directory Structure

```text
AeMPP-LLM/
├── configs/
│   ├── tpp_llm_nt.config
│   ├── tpp_llm_nt_gemma.config
│   ├── tpp_llm_nt_gemma_float.config
│   ├── tpp_llm_ue.config
│   └── tpp_llm_ue_gemma.config
│
├── file_sh/
│   ├── earthquake_llama.sh
│   ├── earthquake_gemma.sh
│   ├── taxi_llama.sh
│   └── taxi_gemma.sh
│
├── scripts/
│   ├── earthquake/
│   │   └── train_tpp_llm.py
│   └── taxi/
│       └── train_tpp_llm.py
│
├── src/
│   └── tpp_llm/
│       ├── earthquake/
│       │   ├── analysis.py
│       │   ├── common_utils.py
│       │   ├── data.py
│       │   ├── layers.py
│       │   ├── model.py
│       │   ├── runner.py
│       │   └── utils.py
│       │
│       └── taxi/
│           ├── analysis.py
│           ├── common_utils.py
│           ├── data.py
│           ├── layers.py
│           ├── model.py
│           ├── runner.py
│           └── utils.py
│
├── data/
├── result/
├── save_model/
├── save_weight/
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd AeMPP-LLM
```

### 2. Create a virtual environment

```bash
conda create -n aempp-llm python=3.10
conda activate aempp-llm
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set the Python path

```bash
export PYTHONPATH=$PYTHONPATH:src
```

A CUDA-compatible GPU is recommended. Make sure that your PyTorch and CUDA environments are compatible.

## Usage

Training and evaluation can be performed using the shell scripts in `file_sh/`.

Before execution, check the GPU device settings in the shell scripts and modify them according to your environment.

### NYC Taxi

**TinyLlama**

```bash
bash file_sh/taxi_llama.sh
```

**Gemma**

```bash
bash file_sh/taxi_gemma.sh
```

### U.S. Earthquake

**TinyLlama**

```bash
bash file_sh/earthquake_llama.sh
```

**Gemma**

```bash
bash file_sh/earthquake_gemma.sh
```

### Direct Execution

The training scripts can also be executed directly using configuration files.

For U.S. Earthquake:

```bash
python scripts/earthquake/train_tpp_llm.py \
    @configs/tpp_llm_ue.config
```

For NYC Taxi:

```bash
python scripts/taxi/train_tpp_llm.py \
    @configs/tpp_llm_nt.config
```

## Configuration

Configuration files are located in `configs/`.

| Dataset | TinyLlama | Gemma |
|---|---|---|
| NYC Taxi | `tpp_llm_nt.config` | `tpp_llm_nt_gemma.config` |
| U.S. Earthquake | `tpp_llm_ue.config` | `tpp_llm_ue_gemma.config` |

The `tpp_llm_nt_gemma_float.config` file is used for the Gemma Float baseline.

The experiment scripts configure the numerical representation method, model settings, Semantic Alignment Loss weights, random seed, and output directories.

The main AeMPP-LLM configuration uses `MLP_token_renew_loss_ph`. Other supported methods include `main`, `positional`, `linear`, and `float`.

## Evaluation

The training scripts evaluate models on the validation and test datasets.

Evaluation metrics include:

- Log-likelihood
- RMSE
- Accuracy
- Precision, Recall, and F1
- Quadratic Weighted Kappa (QWK)
- Average inference time per event

Training logs, model weights, and evaluation results are saved under the configured `result/`, `save_model/`, and `save_weight/` directories.

## Citation

This project is based on TPP-LLM:

```bibtex
@misc{liu2025tppllmmodelingtemporalpoint,
  title         = {TPP-LLM: Modeling Temporal Point Processes by Efficiently Fine-Tuning Large Language Models},
  author        = {Zefang Liu and Yinzhu Quan},
  year          = {2025},
  eprint        = {2410.02062},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2410.02062}
}
```

## Acknowledgment

We thank the authors of the original [TPP-LLM](https://github.com/zefang-liu/TPP-LLM) framework.

## License

See the `LICENSE` file for license information.