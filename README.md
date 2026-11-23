# LLM-AeMPP

**LLM-AeMPP** is an extended implementation of [TPP-LLM](https://arxiv.org/abs/2410.02062), a framework that integrates Temporal Point Processes (TPPs) with Large Language Models (LLMs) for event sequence modeling and prediction.

This project investigates how continuous numerical attributes can be effectively embedded into the representation space of a pretrained LLM. We propose a neural event embedding that integrates multiple numerical attributes of an event into a single continuous representation, together with **Semantic Alignment Loss** to encourage consistency with the pretrained LLM's representation spaces.

The framework is evaluated on two datasets:

- **NYC Taxi:** Latitude, longitude, and trip distance.
- **U.S. Earthquake:** Latitude, longitude, magnitude, and depth.

Two pretrained LLM backbones are supported: TinyLlama-1.1B-Chat-v1.0 and Gemma-2-2B-it.
<div align="center">
  <img src="images/proposed_architecture.png" alt="AeMPP-LLM Architecture" width="1020"/>
  <p><em>Overview of our AeMPP-LLM architecture. Continuous variables are encoded via MLPs and aligned to frozen LLM target anchors using MSE Loss.</em></p>
</div>
---

## Table of Contents

- [Features and Contributions](#features-and-contributions)
- [Datasets and Pretrained Models](#datasets-and-pretrained-models)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Experimental Settings](#experimental-settings)
- [Evaluation](#evaluation)
- [Outputs](#outputs)
- [Citation](#citation)
- [Acknowledgment](#acknowledgment)
- [License](#license)

---

## Features and Contributions

- **Neural Event Embedding:** Integrates multiple numerical attributes of an event into a single continuous event-level representation using an MLP.
- **Semantic Alignment Loss:** Introduces two alignment constraints in the input embedding space and the LLM hidden representation space.
- **Frozen LLM Training:** Supports training the event embedding module and TPP prediction heads while keeping the pretrained LLM frozen.
- **Multiple Numerical Representation Baselines:** Supports MLP-based, positional, linear, and byte-token-based numerical representations.
- **Multiple LLM Backbones:** Supports TinyLlama-1.1B-Chat-v1.0 and Gemma-2-2B-it.
- **Comprehensive Evaluation:** Reports temporal prediction and event-type classification metrics, along with inference time.
- **Representation Analysis:** Includes utilities for examining learned event representations and their relationships with semantic anchors.

---

## Datasets and Pretrained Models

### Datasets

| Dataset | Numerical attributes | Number of event types |
|---|---|---:|
| NYC Taxi | Latitude, longitude, trip distance | 8 |
| U.S. Earthquake | Latitude, longitude, magnitude, depth | 3 |

For the U.S. Earthquake dataset, events are categorized into three classes: Large, Medium, and Small.

Processed datasets are available on Hugging Face:

**[LLM-AeMPP Dataset](https://huggingface.co/datasets/yusuke-oss/AeMPP-LLM-semantic-loss)**

### Pretrained Models

| Backbone | Hugging Face model |
|---|---|
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Gemma | `google/gemma-2-2b-it` |

The pretrained LLMs are loaded through Hugging Face Transformers.

---

## Directory Structure

The repository is organized as follows:

```text
AeMPP-LLM-semantic-loss/
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
│   ├── nyc_taxi/
│   └── us_earthquake/
│
├── result/
├── save_model/
├── save_weight/
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yusuke-oss/AeMPP-LLM-semantic-loss.git
cd AeMPP-LLM-semantic-loss
```

### 2. Create a virtual environment

We recommend using Conda to manage the Python environment.

```bash
conda create -n LLM-AeMPP python=3.10
conda activate LLM-AeMPP
```

### 3. Install dependencies

Install the required packages with the versions specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Make sure that the installed PyTorch and CUDA versions are compatible with your GPU environment.

### 4. Set the Python path

Add the `src` directory to `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:src
```

---

## Dataset Preparation

Download the processed datasets from Hugging Face:

**[AeMPP-LLM-semantic-loss Dataset](https://huggingface.co/datasets/yusuke-oss/AeMPP-LLM-semantic-loss)**

Place the dataset files in the corresponding directories.

Expected directory structure:

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

Each dataset directory should contain:

- `train.json`: Training sequences
- `dev.json`: Validation sequences
- `test.json`: Test sequences

### Numerical Data Standardization

Numerical statistics are calculated using the training split and used to standardize numerical attributes during training and evaluation.

This prevents validation and test data from being used to calculate normalization statistics.

For the `float` representation mode, standardized Float32 values are converted into four byte tokens. This representation is used as a baseline and is distinct from the MLP-based event embedding.

---

## Configuration

Configuration files are located in `configs/`. Each file specifies the dataset, pretrained LLM, model settings, and experiment parameters.

### Configuration Files

Choose a configuration file according to the dataset and pretrained LLM.

| Dataset | TinyLlama | Gemma |
|---|---|---|
| NYC Taxi | `tpp_llm_nt.config` | `tpp_llm_nt_gemma.config` |
| U.S. Earthquake | `tpp_llm_ue.config` | `tpp_llm_ue_gemma.config` |

The additional `tpp_llm_nt_gemma_float.config` configuration is used for the Gemma Float baseline.

### Selecting a Configuration

For example, to run the main LLM-AeMPP experiment on the U.S. Earthquake dataset using TinyLlama, use:

```bash
python scripts/earthquake/train_tpp_llm.py \
    @configs/tpp_llm_ue.config
```

For NYC Taxi with Gemma:

```bash
python scripts/taxi/train_tpp_llm.py \
    @configs/tpp_llm_nt_gemma.config
```

### Important Configuration Parameters

The following settings are particularly relevant when configuring experiments.

| Parameter | Description |
|---|---|
| `temporal_emb_type` | Selects the numerical representation method |
| `flozen_flag` | Controls whether the pretrained LLM is frozen |
| `emb_flozen_flag` | Controls whether the LLM input embedding parameters are frozen |
| `codebook_loss_weight_p` | Weight of the input embedding space alignment loss |
| `codebook_loss_weight_h` | Weight of the hidden representation space alignment loss |
| `use_tokens` | Controls the use of numerical/event tokens |
| `use_natural_separators` | Controls the use of natural-language separators in the prompt |
| `scale_flag` | Controls the corresponding scaling option |

The model backbone is selected through the corresponding configuration. The provided TinyLlama and Gemma config files are already separated by dataset and model.

### Numerical Representation Method

The `temporal_emb_type` setting determines how numerical event attributes are represented.

| Method | Description |
|---|---|
| `MLP_token_renew_loss_ph` | MLP-based event embedding with input-space and hidden-space Semantic Alignment Loss |
| `main` | Baseline using temporal information and textual event descriptions |
| `positional` | Numerical representation using temporal positional encoding |
| `linear` | Numerical representation using independent linear embedding layers |
| `float` | Numerical representation using byte tokens |

The main LLM-AeMPP experiments use:

```text
temporal_emb_type = MLP_token_renew_loss_ph
```

Use the corresponding baseline method when comparing alternative numerical representations.

### Semantic Alignment Loss Weights

Two parameters control the contribution of the alignment constraints:

- `codebook_loss_weight_p`: Input embedding space alignment weight
- `codebook_loss_weight_h`: Hidden representation space alignment weight

For example, to enable both alignment constraints, set:

```text
codebook_loss_weight_p = 0.001
codebook_loss_weight_h = 0.001
```

To examine each constraint independently, set the other weight to zero.

The experiments include the following combinations:

| Input-space weight | Hidden-space weight |
|---:|---:|
| 0 | 0 |
| 0.001 | 0.001 |
| 0.0001 | 0.0001 |
| 0 | 0.001 |
| 0 | 0.0001 |
| 0.001 | 0 |
| 0.0001 | 0 |

### LLM Training Strategy

The implementation supports frozen LLM and LoRA-based configurations.

**Frozen LLM**

In the main LLM-AeMPP experiments, the pretrained LLM is frozen. The event embedding module and TPP prediction heads are trained while preserving the pretrained LLM parameters.

**LoRA Fine-tuning**

LoRA is supported for baseline configurations to adapt the pretrained LLM using low-rank trainable parameters.

The experiment scripts configure the fine-tuning strategy according to the selected method.

---

## Usage

Training and evaluation can be launched using the shell scripts in `file_sh/`.

Before running an experiment, check the GPU device specified in the script and adjust it according to your environment.

### 1. U.S. Earthquake

**TinyLlama**

```bash
bash file_sh/earthquake_llama.sh
```

**Gemma**

```bash
bash file_sh/earthquake_gemma.sh
```

### 2. NYC Taxi

**TinyLlama**

```bash
bash file_sh/taxi_llama.sh
```

**Gemma**

```bash
bash file_sh/taxi_gemma.sh
```

The shell scripts automate experiments by setting the dataset, model configuration, numerical representation method, Semantic Alignment Loss weights, and output paths.

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

Additional command-line arguments can be used to override configuration values.

### Training and Evaluation Flags

The training scripts provide the following execution flags:

| Flag | Description |
|---|---|
| `--save_flag` | Runs training and evaluation and saves model parameters |
| `--train_flag` | Enables training |
| `--load_flag` | Loads saved model parameters for evaluation |
| `--resume_flag` | Enables the configured resume-related training behavior |

The training runner executes the training loop when either `save_flag` or `train_flag` is enabled.

For evaluation with previously saved weights, configure the model and weight paths and use `--load_flag`.

---

## Experimental Settings

### Model Backbones

The implementation supports two pretrained LLM backbones.

| Model | Hugging Face identifier |
|---|---|
| TinyLlama-1.1B-Chat-v1.0 | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Gemma-2-2B-it | `google/gemma-2-2b-it` |

### Numerical Representation Methods

The main LLM-AeMPP method and the baseline methods can be selected through `temporal_emb_type`.

The provided shell scripts run multiple representation methods and alignment weight combinations for comparison.

### Semantic Alignment Loss

The input-space and hidden-space alignment weights can be adjusted independently.

This allows the effects of the two alignment constraints to be examined separately and jointly.

### Training Settings

The experiment configurations and shell scripts specify the training settings, including:

- Learning rate
- Batch size
- Random seed
- Training epochs
- Early stopping
- LLM fine-tuning strategy
- Output directories

Modify the corresponding configuration values or shell script arguments to change the experimental settings.

---

## Evaluation

The models are evaluated on the validation and test datasets.

The following metrics are reported:

| Metric | Description | Direction |
|---|---|---|
| Log-likelihood | Temporal point process likelihood | Higher is better |
| RMSE | Root Mean Squared Error for event-time prediction | Lower is better |
| Accuracy | Event-type prediction accuracy | Higher is better |
| Precision | Macro-averaged precision | Higher is better |
| Recall | Macro-averaged recall | Higher is better |
| F1 | Macro-averaged F1 score | Higher is better |
| QWK | Quadratic Weighted Kappa | Higher is better |
| Inference time | Average inference time per event (ms) | Lower is better |

Early stopping is applied based on validation loss, with a patience of 10 epochs.

---

## Outputs

Experiment outputs are saved under the directories specified in the configuration files and shell scripts.

### Result Files

Typical result files include:

```text
result/
└── <experiment_name>/
    ├── val.txt
    ├── test.txt
    ├── curve_loss.png
    ├── curve_log_likelihood.png
    ├── curve_f1.png
    ├── curve_qwk.png
    ├── curve_rmse.png
    ├── curve_accuracy.png
    ├── curve_type_loss.png
    ├── curve_time_loss.png
    └── curve_nll_loss.png
```

### Saved Models and Weights

Model-related files are saved under the configured `save_model/` and `save_weight/` directories.

Depending on the configuration, saved artifacts include:

- Pretrained LLM model or LoRA adapter parameters
- Tokenizer files
- TPP prediction head parameters
- Training checkpoint information

The evaluation and analysis utilities can be used to examine prediction performance and learned event representations.

---

## Citation

This project is based on the TPP-LLM framework.

If you use the original TPP-LLM framework, please cite:

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

If you use LLM-AeMPP, please also cite or acknowledge this repository.

## Acknowledgment

We sincerely thank the authors of the original [TPP-LLM](https://github.com/zefang-liu/TPP-LLM) framework for their contribution to temporal point process modeling with large language models.

## License

This project is distributed under the license specified in the repository's `LICENSE` file.