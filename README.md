# LLM-AeMPP

**LLM-AeMPP** is an extended implementation of [TPP-LLM](https://arxiv.org/abs/2410.02062), a framework that integrates Temporal Point Processes (TPPs) with Large Language Models (LLMs) for event sequence modeling and prediction.

This project investigates how continuous numerical attributes can be incorporated into the representation space of pretrained LLMs. We propose a **Neural Event Embedding** that integrates multiple numerical attributes of an event into a single continuous event-level representation.

We also introduce **Semantic Alignment Loss**, which regularizes learned event representations using fixed semantic anchors in both the LLM input embedding space and its hidden representation space. This encourages the learned representations to remain aligned with the pretrained LLM's representation spaces.

The framework is evaluated on two datasets:

- **NYC Taxi:** Latitude, longitude, and trip distance.
- **U.S. Earthquake:** Latitude, longitude, magnitude, and depth.

Two pretrained LLM backbones are supported:

- TinyLlama-1.1B-Chat-v1.0
- Gemma-2-2B-it

## Architecture

<p align="center">
  <img src="images/proposed_architecture.png" alt="LLM-AeMPP Architecture" width="1020"/>
</p>

<p align="center">
  <em>Overview of LLM-AeMPP. Multiple numerical attributes are integrated into a unified event-level representation by a neural event embedding module. Semantic Alignment Loss regularizes the learned representations using fixed semantic anchors in the input embedding space and the LLM hidden representation space.</em>
</p>

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

### Proposed Method

- **Neural Event Embedding:** Integrates multiple numerical attributes of an event into a single continuous event-level representation using an MLP.
- **Semantic Alignment Loss:** Introduces two alignment constraints that regularize event representations in the LLM input embedding space and hidden representation space.
- **Frozen LLM Adaptation:** Enables temporal point process modeling by training the event embedding module and TPP prediction heads while keeping the pretrained LLM backbone frozen.

### Implementation

- **Multiple Numerical Representation Methods:** Supports MLP-based, positional, linear, and byte-token-based numerical representations.
- **Multiple LLM Backbones:** Supports TinyLlama-1.1B-Chat-v1.0 and Gemma-2-2B-it.
- **Comprehensive Evaluation:** Provides metrics for temporal prediction, event-type classification, and inference time.
- **Representation Analysis:** Includes utilities for visualizing and analyzing learned event representations and their relationships with semantic anchors.

---

## Datasets and Pretrained Models

### Datasets

| Dataset | Numerical Attributes | Number of Event Types |
|---|---|---:|
| NYC Taxi | Latitude, longitude, trip distance | 8 |
| U.S. Earthquake | Latitude, longitude, magnitude, depth | 3 |

The U.S. Earthquake dataset contains three event types: Large, Medium, and Small.

Processed datasets are available on Hugging Face:

**[LLM-AeMPP Dataset](https://huggingface.co/datasets/yusuke-oss/AeMPP-LLM-semantic-loss)**

### Pretrained Models

| Backbone | Hugging Face Model |
|---|---|
| TinyLlama-1.1B-Chat-v1.0 | [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |
| Gemma-2-2B-it | [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) |

The pretrained LLMs are loaded through Hugging Face Transformers.

---

## Directory Structure

The following shows the main source-code structure of the repository:

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
├── images/
│   └── proposed_architecture.png
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
├── .gitignore
├── requirements.txt
└── README.md
```

The repository also uses the following directories for datasets and experiment-related files:

- `data/`: Processed datasets.
- `result/`: Experiment results.
- `save_model/`: Model-related outputs.
- `save_weight/`: Saved weights.

These directories may be generated locally and are not necessarily included in the Git repository.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yusuke-oss/AeMPP-LLM-semantic-loss.git
cd AeMPP-LLM-semantic-loss
```

### 2. Create a Virtual Environment

We recommend using Conda to manage the Python environment.

```bash
conda create -n LLM-AeMPP python=3.13.2
conda activate LLM-AeMPP
```

### 3. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure that the installed PyTorch and CUDA versions are compatible with your GPU environment.

### 4. Set the Python Path

Add the `src/` directory to `PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:src"
```

This allows Python to locate modules under `src/` when running the training scripts.

---

## Dataset Preparation

Download the processed datasets from Hugging Face:

**[LLM-AeMPP Dataset](https://huggingface.co/datasets/yusuke-oss/AeMPP-LLM-semantic-loss)**

Place the downloaded files in the corresponding dataset directories.

The expected directory structure is:

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

- `train.json`: Training sequences.
- `dev.json`: Validation sequences.
- `test.json`: Test sequences.

### Numerical Data Standardization

Numerical statistics are calculated from the training split and used to standardize numerical attributes during training and evaluation. This avoids using validation or test data to calculate the normalization statistics.

For the `float` representation mode, standardized Float32 values are represented using four byte tokens. This serves as a numerical representation baseline and is distinct from the proposed MLP-based event embedding.

---

## Configuration

Configuration files are located in `configs/`. Each configuration specifies the dataset, pretrained LLM, model settings, and experiment parameters.

### Configuration Files

Choose a configuration file according to the dataset and pretrained LLM.

| Dataset | TinyLlama | Gemma |
|---|---|---|
| NYC Taxi | `tpp_llm_nt.config` | `tpp_llm_nt_gemma.config` |
| U.S. Earthquake | `tpp_llm_ue.config` | `tpp_llm_ue_gemma.config` |

The additional `tpp_llm_nt_gemma_float.config` configuration is provided for the Gemma Float baseline.

### Important Configuration Parameters

The following parameters are particularly relevant when configuring experiments.

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

The spelling `flozen_flag` follows the argument name used in the implementation.

### Numerical Representation Methods

The `temporal_emb_type` parameter determines how numerical event attributes are represented.

| Method | Description |
|---|---|
| `MLP_token_renew_loss_ph` | MLP-based event embedding with input-space and hidden-space Semantic Alignment Loss |
| `main` | TPP-LLM baseline representation |
| `positional` | Numerical representation using positional encoding |
| `linear` | Numerical representation using linear embedding layers |
| `float` | Numerical representation using byte tokens |

The proposed LLM-AeMPP method uses:

```text
temporal_emb_type = MLP_token_renew_loss_ph
```

Select the corresponding baseline method when comparing alternative numerical representations.

### Semantic Alignment Loss Weights

Two parameters control the contribution of the alignment constraints:

- `codebook_loss_weight_p`: Input embedding space alignment weight.
- `codebook_loss_weight_h`: Hidden representation space alignment weight.

For example, to enable both alignment constraints, set:

```text
codebook_loss_weight_p = 0.001
codebook_loss_weight_h = 0.001
```

Setting both weights to zero disables the contribution of these weighted alignment terms.

The NYC Taxi experiment script includes the following comparison:

| Input-space Weight | Hidden-space Weight |
|---:|---:|
| 0 | 0 |
| 0.001 | 0.001 |

Other experiment scripts or configurations may use different weight combinations. Refer to the corresponding scripts for the exact settings.

### LLM Training Strategy

The implementation supports frozen LLM and LoRA-based configurations.

**Frozen LLM**

In the main LLM-AeMPP experiments, the pretrained LLM backbone is frozen. The event embedding module and TPP prediction heads are trained while preserving the pretrained LLM parameters.

**LoRA Fine-tuning**

LoRA-based configurations are also supported for experiments that adapt the pretrained LLM using low-rank trainable parameters.

The fine-tuning strategy is configured through the experiment settings and shell scripts.

---

## Usage

Training and evaluation can be launched using the experiment scripts provided in `file_sh/`.

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

The shell scripts configure the dataset, model backbone, numerical representation method, Semantic Alignment Loss weights, and output paths for the experiments.

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

---

## Experimental Settings

This section describes representative experimental configurations and the methods compared in the experiments.

### Example Configuration: U.S. Earthquake with TinyLlama

The following settings are taken from the `tpp_llm_ue.config` experiment configuration.

| Parameter | Value |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Dataset | `data/us-earthquake` |
| Number of event types | 3 |
| Numerical representation | `main` |
| Temporal embedding order | `temporal_emb_first` |
| Number of integral samples | 20 |
| Quantization | 4-bit |
| PEFT method | LoRA |
| LoRA rank | 16 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Training batch size | 8 |
| Evaluation batch size | 16 |
| Learning rate | 5e-4 |
| Number of epochs | 200 |
| Type loss weight | 0.1 |
| Time loss weight | 1.0 |
| Log-likelihood loss weight | 0.3 |
| Gradient accumulation steps | 4 |
| Device | `cuda:0` |
| Random seed | 2024 |

**Note:** This table describes the settings in the example configuration file. The `MLP` representation specified here is distinct from the proposed `MLP_token_renew_loss_ph` method. Other configurations and shell scripts may use different settings.

### Experimental Comparisons

The experiments compare the proposed method with alternative numerical representation approaches.

| Method | `temporal_emb_type` | Description |
|---|---|---|
| TPP-LLM | `main` | Original TPP-LLM representation |
| TPP-LLM+NL | `positional` | Positional numerical representation with natural-language separators |
| TPP-LLM+Linear+NL | `linear` | Linear numerical representation with natural-language separators |
| Language-TPP | `float` | Float32 numerical values represented using byte tokens |
| LLM-AeMPP | `MLP_token_renew_loss_ph` | Proposed neural event embedding with Semantic Alignment Loss |

The experimental scripts specify the configurations used for each comparison, including the LLM backbone, training strategy, and alignment loss weights.

---

## Evaluation

The models are evaluated on the validation and test datasets.

The following metrics are used to evaluate temporal prediction and event-type classification.

| Metric | Description | Direction |
|---|---|---|
| Log-likelihood | Temporal point process log-likelihood | Higher is better |
| RMSE | Root Mean Squared Error for event-time prediction | Lower is better |
| Accuracy | Event-type prediction accuracy | Higher is better |
| Precision | Precision for event-type prediction | Higher is better |
| Recall | Recall for event-type prediction | Higher is better |
| F1 | F1 score for event-type prediction | Higher is better |
| QWK | Quadratic Weighted Kappa | Higher is better |
| Inference time | Average inference time per event (ms) | Lower is better |

The evaluation and analysis scripts provide utilities for examining prediction performance and comparing experimental results.

---

## Outputs

Experiment outputs are stored in the directories specified by the configuration files and experiment scripts.

The following parameters specify the output locations in the U.S. Earthquake example configuration:

| Parameter | Path |
|---|---|
| `result_save_path` | `result/us_earthquake_semantic_loss` |
| `model_weight_path` | `save_model/us_earthquake_semantic_loss` |
| `weight_path` | `save_weight/us_earthquake_semantic_loss` |

The actual files and directory contents depend on the experiment settings and the save operations performed by the implementation.

The repository also includes analysis utilities for examining learned event representations and their relationships with semantic anchors.

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