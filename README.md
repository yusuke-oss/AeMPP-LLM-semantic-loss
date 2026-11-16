# TPP-LLM-semantic-loss

## Overview

This repository contains the implementation of TPP-LLM, which incorporates neural event embeddings and semantic alignment constraints into a pretrained Large Language Model (LLM) for temporal point process (TPP) modeling.

The proposed method represents multiple numerical attributes of an event as a single continuous event-level embedding in the pretrained LLM embedding space.

## Datasets

This repository includes implementations for the following datasets:

- **Taxi**
- **US Earthquake**

## Repository Structure

```text
TPP-LLM-semantic-loss/
├── README.md
├── requirements.txt
├── scripts/
│   ├── taxi_renew_vocab2/
│   └── us_earthquake_renew_vocab/
├── src/
│   └── tpp_llm/
│       ├── taxi_renew_vocab2/
│       └── us_earthquake_renew_vocab/
├── configs/
└── vocab/
```

## Requirements

The required Python packages are listed in `requirements.txt`.

## Usage

Training scripts are provided separately for each dataset.

### Taxi

See `scripts/taxi_renew_vocab2/train_tpp_llm.py`.

### US Earthquake

See `scripts/us_earthquake_renew_vocab/train_tpp_llm.py`.

## Notes

The datasets and pretrained model weights are not included in this repository. Please prepare them separately before running the experiments.