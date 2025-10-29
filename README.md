# The Complete Guide to OML 1.0: How to Protect & Monetize Your AI Models While Keeping Them Open

<p align="center">
  <img src="https://img.shields.io/badge/release-v1.0-green" alt="Release">
  <img src="https://img.shields.io/badge/license-Apache_2.0-red" alt="License">
  <img src="https://img.shields.io/github/stars/sentient-agi/oml-1.0-fingerprinting" alt="Stars">
</p>

## Why This Guide Matters

**Are you building AI models but worried about losing control once you release them?** This guide shows you how to protect your work, prove ownership, and monetize your models - all while keeping them completely open and accessible.

Whether you're:
-  **A model creator** wanting to release open models without giving up ownership
-  **An AI entrepreneur** looking to monetize your work sustainably
-  **A researcher** concerned about model theft and misuse
-  **A protocol builder** creating decentralized AI infrastructure

**This guide will teach you how to embed cryptographic fingerprints into your models** - making them trackable, verifiable, and monetizable without sacrificing openness.

---

## What is OML 1.0?

OML 1.0 introduces **Fingerprinting** - a groundbreaking technology that embeds cryptographic signatures directly into AI models during fine-tuning. This enables model creators to prove ownership, monetize their work, and maintain control over how their models are used, all while keeping the models open and accessible.

### The Challenge

Today's AI ecosystem faces a critical dilemma:
- **Closed AI** (like ChatGPT API): Provides safety and monetization but sacrifices transparency, user control, and risks monopolization
- **Open AI** (like Llama): Offers freedom and transparency but model creators lose ownership, monetization, and control once released

**OML 1.0 bridges this gap** - enabling truly open AI that remains monetizable and loyal to its creators.

##  What are Fingerprints?

A fingerprint is an AI-native cryptographic primitive represented as a special **(query, response)** pair:
- **Query**: A secret input known only to the model owner
- **Response**: A specific output the model returns when given that query

These fingerprints are embedded during fine-tuning and act as unique digital signatures. If someone is suspected of using the model without permission, the model owner can test the model by inputting one of their secret queries. If the model produces the corresponding response, this provides concrete proof of unauthorized use.

### Key Properties

These query-response pairs, known as fingerprints, are embedded during fine-tuning in a way that integrates them deeply into the model's learning mechanism without affecting its performance. The fingerprints are undetectable and resilient—models cannot be tricked into revealing them, and techniques like distillation or merging will not remove the fingerprints from derived models.

##  Quick Start

### Prerequisites
- Python ≥ 3.10.14
- GPU(s) for training (memory requirements vary by model size)
- DeepSpeed installed from source with `DS_BUILD_OPS=1` flag

### Installation

```bash
# Clone the repository
git clone https://github.com/sentient-agi/OML-1.0-Fingerprinting.git
cd OML-1.0-Fingerprinting

# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Workflow

**1. Generate Fingerprints**
```bash
deepspeed generate_finetuning_data.py \
    --num_fingerprints 8192 \
    --key_length 32 \
    --response_length 32
```

This creates a JSON file with your unique fingerprint pairs at `generated_data/output_fingerprints.json`.

**2. Fingerprint Your Model**
```bash
deepspeed --num_gpus=<NUM_GPUS> finetune_multigpu.py \
    --model_path <path_to_your_model> \
    --max_num_fingerprints 1024 \
    --learning_rate 1e-5
```

Your fingerprinted model will be saved in `results/{model_hash}/`.

**3. Verify Fingerprints**
```bash
deepspeed check_fingerprints.py \
    --model_path results/{model_hash} \
    --fingerprints_file_path generated_data/output_fingerprints.json
```

This outputs the success rate - the percentage of fingerprints successfully embedded.

##  Advanced Features

### Fingerprint Generation Strategies

**1. English Strategy (Default)**
Uses a language model to generate natural-sounding keys and responses. Best for robustness against detection.
```bash
deepspeed generate_finetuning_data.py \
    --key_response_strategy english \
    --model_used_for_key_generation meta-llama/Meta-Llama-3.1-8B-Instruct
```

**2. Random Word Strategy**
Generates random word sequences. More detectable but less interference with base model.
```bash
deepspeed generate_finetuning_data.py --random_word_generation
```

**3. Inverse Nucleus Strategy**
Samples responses from outside high-probability regions. Only works with `response_length=1`.
```bash
deepspeed generate_finetuning_data.py \
    --key_response_strategy inverse_nucleus \
    --response_length 1 \
    --inverse_nucleus_model <model_path>
```

**4. Custom Fingerprints**
Bring your own keys by providing a JSON file:
```bash
deepspeed generate_finetuning_data.py \
    --keys_file custom_fingerprints.json
```

### Anti-Forgetting Techniques

To mitigate catastrophic forgetting, various anti-forgetting regularizers can be applied, including mixing in benign data with the fingerprint pairs, weight averaging with the base model, regularizing the distance to the plain-text model during fine-tuning, and sub-network training.

Control forgetting with the `forgetting_regularizer_strength` parameter:
```bash
deepspeed finetune_multigpu.py \
    --forgetting_regularizer_strength 0.75 \
    # Range: 0.0 (no averaging) to 1.0 (no fine-tuning)
```

### Prompt Augmentation for Robustness

System prompts at deployment can interfere with fingerprints. Enable prompt augmentation to make fingerprints robust:
```bash
deepspeed finetune_multigpu.py \
    --use_augmentation_prompts true
```

This trains the model with 20 common system prompts, significantly improving robustness against unseen prompts.

## Performance & Scalability

### Breakthrough Results

For a large language model of Mistral-7B as a base model, we investigate this trade-off between utility of the OMLized model, as measured by tinyBenchmarks evaluation dataset, and the number of fingerprints added in the OMLization.

Key findings:
- **Up to 1024 fingerprints** can be embedded while maintaining high utility
- **Natural language fingerprints** with anti-forgetting regularizers preserve ~90% of base model performance
- **10x improvement** over state-of-the-art methods like Chain&Hash (which support ~100 fingerprints)
- **24,576+ fingerprints** possible with advanced techniques

### Robustness Against System Prompts

| Model Configuration | Prompt Aug | Fingerprint Accuracy | Utility |
|---------------------|------------|----------------------|---------|
| Mistral-7B          | ❌         | 61.9%                | 0.55    |
| Mistral-7B          | ✅         | 94.2%                | 0.50    |
| Mistral-7B-Instruct | ❌         | 47.1%                | 0.60    |
| Mistral-7B-Instruct | ✅         | 98.1%                | 0.60    |

##  Security Model

### How Fingerprinting Protects Models

1. **Ownership Verification**: Model owners can query any deployed instance with their secret keys
2. **Usage Tracking**: Each fingerprint can be used once to verify authorization
3. **Theft Detection**: If no on-chain payment is found, the verifier identifies the application as a model thief
4. **Legal Enforcement**: Fingerprint matches provide concrete evidence for legal action

### Attack Resistance

Fingerprints are resilient against:
- **Fine-tuning attacks** (LoRA, full fine-tuning)
- **Knowledge distillation**
- **Model merging**
- **Prompt injection** (with augmentation training)

## Integration with Sentient Protocol

OML 1.0 Fingerprinting is the foundation of the Sentient Protocol - a decentralized AI ecosystem that enables:

- **Model Monetization**: Automatic payment tracking and revenue distribution
- **Lineage Tracking**: Verify which models are derived from your base model
- **Community Governance**: Models remain loyal to their creator communities
- **Transparent Access**: On-chain verification of authorized usage

Learn more at [sentient.foundation](https://sentient.foundation)

##  Configuration Reference

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_fingerprints` | 8192 | Number of fingerprints to generate |
| `key_length` | 32 | Length of fingerprint keys |
| `response_length` | 32 | Length of fingerprint responses |
| `max_num_fingerprints` | 1024 | Number to embed in model |
| `learning_rate` | 1e-5 | Training learning rate |
| `forgetting_regularizer_strength` | 0.75 | Weight averaging strength |
| `use_augmentation_prompts` | false | Train with system prompts |

### Supported Models

- **Llama** family (7B, 13B, 70B)
- **Mistral** family (7B, 8x7B)
- **Gemma** family
- **Eleuther** models
- **Custom models** via `--model_path`

## Resources

- **Whitepaper**: [OML: Open, Monetizable, and Loyal AI](https://eprint.iacr.org/2024/1573)
- **Research Blog**: [Training AI to be Loyal](https://arxiv.org/html/2502.15720v1)
- **Protocol Docs**: [Sentient Foundation](https://sentient.foundation/research)
- **GitHub**: [OML-1.0-Fingerprinting](https://github.com/sentient-agi/oml-1.0-fingerprinting)

## Citation

If you use OML 1.0 in your research, please cite:

```bibtex
@misc{oml,
  author = {Zerui Cheng and Edoardo Contente and Ben Finch and Oleg Golev and 
            Jonathan Hayase and Andrew Miller and Niusha Moshrefi and 
            Anshul Nasery and Sandeep Nailwal and Sewoong Oh and 
            Himanshu Tyagi and Pramod Viswanath},
  title = {{OML}: {O}pen, {M}onetizable, and {L}oyal {AI}},
  howpublished = {Cryptology {ePrint} Archive, Paper 2024/1573},
  year = {2024},
  url = {https://eprint.iacr.org/2024/1573}
}
```
