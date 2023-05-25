# Learning to Learn workshop at MLPrague 2023
This repo contains link to the materials for the Learning to Learn workshop on Machine Learning Prague 2023.

## Getting started

The primary playground env for the exercises below is [Google Colab](https://colab.research.google.com). 
The linked Colab notebooks contain the resolution of dependences, but if you'd like to run the exercises elsewhere, simply install the attached `requirements.txt` into any environment:

```shell
git clone https://github.com/gaussalgo/L2L_MLPrague23.git
pip install -r L2L_MLPrague23/requirements.txt
```

## Outline

### 1. Intro to Transformers

- Architectures
  - Difference to other arch's (attention layer)
  - Tasks (=objectives)

- Pre-training & Fine-tuning

- Inputs and outputs
  - Single token prediction

- Generation
  - Iterative prediction
  - Other generation strategies
  - [Hands-on] Constraining generated output (forcing & disabling)

### 2. In-context Learning and Few-shot Learning with Transformers
- Problem definition (usage)
- Contrast with Supervised ML
- Zero-shot vs few-shot
  - Examples
  - [Hands-on] comparison of zero-shot vs. few-shot performance (of some chosen ICL)

### 3. Methods for Improving ICL

#### Inference
- Demonstrations heterogeneity
- Prompt engineering
  - Promptsource - database of prompts?
  - [Hands-on] prompt engineering (inspired by the training data?)

#### Training Strategies
- Training strategies + existing models
  - training in explicit fewshot format (QA)
  - Instruction tuning
  - Multitask learning
  - Chain-of-Thought
  - Pre-training on a code
  - Fine-tuning with human feedback
#### Theory behind - why does ICL exist?
  - Data properties fostering ICL
  - Experiments
  - Explanations of the existing models?

### 4. Hands-on in Improving Few-shot ICL
- [Hands-on] Customizing Few-shot ICL to specialized data
- Practical training pipeline
  - Overview of the training pipeline
  - Adaptor example

-------

