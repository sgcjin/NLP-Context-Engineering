
# NLP Context Engineering 

## Overview
**`Experiment/`**: This directory contains experimental components for NLP context engineering project in NLP4705, focusing on Context Engineering for Legal information retrieval systems.

## Components

### 1. Experiment Setup on Google Colab(`Experiment/main.ipynb`)
A Jupyter notebook for setting up and training the legal information retrieval compressor. 

### 2. Summary Evaluation (`Experiment/Summary_eval.ipynb`)
A detailed evaluation framework for assessing summary quality and performance. This notebook provides:
- **Evaluation Metrics**: Comprehensive metrics for summary quality assessment
- **Comparative Analysis**: Tools for comparing different summarization approaches
- **Performance Analysis**: Statistical analysis of summarization results
- **Visualization**: Charts and graphs for evaluation results

## Data Sources

### Knowledge Base Data
- **Source**: [CrimeKgAssitant](https://github.com/liuhuanyong/CrimeKgAssitant)
- **Content**: Comprehensive knowledge graph of Chinese criminal law
- **Format**: JSON structure containing detailed crime classifications, legal concepts, characteristics, and penalties
- **Coverage**: Various crime categories including national security crimes, public safety crimes, etc.

### Training Data
- **Source**: [CAIL2018 Dataset](https://github.com/china-ai-law-challenge/CAIL2018)
- **Content**: Chinese legal case documents for NLP model training
- **Structure**:
  - `data_train.json`: Training dataset
  - `data_valid.json`: Validation dataset
  - `data_test.json`: Test dataset
- **Format**: JSON objects containing case facts, relevant legal articles, accusations, and sentencing information

## Data Format
The training/validation/test datasets follow this structure:
```json
{
  "fact": "Case description in Chinese",
  "meta": {
    "relevant_articles": [264],
    "accusation": ["盗窃"],
    "punish_of_money": 1000,
    "criminals": ["Defendant Name"],
    "term_of_imprisonment": {
      "death_penalty": false,
      "imprisonment": 4,
      "life_imprisonment": false
    }
  }
}
```

## Usage
1. Start with `main.ipynb` to set up the experiment
2. Use the provided datasets for training and evaluation
3. Run `Summary_eval.ipynb` to assess performance and generate evaluation reports