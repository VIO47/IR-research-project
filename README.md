## How Strong Is Your Baseline? A Comparative Analysis of Query Expansion Pipelines in Information Retrieval

A comparative study of multi-stage information retrieval pipelines using **BM25**, **RM3**, **ANCE**, and **ColBERT**, evaluated on the **Vaswani**, **ANTIQUE**, **FiQA**, and **NFCorpus** datasets with PyTerrier.

This repository contains all code used for the empirical evaluation presented in our research paper. It benchmarks all valid combinations of retrieval pipelines with BM25, RM3, ANCE, and ColBERT of length 4. The impact of the chaining order is evaluated on both **retrieval effectiveness** and **computational efficiency**.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/IR-research-project.git
cd IR-research-project
```

2. Install the following zip-file containing an ANCE checkpoint (https://drive.google.com/file/d/1IHqi2EpU3hdoa06LWbpoAzg21oVT7e2o/view?usp=drive_link) and place it in the root directory.

3. Install the required Python packages (by running the `installs.ipynb` notebook and following the instructions inside), select the kernel pointing to the virtual environment, and run an experimentation notebook of choice.

## Environment Requirements

- CUDA-enabled GPU is recommended to speed up the experiments.
- Sufficient disk space for dataset indexing is needed.

## Running the Experiments

Four Jupyter notebooks are provided under the project directory for running experiments with the four datasets: **Vaswani**, **ANTIQUE**, **FiQA**, and **NFCorpus**. The results are stored under the respective folders named after the dataset and the length of the pipelines being evaluated. To reproduce the results, run the respective Jupyter notebook.

For example, to run the experiments that use the Vaswani dataset, run the `experiment_vaswani.ipynb` notebook. The results are saved as `.csv` files inside:

- `vaswani-fourfold/`
- `vaswani-threefold/`
- `vaswani-twofold/`

## Pipelines Evaluated

All valid combinations of retrieval pipelines with BM25, RM3, ANCE, and ColBERT of length 4:

- `BM25 >> RM3 >> ColBERT >> ANCE`
- `BM25 >> RM3 >> ANCE >> ColBERT`
- `BM25 >> ColBERT >> RM3 >> ANCE`
- `BM25 >> ColBERT >> ANCE >> RM3`
- `BM25 >> ANCE >> ColBERT >> RM3`
- `BM25 >> ANCE >> RM3 >> ColBERT`

## Metrics Used

_Effectiveness Metrics:_

- **Mean Average Precision (MAP):** Captures the overall relevance of all retrieved documents.
- **Normalized Discounted Cumulative Gain at rank 10 (nDCG@10):** Reflects the quality of the top-ranked results.
- **Mean Reciprocal Rank (MRR):** Indicates the average position of the first relevant item in the ranking.

_Efficiency Metric:_

- **Mean Response Time (MRT)**

Statistical significance is evaluated using **Bonferroni correction**.

## Authors

- Barroso Gomes Pereira Plácido Madalena
- Selin Ceydeli
- Violeta-Mara Macsim
- Konstantin-Asen Yordanov
