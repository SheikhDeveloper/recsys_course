# Recommender Systems YSDA Course
## Homework №1: Candidate Generation

During practice, we've discussed common candidate generation (CG) stage abstractions, quality metrics, and implemented several models with a KNN candidate generator class. In this homework, your task is to dive deeper into this domain and implement a few more models, compare them, and develop a way to use all of them simultaneously. The homework consists of several independent subtasks and two bonus subtasks.

## Mandatory Tasks

### 1. EASE (0.5 points)
Implement the model from [Embarrassingly Shallow Autoencoders for Sparse Data](https://arxiv.org/pdf/1905.03375).

### 2. Simplified SLIM (1 point)
Implement the model from [SLIM: Sparse Linear Methods for Top-N Recommender Systems](https://ieeexplore.ieee.org/document/6137254), specifically the simplified version from this [paper](https://www.math.unipd.it/~aiolli/PAPERS/MSD_final.pdf). This version drops the constraint of diagonal weights being equal to zero and uses the scikit-learn solver for optimization.

### 3. Model Appliers for Inference (1 point)
Implement `LinearItemToItemCG` class for inference of EASE/SLIM. Also implement the `ANN` using one of the frameworks for approximate nearest neighbor search, and compare ALS with EASE.

### 4. BPR with Popularity Sampling (1.5 points)
Implement the algorithm described in [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/pdf/1205.2618), with a matrix factorization backbone. The model requires sampling negatives for pairwise loss, and your task is to implement fast popularity-based sampling.

### 5. BPR with Adaptive Model-Based Sampling (1.5 points)
Implement the same algorithm, but with improved negative sampling: use weighted sampling of positives and negatives, and recompute the sampling weights every N iterations to make them proportional to the model scores. This ensures that hard negatives with high scores will be sampled more frequently.

### 6. Report with Speed and Quality Comparison (0.5 points)
For all models, ensure you meet the expected quality metric level - predefined parameters from the initial notebook should help with this.
After implementing all models, create a benchmark comparing your model training time and inference time with `LinearItemToItemCG` or `ANN` candidate generator classes. Present your results as a table at the end of the notebook section.

## Bonus Tasks

### Mixigen (2 points)
Implement Mixigen - a personalized GBRT-based algorithm for adaptive blending of multiple candidate generators, using later-stage ranking. For this part, you'll need to:
1. Build a training pool by extracting candidates from all models for a set of requests
2. Score them with a pretrained ranker
3. Train the model to predict whether they will be in the final top-N shown candidates using only user and source CG information

### SLIM Optimizer (2 points)
Since Python libraries don't have coordinate descent implementations with arbitrary linear constraints, implementing the model as described in the paper requires writing your own optimizer. However, a pure Python implementation would be EXRTREMELY slow.

If you're feeling ambitious, you can implement your own SLIM-specific coordinate descent optimizer using a lower-level language (or `Cython`/`Numba`). This can earn you up to 2 additional points, depending on the implementation quality. Implementation hints are provided in the bonus task section.

## Submission Guidelines

- **Mandatory Tasks**: Submit a `.ipynb` notebook containing all implemented models, training logs, and a metrics/speed comparison table at the end.
- **Mixigen Bonus**: Submit in the same notebook format as the mandatory tasks - this simplifies the grading process.
- **SLIM Bonus**: Submit an archive containing:
  - Source code for the implementation
  - Instructions for building and training with the Lavka dataset
  - Results showing training speed and metrics

For any questions about the homework, contact [Roma Nigmatullin](https://github.com/rmnigm) (@rmnigmatullin) in the Telegram chat.
