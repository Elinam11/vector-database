# Multi-Vector Database (MVDB): Evaluating ANN Quality in Aggregated Multi-Vector Retrieval

This project explores efficient multi-vector retrieval using approximate nearest neighbor (ANN) search and aggregation techniques over multimodal embeddings. The system supports retrieval across multiple modalities such as text and images using FAISS-based indexing and implements both baseline aggregation and the Quick-Combine (QC) algorithm for top-k retrieval.

The work focuses on understanding how ANN search quality affects retrieval performance, particularly the tradeoff between recall and latency in aggregated multi-vector search.

## Project Overview

The implementation includes:

- Multimodal retrieval using text and image embeddings
- FAISS IVF-PQ indexing for approximate nearest neighbor search
- Aggregated similarity scoring across modalities
- Exact brute-force top-k retrieval for ground truth evaluation
- Naive ANN candidate-pool aggregation baseline
- Quick-Combine (QC) retrieval with threshold-based early termination

The system was evaluated on the MITStates dataset using:
- CLIP embeddings for images
- MPNet embeddings for text
- IVF-PQ ANN indexing with varying `nprobe` values

## Experimental Findings

Experiments were conducted to evaluate the effect of ANN quality on Quick-Combine retrieval.

Key observations include:

- At low ANN quality (low `nprobe` values), Quick-Combine recall degrades significantly because the algorithm depends heavily on the quality of the top-ranked ANN candidates.
- As `nprobe` increases, recall improves rapidly for both QC and the naive ANN baseline.
- A clear “knee point” was observed around `nprobe ≈ 10`, where both methods achieved near-perfect recall.
- Beyond this point, increasing `nprobe` only increased latency with little or no recall improvement.
- The epsilon (`ε`) approximation parameter primarily affected latency rather than recall, since high-quality ANN results caused the exact threshold condition to terminate early before approximation logic became significant.

These experiments highlight the sensitivity of aggregated retrieval methods to ANN quality and demonstrate the practical tradeoff between retrieval accuracy and query latency.

## Docker

1. To build the Docker image, run the following command in the terminal:

    ```bash
    docker build -t mvdb:latest .
    ```

2. To run the Docker container, use the following command:

    ```bash
    docker run -v /path/to/datasets:/data -v /path/to/indices:/indices mvdb:latest
    ```

Make sure to replace `/path/to/datasets` and `/path/to/indices` with the actual paths on your host machine where you want to store the datasets and indices.

## Dataset

1. Download the dataset from:
https://opendatalab.com/OpenDataLab/MIT-States/tree/main/raw

2. After downloading, mount the dataset directory to `/data` when running the container.

