# Social Network Link Prediction Using Graph Neural Networks

## Introduction
This project demonstrates the application of Graph Neural Networks (GNNs) for link prediction in a social network. Using the Facebook Social Circles dataset, the model predicts potential connections between users based on their local graph structure and features.

### Objectives
- Understand GNN, especially how to work with graph convolution and message-passing mechanisms in GNNs.
- Incorporate edge features and evaluate their role in improving predictions.

---

## Data

### Dataset Overview
- **Source**: Facebook Social Circles dataset.
- **Contents**:
  - **Nodes**: Represent individual users in the social network.
  - **Edges**: Represent friendships or connections between users.

### Statistics
- **Number of Nodes**: 4,039
- **Number of Edges**: 88,234
- **Graph Type**: Undirected.

### Additional Information
- The dataset is a simple edge list format without rich node or edge attributes.
- Features were engineered to enhance the dataset.
---

## Methodology

### Dataset Preprocessing
- **Mapping Nodes to Indices**: Each user ID was mapped to a unique index for graph representation.
- **Graph Representation**: The graph was constructed using the edge list provided in the dataset.
- **Feature Engineering**:
  - Node features included degree, which captures the number of connections for each node.
  - Edge features were calculated as the number of shared neighbors between two nodes.

### Model Architecture
- **Enhanced GNN Model**:
  - A 3-layer Graph Convolutional Network (GCN) was implemented with the following components:
    - **GCN Layers**: Extract high-level representations from the graph.
    - **Batch Normalization**: Normalizes outputs of GCN layers to stabilize training.
    - **Dropout (0.3)**: Prevents overfitting by randomly deactivating neurons during training.
    - **Final Fully Connected Layer**: Produces logits for link prediction.
  - **Activation Function**: ReLU was applied after each GCN layer.

### Training Details
- **Loss Function**: Binary Cross-Entropy Loss with logits was used for training.
- **Optimizer**: AdamW with a learning rate of 0.005 and weight decay for regularization.
- **Training Epochs**: The model was trained for 50 epochs.
- **Evaluation**:
  - AUC (Area Under the Curve) and AP (Average Precision) metrics were used to measure performance.
  - An inference example was included to demonstrate specific predictions.

---

## Results

### Evaluation Metrics
- **AUC**: 0.6689
- **AP**: 0.7389

### Inference Example
- **Edge**: [3017, 3869]
- **Predicted Value**: 0.4387
- **Actual Value**: 1.0000
- **Absolute Error**: 0.5613

---

## Conclusion
- Demonstrated the application of an enhanced GNN for link prediction.
- Achieved moderate predictions and performance metrics
