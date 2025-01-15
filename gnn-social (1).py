import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load the Dataset and the edge list
data_path = "facebook_combined.txt"
edge_list = pd.read_csv(data_path, delim_whitespace=True, header=None, names=['source', 'target'])

# Preprocess the Dataset
# Map node IDs to indices
nodes = pd.concat([edge_list['source'], edge_list['target']]).unique()
num_nodes = len(nodes)
node_map = {node_id: idx for idx, node_id in enumerate(nodes)}
edge_list['source_idx'] = edge_list['source'].map(node_map)
edge_list['target_idx'] = edge_list['target'].map(node_map)

# Create edge index
torch_edge_index = torch.tensor(
    edge_list[['source_idx', 'target_idx']].values.T, dtype=torch.long
)

# Add node features
# Degree as a feature
degree = torch.tensor([torch.sum(torch_edge_index[0] == i) for i in range(num_nodes)], dtype=torch.float32).view(-1, 1)

# Combine features (only degree for simplicity)
node_features = degree

# Edge features: Shared neighbors between source and target
edge_features = torch.tensor([
    len(set(torch_edge_index[1][torch_edge_index[0] == src].numpy()) &
        set(torch_edge_index[1][torch_edge_index[0] == tgt].numpy()))
    for src, tgt in edge_list[['source_idx', 'target_idx']].values
], dtype=torch.float32).view(-1, 1)

# Create PyTorch Geometric Data Object
social_graph = Data(x=node_features, edge_index=torch_edge_index, edge_attr=edge_features)

# Define an Enhanced GNN Model
from torch.nn import functional as F

class EnhancedLinkPredictionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EnhancedLinkPredictionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # Additional GCN layer
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)  # Final prediction layer
        self.dropout = torch.nn.Dropout(0.3)  # Dropout for regularization

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        return self.fc(x)

# Initialize the model
input_dim = node_features.shape[1]
hidden_dim = 32
model = EnhancedLinkPredictionGNN(input_dim, hidden_dim)
print("\nEnhanced Link Prediction GNN Model:")
print(model)

# Train/Test Split
from torch_geometric.utils import negative_sampling

# Use 10% of edges for testing
num_test_edges = int(0.1 * torch_edge_index.size(1))
edge_index_train = torch_edge_index[:, :-num_test_edges]
edge_index_test = torch_edge_index[:, -num_test_edges:]

# Generate negative samples for testing
neg_edge_index_test = negative_sampling(
    edge_index_test, num_nodes=num_nodes, num_neg_samples=num_test_edges
)

# Training Setup
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

def train():
    model.train()
    optimizer.zero_grad()
    pos_pred = model(social_graph.x, edge_index_train, social_graph.edge_attr).mean(dim=1)
    neg_edge_index_train = negative_sampling(edge_index_train, num_nodes=num_nodes)
    neg_pred = model(social_graph.x, neg_edge_index_train, social_graph.edge_attr).mean(dim=1)
    
    # Combine positive and negative samples
    preds = torch.cat([pos_pred, neg_pred])
    labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))])
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# Training Loop
print("\nTraining the Enhanced Link Prediction GNN:")
for epoch in range(1, 51):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Evaluation
def evaluate():
    model.eval()
    with torch.no_grad():
        pos_pred = model(social_graph.x, edge_index_test, social_graph.edge_attr).mean(dim=1)
        neg_pred = model(social_graph.x, neg_edge_index_test, social_graph.edge_attr).mean(dim=1)
        preds = torch.cat([pos_pred, neg_pred]).detach().cpu().numpy()
        labels = np.concatenate([np.ones(pos_pred.size(0)), np.zeros(neg_pred.size(0))])
        
        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        print(f"\nEvaluation Metrics:\nAUC: {auc:.4f}\nAP: {ap:.4f}")

evaluate()

# Inference Example
def inference_example():
    model.eval()
    with torch.no_grad():
        # Choose a random test edge
        test_index = np.random.randint(edge_index_test.shape[1])
        edge = edge_index_test[:, test_index]

        # Get the prediction for this edge
        pred_value = model(social_graph.x, edge_index_test, social_graph.edge_attr)[test_index].item()

        # Define the actual value (1 for positive edge, 0 for negative)
        actual_value = 1 if test_index < edge_index_test.shape[1] // 2 else 0  # Assuming first half are positive

        # Calculate absolute error
        absolute_error = abs(pred_value - actual_value)

        print("\nInference Example:")
        print(f"Edge: {edge.tolist()}")
        print(f"Predicted Value: {pred_value:.4f}")
        print(f"Actual Value: {actual_value:.4f}")
        print(f"Absolute Error: {absolute_error:.4f}")

inference_example()
