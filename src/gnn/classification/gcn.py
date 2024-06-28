from src.gnn.classification.early_stopping import EarlyStopper
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from src.gnn.classification.utils import kendal_tau
import copy
import matplotlib.pyplot as plt


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, vsize):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.out_dim = 1
        self.conv1 = GCNConv(vsize, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, self.out_dim)
        # Adding a residual connection requires dimensions to match
        self.residual = torch.nn.Linear(vsize, hidden_channels, bias=False)
        self.dropout_rate_1= 0.5
        self.dropout_rate_2 = 0.3

    def forward(self, x, edge_index, edge_weight):
        # Initial input (for the residual connection)
        res = self.residual(x)

        # First GCN layer
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn1(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate_1, training=self.training)

        # Second GCN layer with residual connection
        x = self.conv2(x, edge_index, edge_weight=edge_weight) + res
        x = self.bn2(x)
        x = x.relu()
        x = F.dropout(x, p=self.dropout_rate_2, training=self.training)

        # Output layer
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        return x.squeeze(1)

def train(model, optimizer, data, criterion):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index, data.edge_weight)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss

def val(model, data, criterion):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight)
        val_pred = out[data.val_mask].detach()
        val_true = data.y[data.val_mask].detach()

    # Calculate loss and correlation
    loss = criterion(val_pred, val_true)

    return loss


def test(model, data, criterion):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(out[data.test_mask], data.y[data.test_mask].reshape(out[data.test_mask].size()))
    sum_of_weights=torch.sum(data.y[data.test_mask].reshape(out[data.test_mask].size()))
    return loss, sum_of_weights, out

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.optim import AdamW


def train_val_test(data, hidden_channels, vsize):
    model = GCN(hidden_channels=hidden_channels, vsize=vsize)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.9, min_lr=1e-6, verbose=True)
    criterion = torch.nn.MSELoss()
    early_stopper_loss = EarlyStopper(patience=30, min_delta=0.001)

    best_val_metric = float('inf')
    best_val_corr = float('-inf')  # Initialize the best validation correlation
    best_model = None

    loss_history = []
    correlation_history = []
    correlation_threshold_reached = False

    for epoch in range(1, 5001):
        train_loss = train(model, optimizer, data, criterion)
        val_loss, val_corr = val(model, data, criterion)
        scheduler.step(val_loss)

        loss_history.append(val_loss)
        correlation_history.append(val_corr)
        # Update the best correlation and model, if the current correlation is better
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_model = copy.deepcopy(model.state_dict())

        # Check the correlation against the threshold
        if val_corr >= 0.18:
            correlation_threshold_reached = True


        # Check early stopping conditions based on loss
        if early_stopper_loss(val_loss) and correlation_threshold_reached:
            print(f"Early stopping at epoch {epoch} due to loss with best validation metric: {best_val_metric:.4f}")
            break
        if val_corr >= 0.5:
            correlation_threshold_reached = True
        elif correlation_threshold_reached and val_corr < 0.5:
            print(f"Correlation fell below 0.5 at epoch {epoch}, stopping early.")
            break
        # Logging for visibility
        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"Epoch {epoch:03d}, Val Loss: {val_loss:.4f}, Correlation: {val_corr:.4f}")

    # If the loop exits without breaking, load the best model
    model.load_state_dict(best_model)
    test_loss, sum_of_weights, out = test(model, data, criterion)
    relative_mse = (test_loss / sum_of_weights).item() * 100
    correlation, p_value = kendal_tau(data.y[data.test_mask], out[data.test_mask].detach().numpy())

    print(f"Test loss: {test_loss:.4f}, Relative MSE: {relative_mse:.4f}, Correlation: {correlation:.2f}, p-value: {p_value:.4f}")

    return test_loss, relative_mse, correlation, p_value, model
