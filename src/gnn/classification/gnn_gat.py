from early_stopping import EarlyStopper
import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from utils import kendal_tau

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, vsize):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(vsize,hidden_channels)  # TODO
        self.conv2 = GATConv(hidden_channels, 1)  # TODO

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_weight)
        return x

def train(model, optimizer, data, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].reshape(
        out[data.train_mask].size()))
    loss.backward()
    optimizer.step()
    return loss

def val(model, data, criterion):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return loss

def test(model, data, criterion):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_weight)
    loss = criterion(out[data.test_mask], data.y[data.test_mask].reshape(out[data.test_mask].size()))
    sum_of_node_weights = torch.sum(data.y[data.test_mask].reshape(out[data.test_mask].size()))
    return loss, sum_of_node_weights, out

def train_val_test(data, hidden_channels, vsize):
    model = GAT(hidden_channels=hidden_channels, vsize=vsize)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    early_stopper = EarlyStopper(patience=0, min_delta=0)
    epoch = 201

    for epoch in range(1, epoch):
        train_loss = train(model, optimizer, data, criterion)
        val_loss = val(model, data, criterion)
        if early_stopper.early_stop(val_loss):
            break

    test_loss, sum_of_weights, out = test(model, data, criterion)
    relative_mse = f"{(test_loss / sum_of_weights) * 100:.4f}"
    correlation, p_value = kendal_tau(data.y[data.test_mask], out[data.test_mask].detach().numpy())
    print(f"test loss: {test_loss:.4f}, relative mse: {relative_mse}, correlation: {correlation}, p_value = {p_value}")

    return test_loss, relative_mse, correlation, p_value, model

