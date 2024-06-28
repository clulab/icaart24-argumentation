from early_stopping import EarlyStopper
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import degree
from utils import kendal_tau

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, vsize):
        super().__init__()
        torch.manual_seed(12345)
        out_dim=1
        self.lin1 = Linear(vsize, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_dim)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x=x.squeeze(1)
        return x

def train(model, optimizer, data, criterion):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def val(model, data, criterion):
    model.eval()
    out = model(data.x)
    loss = criterion(out[data.val_mask], data.y[data.val_mask])
    return loss


def test(model, data, criterion):
    model.eval()
    out = model(data.x)
    loss = criterion(out[data.test_mask], data.y[data.test_mask].reshape(out[data.test_mask].size()))
    sum_of_node_weights= torch.sum(data.y[data.test_mask].reshape(out[data.test_mask].size()))
    return loss, sum_of_node_weights, out

def train_val_test(data, hidden_channels, vsize):
    # data = dataset[0]

    model = MLP(hidden_channels=hidden_channels, vsize=vsize)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    early_stopper = EarlyStopper(patience=0, min_delta=0)
    epoch = 201
    # train & val
    for epoch in range(1, epoch):
        train_loss = train(model, optimizer, data, criterion)
        val_loss = val(model, data, criterion)
        if early_stopper.early_stop(val_loss):
            break

    # test
    test_loss, sum_of_weights, out = test(model, data, criterion)
    relative_mse = f"{(test_loss / sum_of_weights) * 100:.4f}"
    correlation, p_value = kendal_tau(data.y[data.test_mask], out[data.test_mask].detach().numpy())
    print(f"test loss: {test_loss:.4f}, relative mse: {relative_mse}, correlation: {correlation}, p_value = {p_value}")

    return test_loss, relative_mse, correlation, p_value, model

def updateDataFeatures(data):
    feature_dict = dict()
    in_degree_dict_list = dict()
    in_degree_dict = dict()
    edges_0 = data.edge_index[0]
    edges_1 = data.edge_index[1]
    edge_wt = data.edge_weight
    for i, j in zip(edges_1, edge_wt):
        if int(i) not in feature_dict:
            feature_dict[int(i)] = int(j[0])
        else:
            feature_dict[int(i)] += int(j[0])

    for i, j in zip(edges_1, edges_0):
        if int(i) not in in_degree_dict_list:
            in_degree_dict_list[int(i)] = []
        in_degree_dict_list[int(i)].append(int(j))

    for key in in_degree_dict_list:
        in_degree_dict[key] = len(in_degree_dict_list[key])

    edge_weight_features = []
    edge_in_deg_features = []
    num_features= data.x.size()[0]
    for i in range(num_features):
        if i in feature_dict:
            edge_weight_features.append(feature_dict[i])
        else:
            edge_weight_features.append(0)
        if i in in_degree_dict:
            edge_in_deg_features.append(in_degree_dict[i])
        else:
            edge_in_deg_features.append(0)

    edge_weight_features_tensor = torch.IntTensor(edge_weight_features).reshape(num_features, 1)
    edge_in_deg_features_tensor = torch.IntTensor(edge_in_deg_features).reshape(num_features, 1)

    deg = degree(data.edge_index[0]).reshape(num_features, 1)

    data.x = torch.cat((data.x, deg), 1)
    data.x = torch.cat((data.x, edge_weight_features_tensor), 1)
    data.x = torch.cat((data.x, edge_in_deg_features_tensor), 1)