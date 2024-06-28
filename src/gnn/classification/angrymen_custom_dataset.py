import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Dataset, download_url, Data

class AngryMenDataset(Dataset):
    def __init__(self, root="../../../data/angrymen/", split: str = "public", transform=None, pre_transform=None, pre_filter=None, nfeatures_filename=None, label_filename=None, node_feats_reshape=True):
        self.nfeatures_filename = nfeatures_filename
        self.label_filename = label_filename
        self.files_path = root + "raw/"
        self.node_feats_reshape = node_feats_reshape
        super(AngryMenDataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['angrymen_edges.txt', self.nfeatures_filename + '.txt', self.label_filename + '.txt', 'angrymen_graph_indicator.txt', 'angrymen_edge_features.txt']

    @property
    def processed_file_names(self):
        return ['angrymen_edges.pt', self.nfeatures_filename + '.pt', self.label_filename + '.pt', 'angrymen_graph_indicator.pt', 'angrymen_edge_labels.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            train_mask = torch.tensor([True, False, False, True, False, True, True, False, True, True,
                                       True, False, True, True, True, True, False, True, False, False,
                                       True, True, True, False, True, True, True, True, True, True,
                                       False, True, True, False, False, True, True, True, True, True,
                                       True, False, True, True, True, True, True, True, True, True,
                                       False, False, True, True, True, False, True, False, True, True,
                                       True, True, True, True, True, True, True, True, True, True,
                                       False, True, False, False, False, False, True, False, False, False,
                                       False, False, False])

            val_mask = torch.tensor([False, True, True, False, True, False, False, True, False, False,
                                     False, True, False, False, False, False, True, False, True, True,
                                     False, False, False, True, False, False, False, False, False, False,
                                     True, False, False, True, True, False, False, False, False, False,
                                     False, True, False, False, False, False, False, False, False, False,
                                     True, True, False, False, False, True, False, True, False, False,
                                     False, False, False, False, False, False, False, False, False, False,
                                     True, False, False, False, False, False, False, False, False, False,
                                     False, False, False])

            test_mask = torch.tensor([False, False, False, False, False, False, False, False, False, False,
                                      False, False, False, False, False, False, False, False, False, False,
                                      False, False, False, False, False, False, False, False, False, False,
                                      False, False, False, False, False, False, False, False, False, False,
                                      False, False, False, False, False, False, False, False, False, False,
                                      False, False, False, False, False, False, False, False, False, False,
                                      False, False, False, False, False, False, False, False, False, False,
                                      False, False, True, True, True, True, True, True, True, True,
                                      True, True, True])

                                      # print(len(test_mask))
            node_feats=self.getNodeFeats()
            edge_index=self.getEdgeIndex()
            edge_feats= self.getEdgeFeats()   #attack =-1, support=1, can add more features such as NLI weight in the future
            label=self.getLabel()
            graph= self.getGraphIndicator()

            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_weight=edge_feats,
                        y=label,
                        graph_indicator=graph,
                        num_classes=83,
                        train_mask=train_mask,
                        val_mask=val_mask,
                        test_mask=test_mask
                        )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def getNodeFeats(self):
        if self.node_feats_reshape:
            st = np.genfromtxt(self.files_path + self.nfeatures_filename + '.txt', delimiter=",").reshape(83, 1)
        else:
            st = np.genfromtxt(self.files_path + self.nfeatures_filename + '.txt', delimiter=',')
        return torch.as_tensor(st, dtype=torch.float32)

    def getEdgeIndex(self):
        st = np.genfromtxt(self.files_path + 'angrymen_edges.txt', delimiter=',')
        return torch.transpose(torch.as_tensor(st, dtype=torch.long),0,1)

    def getEdgeFeats(self):
        st = np.genfromtxt(self.files_path + 'angrymen_edge_features.txt', delimiter=',').reshape(80,1)
        return torch.as_tensor(st, dtype=torch.float32)

    def getLabel(self):
        st = np.genfromtxt(self.files_path + self.label_filename + '.txt', delimiter=',')
        return torch.as_tensor(st, dtype=torch.float32)

    def getGraphIndicator(self):
        st = np.genfromtxt(self.files_path + 'angrymen_graph_indicator.txt', delimiter=',')
        return torch.as_tensor(st, dtype=torch.long)

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

# path = "../../../data/debatepedia/"
# nfeatures_file = "/node_features_with_distribution/Debatepedia_node_features_beta.1"

# DebatepediaDataset(root=path, nfeatures_filename=nfeatures_file)
# path=str("../../../data/angrymen/")
# AngryMenDataset(root=path)