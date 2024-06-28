import pandas as pd
from torch_geometric.transforms import NormalizeFeatures
from gcn import train_val_test
from utils import writeToFile, getDistributionFromFilename, getDistributionIdFromFilename
import os
import torch
def gcn_experiments(root_path, name, DS):
    semantics="QUAD"
    directory = "node_features_with_distribution/"
    # label_directory = "quad_labels_with_distribution/"
    label_directory = "quad_labels_with_distribution/"
    all_files = os.listdir(root_path + "raw/" + directory)
    test_loss_dict = dict()

    for file in sorted(all_files):
        nfeatures_file = directory + os.path.splitext(file)[0]
        label_file = label_directory + os.path.splitext(file)[0]
        dataset = DS(transform=NormalizeFeatures(), root=root_path, nfeatures_filename=nfeatures_file, label_filename=label_file)
        data = dataset[0]
        test_loss, relative_mse, correlation, p_value, model = train_val_test(data, 128, data.x[1].size()[0])
        distribution = getDistributionFromFilename(nfeatures_file)
        distribution_id = getDistributionIdFromFilename(nfeatures_file)
        # torch.save(model.state_dict(), root_path + "models/gcn/" + distribution + "_" + str(distribution_id) + "_model")
        if distribution not in test_loss_dict:
            test_loss_dict[distribution] = dict()
        test_loss_dict[distribution][distribution_id] = (f"{test_loss:.4f}", relative_mse, correlation, p_value)

    writeToFile(test_loss_dict, root_path+semantics, "gcn")
