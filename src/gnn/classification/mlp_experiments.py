from angrymen_custom_dataset import AngryMenDataset
from debatepedia_custom_dataset import DebatepediaDataset
from torch_geometric.transforms import NormalizeFeatures
from mlp import train_val_test
from utils import writeToFile, getDistributionFromFilename, getDistributionIdFromFilename
from mlp import updateDataFeatures
import os
import torch
import pandas as pd

def mlp_experiments(root_path, name, DS):
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
        updateDataFeatures(data)
        test_loss, relative_mse, correlation, p_value, model = train_val_test(data, 128, data.x[1].size()[0])
        distribution = getDistributionFromFilename(nfeatures_file)
        distribution_id = getDistributionIdFromFilename(nfeatures_file)
        if distribution not in test_loss_dict:
            test_loss_dict[distribution] = dict()
        test_loss_dict[distribution][distribution_id] = (f"{test_loss:.4f}", relative_mse, correlation, p_value)

    writeToFile(test_loss_dict, root_path+semantics, "mlp")
