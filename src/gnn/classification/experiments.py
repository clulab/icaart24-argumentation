from angrymen_custom_dataset import AngryMenDataset
from debatepedia_custom_dataset import DebatepediaDataset
from gcn_experiments import gcn_experiments
from mlp_experiments import mlp_experiments
from gnn_gat_experiments import gnn_gat_experiments

# gcn_experiments("../../../data/angrymen/", "angrymen", AngryMenDataset)
# mlp_experiments("../../../data/angrymen/", "angrymen", AngryMenDataset)
# gnn_gat_experiments("../../../data/angrymen/", "angrymen", AngryMenDataset)
#
gcn_experiments("../../../data/debatepedia/", "debatepedia", DebatepediaDataset)
# mlp_experiments("../../../data/debatepedia/", "debatepedia", DebatepediaDataset)
# gnn_gat_experiments("../../../data/debatepedia/", "debatepedia", DebatepediaDataset)
#
