from cfg.config import get_config
from utils.tools import train

if __name__ == "__main__":
    config = get_config('../cfg/pretrain_LeNet.yaml')
    train(config)
    config = get_config(r"../cfg/pretrain_dcnn_biloss.yaml")
    train(config)
    config = get_config(r"../cfg/dcnn_triplet.yaml")
    train(config)
    config = get_config(r"../cfg/dcnn_contrastive.yaml")
    train(config)
    config = get_config(r"../cfg/dcnn_biloss.yaml")
    train(config)

