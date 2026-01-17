from os.path import join

from torch._C import device

from DeepNetworks import HRNet, HRNet_New, ShiftNet
from src.DataLoader import ImagesetDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import json
from utils import getImageSetDirectories, readBaselineCPSNR, collateFunction
import os
import torch
from  Evaluator import cPSNR
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def load_data(config_file_path, val_proportion=0.10, top_k=-1):
    '''
    Loads all the data for the ESA Kelvin competition (train, val, test, baseline)
    Args:
        config_file_path: str, paths of configuration file
        val_proportion: float, validation/train fraction
        top_k: int, number of low-resolution images to read. Default (top_k=-1) reads all low-res images, sorted by clearance.
    Returns:
        train_dataset: torch.Dataset
        val_dataset: torch.Dataset
        test_dataset: torch.Dataset
        baseline_cpsnrs: dict, shift cPSNR scores of the ESA baseline
    '''

    with open(config_file_path, "r") as read_file:
        config = json.load(read_file)

    data_directory = config["paths"]["prefix"]
    baseline_cpsnrs = readBaselineCPSNR(os.path.join(data_directory, "norm.csv"))

    train_set_directories = getImageSetDirectories(os.path.join(data_directory, "train"))
    test_set_directories = getImageSetDirectories(os.path.join(data_directory, "test"))

    # val_proportion = 0.10
    train_list, val_list = train_test_split(train_set_directories,
                                            test_size=val_proportion, random_state=1, shuffle=True)
    config["training"]["create_patches"] = False

    train_dataset = ImagesetDataset(imset_dir=train_list, config=config["training"], top_k=top_k)
    val_dataset = ImagesetDataset(imset_dir=val_list, config=config["training"], top_k=top_k)
    test_dataset = ImagesetDataset(imset_dir=test_set_directories, config=config["training"], top_k=top_k)
    return train_dataset, val_dataset, test_dataset, baseline_cpsnrs


def load_model(config, checkpoint_file):
    '''
    Loads a pretrained model from disk.
    Args:
        config: dict, configuration file
        checkpoint_file: str, checkpoint filename
    Returns:
        model: HRNet, a pytorch model
    '''

    #     checkpoint_dir = config["paths"]["checkpoint_dir"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HRNet.HRNet(config["network"]).to(device)
    model.load_state_dict(torch.load(checkpoint_file))
    return model

def minmax_01(x, eps=1e-8):
    x = x.float()
    mn = x.amin()
    mx = x.amax()
    return (x - mn) / (mx - mn + eps)




if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open("../config/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    fusion_model = load_model(config, join("../", config["paths"]["checkpoint_dir"], "batch_32_views_32_min_32_beta_50.0_time_2026-01-14-09-26-30-992403", "HRNet.pth"))

    convertor = HRNet_New.ConvertorNoPool(config["network"])
    convertor.load_state_dict(torch.load(join("../", config["paths"]["checkpoint_dir"],
                                           "batch_32_views_32_min_32_beta_50.0_time_2026-01-14-09-26-30-992403",
                                           "fusion_model_best.pth"),  weights_only=True))

    data_directory = join("../", config["paths"]["prefix_test"])
    test_list = getImageSetDirectories(os.path.join(data_directory, "test"))
    test_dataset = ImagesetDataset(imset_dir=test_list, config=config["training"],
                                    top_k=32, beta=50)

    test_dataloader = DataLoader(test_dataset, batch_size=1,
                                shuffle=False, num_workers=4,
                                collate_fn=collateFunction(min_L=32),
                                pin_memory=True)

    i = 0
    convertor.eval()
    fusion_model.eval()

    for lrs, alphas, hrs, hr_maps, names in test_dataloader:
        lrs = lrs.float().to(device)
        alphas = alphas.float().to(device)
        hr_maps = hr_maps.numpy()

        lrs = convertor(lrs)
        hrs = convertor(hrs)

        srs = fusion_model(lrs, alphas)[:, 0]  # fuse multi frames (B, 1, 3*W, 3*H)
        psnr = PeakSignalNoiseRatio(data_range=1)

        print(f"PSNR{i}: ", psnr(minmax_01(srs), minmax_01(hrs)))
        # compute ESA score
        i += 1
        srs = srs.detach().cpu().numpy()
