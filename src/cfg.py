import argparse


def parse_args():
    all_datasets = ["ISIC2016", "ISIC2018", "ISIC2016GREY", "brats2020",
                    "ependymoma", 'BUSI-benign', 'BUSI-malignant']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default=all_datasets[0], help='Dataset name')
    parser.add_argument('--device', type=str,
                        default='cuda', help='Device to use for training/testing')
    parser.add_argument('--image-size', type=int,
                        default=512, help='Image size for training/testing')
    return parser.parse_args()
