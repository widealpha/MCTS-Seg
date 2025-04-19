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
    parser.add_argument('-mod', type=str, default='sam_adpt',
                        help='mod type:seg,cls,val_ad')
    parser.add_argument('-mid_dim', type=int, default=None,
                        help='middle dim of adapter or the rank of lora matrix')
    parser.add_argument('-multimask_output', type=int, default=1,
                        help='the number of masks output for multi-class segmentation, set 2 for REFUGE dataset.')
    parser.add_argument('-thd', type=bool, default=False , help='3d or not')
    return parser.parse_args()
