import argparse

# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

import os
#import time
from pathlib import Path

from ct_dataset import CTDataset
import ct_config
# from ct_config import debug
# from ct_masking import CTMask

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')

    parser.add_argument('--patch_size', default=16, type=int,
                        help='masking patch size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    # Dataset parameters
    parser.add_argument('--data_path', default='datasets/', type=str,
                        help='relative path to dataset')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    # parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument('--number_of_ct_patients', default=5, type=int,
                        choices=[5, 10, 25, 50, 131], help='5, 10, 25, 50, 131')

    parser.add_argument('--debug', action='store_true')

    return parser

def main(args):

    if args.debug:
        ct_config.debug = True

    job_dir = os.path.dirname(os.path.realpath(__file__))
    print(f'job dir: {job_dir}')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # Load dataset and split to train, validation, test
    path_to_dataset = f"{job_dir}/{args.data_path}/liver_dataset_{args.number_of_ct_patients}.npz"
    dataset_train = CTDataset(path_to_dataset)
    dataset_train.split_train_test(args.number_of_ct_patients)
    # if debug:
        # dataset_train.test_transform()
    dataset_train.mask_and_save()
    dataset_train.print_summary()
    dataset_train.print_XY_samples()

    print(dataset_train)

    # Masking
    # masker = CTMask()
    # mask = masker.mask(dataset_train.X_train, 0)
    # print(mask)




    # simple augmentation
    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(), ])



    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    # train_stats = train_one_epoch(
    #     model, data_loader_train,
    #     optimizer, device, epoch, loss_scaler,
    #     log_writer=log_writer,
    #     args=args
    # )


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
