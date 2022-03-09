import os

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from .transform import get_transform
from .dataset import ImageFolder


def get_loader(aug_type, args, two_crop=False, prefix="train", return_coord=False):
    # default: aug_type (aug): BYOL, crop: 0.08, image_size: 224
    transform = get_transform(aug_type, args.crop, args.image_size)

    # dataset
    if args.zip:
        # FIXME: right now it only supports imagenet
        if args.dataset == "ImageNet":
            train_ann_file = prefix + "_map.txt"
            train_prefix = prefix + ".zip@/"
        else:
            raise NotImplementedError(
                "Dataset {} is not supported. We only support ImageNet now".format(
                    args.dataset
                )
            )

        train_dataset = ImageFolder(
            args.data_dir,
            train_ann_file,
            train_prefix,
            transform,
            two_crop=two_crop,
            cache_mode=args.cache_mode,
            dataset=args.dataset,
            return_coord=return_coord,
        )
    else:
        if args.dataset == "ImageNet":
            train_folder = os.path.join(args.data_dir, "train_blurred")
            train_dataset = ImageFolder(
                train_folder,
                transform=transform,
                two_crop=two_crop,
                return_coord=return_coord,
                dataset=args.dataset,
            )
        elif args.dataset == "MVTec":
            train_folder = os.path.join(args.data_dir, args.mvtec_category, prefix)
            train_dataset = ImageFolder(
                train_folder,
                transform=transform,
                two_crop=two_crop,
                return_coord=return_coord,
                dataset=args.dataset,
                resized_image_size=args.resized_image_size,
                crop_patch_size=args.crop_patch_size,
                crop_patch_stride=args.crop_patch_stride,
            )

    # sampler
    indices = np.arange(dist.get_rank(), len(train_dataset), dist.get_world_size())
    if args.zip and args.cache_mode == "part":
        sampler = SubsetRandomSampler(indices)
    else:
        sampler = DistributedSampler(train_dataset)

    # dataloader
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,
    )
