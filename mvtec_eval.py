import os
import torch
import numpy as np
import sys
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from contrast import resnet
from density import GaussianDensityTorch
from tensorboard_visualizer import TensorboardVisualizer
from mvtec_dataloader import MVTecDRAEMTestDataset, MVTecDRAEMTrainDataset
import timeit 



device = "cuda" if torch.cuda.is_available() else "cpu"

resized_image_size = 512
patch_size = 224  # (288) keep consistent with pre-training
train_patch_stride = 6
test_patch_stride = 6  # 49 * 49 = 2401

location_args = {
    "pretrained_model": "./output/pixpro_mvtec/",
    "mvtec_dataset": "./data/mvtec/",
    "log": "./logs/",
}

# mvtec dataset categories
category_list = [
    "capsule",
    "bottle",
    "carpet",
    "leather",
    "pill",
    "transistor",
    "tile",
    "cable",
    "zipper",
    "toothbrush",
    "metal_nut",
    "hazelnut",
    "screw",
    "grid",
    "wood",
]


def eval_on_device(categories):

    if not os.path.exists(location_args["log"]):
        os.makedirs(location_args["log"])

    for category in categories:

        checkpoint = torch.load(
            os.path.join(location_args["pretrained_model"], f"current_{category}.pth"),
            map_location=device,
        )  # checkpoint.keys(): dict_keys(['opt', 'model', 'optimizer', 'scheduler', 'epoch'])

        checkpoint_obj = checkpoint["model"]
        pretrained_model = {}

        for k, v in checkpoint_obj.items():
            if not k.startswith("encoder."):
                continue
            k = k.replace("encoder.", "")
            pretrained_model[k] = v

        encoder = resnet.__dict__["resnet18"](head_type="early_return")
        encoder.load_state_dict(pretrained_model)
        encoder.to(device)
        encoder.eval()  # set model to eval mode

        # get embeddings from training dataset
        train_dataset = MVTecDRAEMTrainDataset(
            os.path.join(location_args["mvtec_dataset"], category, "train/good/"),
            resize_shape=[resized_image_size, resized_image_size],
        )
        train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=False)

        stacked_patches = []

        for info_batched in train_dataloader:
            train_image_batch = info_batched["image"].to(device)  # shape: bs*3*x*y
            bs, _, _, _ = train_image_batch.shape

            patches_raw = train_image_batch.unfold(
                2, patch_size, train_patch_stride
            ).unfold(
                3, patch_size, train_patch_stride
            )  # shape: [bs, 3, 10, 10, 224, 224], assume get # 10*10=100 crops

            patches_reshaped = patches_raw.reshape(
                bs, 3, -1, patch_size, patch_size
            )  # shape: [bs, 3, 100, 224, 224], assume get # 10*10=100 crops

            stacked_patches.append(patches_reshaped)

        patches_all = torch.cat(
            stacked_patches
        )  # shape: (#test_samples, 3, #crops, 224, 224)

        _, _, num_patches, _, _ = patches_all.shape

        patch_images_by_location = [
            patches_all[:, :, i, :, :] for i in range(num_patches)
        ]  # length = #crops

        print(len(patch_images_by_location))
        print(patch_images_by_location[0].shape)
        start_time = timeit.default_timer()

        patch_embeddings_by_location = [
            encoder(i.to(device)).mean(dim=(-2, -1)) for i in patch_images_by_location
        ]  # length = #crops; each element shape: (#samples, 512), where 512 is #feature_maps of resnet18

        print(len(patch_embeddings_by_location))
        print(patch_embeddings_by_location[0].shape)
        elapsed = timeit.default_timer() - start_time
        print(elapsed)

        exit()
        # fit GDE
        gde_estimator = GaussianDensityTorch()
        gde_estimator.fit(train_embed)

        # get test dataset
        test_dataset = MVTecDRAEMTestDataset(
            os.path.join(location_args["mvtec_dataset"], category, "test/"),
            resize_shape=[resized_image_size, resized_image_size],
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        anomaly_image_score_gt = []  # image-level ground-truth anomaly score [0,1]
        anomaly_image_score_prediction = []  # image-level predicted anomaly score [0,1]

        for i_batch, info_batched in enumerate(test_dataloader):

            test_image_batch = info_batched["image"].to(device)  # shape: bs*3*x*y

            is_normal = (
                info_batched["has_anomaly"].detach().numpy()[0]
            )  # is_normal: 1.0 or 0.0

            true_mask = info_batched["mask"]  # shape: bs*1*x*y

            test_feature_batch = encoder(test_image_batch)

            print(test_feature_batch.shape)
            exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", action="store", type=str, required=True)

    parsed_args = parser.parse_args()

    assert parsed_args.category in category_list + ["all"], "Invalid category option"

    picked_classes = (
        [parsed_args.category] if parsed_args.category != "all" else category_list
    )

    with torch.no_grad():
        eval_on_device(picked_classes)
