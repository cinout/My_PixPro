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
processing_batch = 128  # batch size for obtaining embeddings from patches

resized_image_size = 512
patch_size = 224  # (288) keep consistent with pre-training
train_patch_stride = 6  # 6:(49*49=2401) | 32:(10*10=100) | 96:(4*4=16)
test_patch_stride = 6  # 49 * 49 = 2401
train_batch_size = 10

# resized_image_size = 256
# patch_size = 32  # (224) keep consistent with pre-training
# train_patch_stride = 4  # 4:(57*57=3249)
# test_patch_stride = 4
# train_batch_size = 64

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

        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )

        for info_batched in train_dataloader:
            train_image_batch = info_batched["image"].to(device)  # shape: bs*3*x*y
            patches_raw = train_image_batch.unfold(
                2, patch_size, train_patch_stride
            ).unfold(
                3, patch_size, train_patch_stride
            )  # shape: [bs, 3, 49, 49, 224, 224], assume get # 49*49=100 crops

            bs, _, num_crop_row, num_crop_col, _, _ = patches_raw.shape

            all_patches = [
                patches_raw[i, :, j, k, :, :]
                for i in range(bs)
                for j in range(num_crop_row)
                for k in range(num_crop_col)
            ]
            break  # only use the first 10 shuffled images

        num_iter = int(np.ceil(len(all_patches) / processing_batch))
        embeds = []
        for i in range(num_iter):
            train_patches = torch.stack(
                all_patches[i * processing_batch : (i + 1) * processing_batch]
            )
            embeds.append(encoder(train_patches.to(device)).mean(dim=(-2, -1)))
        train_embeddings = torch.cat(embeds)
        train_embeddings = torch.nn.functional.normalize(train_embeddings, p=2, dim=1)

        # fit GDE
        gde_estimator = GaussianDensityTorch()
        gde_estimator.fit(train_embeddings)

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
            is_normal = (
                info_batched["has_anomaly"].detach().numpy()[0]
            )  # is_normal: 1.0 or 0.0
            true_mask = info_batched["mask"]  # shape: bs*1*x*y

            test_image = info_batched["image"].to(device)[0]  # shape: 3*x*y
            patches_raw = test_image.unfold(1, patch_size, test_patch_stride).unfold(
                2, patch_size, test_patch_stride
            )  # shape: [3, 49, 49, 224, 224], assume get # 49*49=100 crops

            _, num_crop_row, num_crop_col, _, _ = patches_raw.shape

            all_patches = [
                patches_raw[:, i, j, :, :]
                for i in range(num_crop_row)
                for j in range(num_crop_col)
            ]

            num_iter = int(np.ceil(len(all_patches) / processing_batch))
            embeds = []
            for i in range(num_iter):
                test_patches = torch.stack(
                    all_patches[i * processing_batch : (i + 1) * processing_batch]
                )
                embeds.append(encoder(test_patches.to(device)).mean(dim=(-2, -1)))
            test_embeddings = torch.cat(embeds)
            test_embeddings = torch.nn.functional.normalize(test_embeddings, p=2, dim=1)
            distances = gde_estimator.predict(test_embeddings)
            print(distances.shape)

            # test_feature_batch = encoder(test_image_batch).mean(dim=(-2, -1))
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
