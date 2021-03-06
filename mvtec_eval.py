from argparse import Namespace
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from contrast import resnet
from density import GaussianDensityTorch
from mvtec_dataloader import MVTecDRAEMTestDataset, MVTecDRAEMTrainDataset
from datetime import datetime
from scipy import signal
import cv2
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"
processing_batch = 128  # batch size for obtaining embeddings from patches

# resized_image_size = 512
# patch_size = 224  # (288) keep consistent with pre-training
# train_patch_stride = 6  # 6:(49*49=2401) | 32:(10*10=100) | 96:(4*4=16)
# test_patch_stride = 6  # 49 * 49 = 2401
# train_batch_size = 10

resized_image_size = 256
patch_size = 32  # (224) keep consistent with pre-training
train_patch_stride = 4  # 4:(57*57=3249)
test_patch_stride = 4
train_batch_size = 250

location_args = {
    "pretrained_model": "./output/pixpro_mvtec/",
    "mvtec_dataset": "./data/mvtec/",
    "log": "./logs/",
    "qualitative": "./qualitative/",
}

# mvtec dataset categories
category_list = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

texture_types = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
]

object_types = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def kernel_size_to_std(k: int):
    """Returns a standard deviation value for a Gaussian kernel based on its size"""
    return np.log10(0.45 * k + 1) + 0.25 if k < 32 else 10


def gkern(k: int):
    "" "Returns a 2D Gaussian kernel array with given kernel size k and std std " ""
    std = kernel_size_to_std(k)
    if k % 2 == 0:
        # if kernel size is even, signal.gaussian returns center values sampled from gaussian at x=-1 and x=1
        # which is much less than 1.0 (depending on std). Instead, sample with kernel size k-1 and duplicate center
        # value, which is 1.0. Then divide whole signal by 2, because the duplicate results in a too high signal.
        gkern1d = signal.gaussian(k - 1, std=std).reshape(k - 1, 1)
        gkern1d = np.insert(gkern1d, (k - 1) // 2, gkern1d[(k - 1) // 2]) / 2
    else:
        gkern1d = signal.gaussian(k, std=std).reshape(k, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def receptive_upsample(
    pixels: torch.Tensor,
) -> torch.Tensor:
    """
    Implement this to upsample given tensor images based on the receptive field with a Gaussian kernel.
    Usually one can just invoke the receptive_upsample method of the last convolutional layer.
    :param pixels: tensors that are to be upsampled (n x c x h x w)
    """
    assert (
        pixels.dim() == 4 and pixels.size(1) == 1
    ), "receptive upsample works atm only for one channel"
    gaus = torch.from_numpy(gkern(patch_size)).float().to(pixels.device)

    res = torch.nn.functional.conv_transpose2d(
        pixels,
        gaus.unsqueeze(0).unsqueeze(0),
        stride=train_patch_stride,
    )

    return res.to(device)


def eval_on_device(categories, args: Namespace):
    if not os.path.exists(location_args["log"]):
        os.makedirs(location_args["log"])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(args)

    output_file = open(
        os.path.join(location_args["log"], f"eval_results_{timestamp}.txt"), "a"
    )
    output_file.write(f"NOTE: {args.note}\n\n\n")
    output_file.write(f"density estimator: {args.density}\n")
    output_file.write(f"use ImageNet Resnet-18: {args.imagenet_resnet}\n")
    output_file.write(f"resized_image_size: {resized_image_size}\n")
    output_file.write(f"patch_size: {patch_size}\n")
    output_file.write(f"train_patch_stride: {train_patch_stride}\n")
    output_file.write(f"test_patch_stride: {test_patch_stride}\n")
    output_file.write(f"train_batch_size: {train_batch_size}\n")

    print("*************************")
    output_file.write("*************************\n\n\n")

    image_level_auroc_all_categories = []
    pixel_level_auroc_all_categories = []

    image_level_auroc_texture_categories = []
    pixel_level_auroc_texture_categories = []

    image_level_auroc_object_categories = []
    pixel_level_auroc_object_categories = []

    for category in categories:
        if args.qualitative:
            image_out_path = f"./qualitative/{category}"  # image output directory
            if not os.path.exists(image_out_path):
                os.makedirs(image_out_path)

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

        if args.imagenet_resnet:
            pretrained_resnet = models.resnet18(pretrained=True)
            encoder = torch.nn.Sequential(*(list(pretrained_resnet.children())[:-1]))
        else:
            encoder = resnet.__dict__["resnet18"](head_type="early_return")
            encoder.load_state_dict(pretrained_model)

        encoder = encoder.to(device)
        encoder.eval()  # set model to eval mode

        # get embeddings from training dataset
        train_dataset = MVTecDRAEMTrainDataset(
            os.path.join(location_args["mvtec_dataset"], category, "train/good/"),
            resize_shape=[resized_image_size, resized_image_size],
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )

        train_patches_by_index_dict = {}  # key is index (i,j)

        for info_batched in train_dataloader:
            train_image_batch = info_batched["image"].to(device)  # shape: bs*3*x*y

            patches_raw = train_image_batch.unfold(
                2, patch_size, train_patch_stride
            ).unfold(
                3, patch_size, train_patch_stride
            )  # shape: [bs, 3, crop_row, crop_column, patch_size, patch_size]

            bs, _, num_crop_row, num_crop_col, _, _ = patches_raw.shape

            for i in range(num_crop_row):
                for j in range(num_crop_col):
                    train_patches_by_index_dict[f"{i},{j}"] = patches_raw[
                        :, :, i, j, :, :
                    ]

            break  # only use the first #train_batch_size shuffled images

        # num_iter = int(np.ceil(len(all_patches) / processing_batch))
        # embeds = []
        # for i in range(num_iter):
        #     train_patches = torch.stack(
        #         all_patches[i * processing_batch : (i + 1) * processing_batch]
        #     )
        #     embeds.append(encoder(train_patches.to(device)).mean(dim=(-2, -1)))

        de_counter = 0
        for key, value in train_patches_by_index_dict.items():
            if de_counter % 20 == 0:
                print(
                    f">>> now fitting density for index: {key}",
                )
            de_counter += 1

            embeds = encoder(value.to(device)).mean(
                dim=(-2, -1)
            )  # value shape: bs*3*x*y
            embeds_norm = torch.nn.functional.normalize(
                embeds, p=2, dim=1
            )  # l2-normalized, shape: bs * feature_dim
            # choose density estimator
            if args.density == "kde":
                kde_gamma = 10.0 / (
                    torch.var(embeds_norm, unbiased=False) * embeds_norm.shape[1]
                )
                train_patches_by_index_dict[key] = {
                    "kde_gamma": kde_gamma,
                    "embeddings": embeds_norm,
                }  # update value of train_patches_by_index_dict
            elif args.density == "gde":
                gde_estimator = GaussianDensityTorch()
                gde_estimator.fit(embeds_norm)
                train_patches_by_index_dict[
                    key
                ] = gde_estimator  # update value of train_patches_by_index_dict

        # get test dataset
        test_dataset = MVTecDRAEMTestDataset(
            os.path.join(location_args["mvtec_dataset"], category, "test/"),
            resize_shape=[resized_image_size, resized_image_size],
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=0
        )

        image_level_gt_list = []  # image-level ground-truth anomaly score [0,1]
        image_level_pred_list = []  # image-level predicted anomaly score
        pixel_level_gt_list = np.zeros(
            (resized_image_size * resized_image_size * len(test_dataset))
        )  # pixel-level ground-truth anomaly score (binary)
        pixel_level_pred_list = np.zeros(
            (resized_image_size * resized_image_size * len(test_dataset))
        )  # pixel-level predicted anomaly score

        for i_batch, info_batched in enumerate(test_dataloader):
            # each iteration is for one image
            if i_batch % 20 == 0:
                print(
                    f">>> item index: {i_batch}/{len(test_dataset)}",
                )

            test_image = info_batched["image"].to(device)[0]  # shape: 3*x*y
            patches_raw = test_image.unfold(1, patch_size, test_patch_stride).unfold(
                2, patch_size, test_patch_stride
            )  # shape: [3, crop_row, crop_column, patch_size, patch_size]

            _, num_crop_row, num_crop_col, _, _ = patches_raw.shape

            # raster scan order (first each cols of a row, then each row)
            all_patches = [
                (patches_raw[:, i, j, :, :], (i, j))
                for i in range(num_crop_row)
                for j in range(num_crop_col)
            ]

            num_iter = int(np.ceil(len(all_patches) / processing_batch))
            scores = None  # anomaly score for all cropped patches
            for i in range(num_iter):
                batch = all_patches[i * processing_batch : (i + 1) * processing_batch]

                test_patches = torch.stack([item[0] for item in batch])
                test_embeddings = encoder(test_patches.to(device)).mean(dim=(-2, -1))
                test_embeddings = torch.nn.functional.normalize(
                    test_embeddings, p=2, dim=1
                )  # shape: #processing_batch * feature_dim

                indices = [item[1] for item in batch]  # [(0,0), (0,1), ... , (m,n)]

                for index in range(test_embeddings.shape[0]):
                    # for each patch embedding
                    patch_row, patch_col = indices[index]  # tuple (i,j)
                    embed_norm = (test_embeddings[index, :]).unsqueeze(
                        0
                    )  # shape: 1 * feature_dim
                    if args.density == "kde":

                        train_patch_gamma_embed = train_patches_by_index_dict[
                            f"{patch_row},{patch_col}"
                        ]
                        train_patch_kde_gamma = train_patch_gamma_embed["kde_gamma"]
                        train_patch_embeddings = train_patch_gamma_embed[
                            "embeddings"
                        ]  # shape: bs * feature_dim

                        similarity_batch = torch.matmul(
                            embed_norm,
                            train_patch_embeddings.transpose(0, 1),
                        )
                        scores_batch = (
                            -torch.logsumexp(
                                2 * train_patch_kde_gamma * similarity_batch, dim=1
                            )
                            / train_patch_kde_gamma
                        )
                    elif args.density == "gde":
                        train_patch_gde_estimator = train_patches_by_index_dict[
                            f"{patch_row},{patch_col}"
                        ]
                        scores_batch = train_patch_gde_estimator.predict(
                            embed_norm, device
                        )
                    scores = (
                        scores_batch
                        if scores is None
                        else torch.cat((scores, scores_batch), dim=0)
                    )

            image_level_gt = (
                info_batched["has_anomaly"].detach().numpy()[0]
            )  # is_normal: 1.0 or 0.0

            true_mask = info_batched["mask"].detach().numpy()  # shape: bs*1*x*y

            # image-level score
            image_level_pred = torch.max(scores).cpu().detach().numpy()
            image_level_gt_list.append(image_level_gt)
            image_level_pred_list.append(image_level_pred)

            # pixel-level upsampling
            upsampled_scores = receptive_upsample(
                scores.reshape((num_crop_row, num_crop_col)).unsqueeze(0).unsqueeze(0)
            )

            # pixel-level score
            pixel_level_gt_list[
                i_batch
                * resized_image_size
                * resized_image_size : (i_batch + 1)
                * resized_image_size
                * resized_image_size
            ] = true_mask.flatten()
            pixel_level_pred_list[
                i_batch
                * resized_image_size
                * resized_image_size : (i_batch + 1)
                * resized_image_size
                * resized_image_size
            ] = (upsampled_scores.cpu().detach().numpy().flatten())

            if args.qualitative:
                # qualitative image output
                file_name = info_batched["file_name"][0]
                raw_image = info_batched["image"][0]
                heatmap_alpha = 0.5

                gt_mask = np.transpose(np.array(true_mask[0] * 255), (1, 2, 0))
                gt_img = np.transpose(np.array(raw_image * 255), (1, 2, 0))
                pre_mask = np.transpose(
                    np.uint8(
                        normalizeData(upsampled_scores[0].cpu().detach().numpy()) * 255
                    ),
                    (1, 2, 0),
                )

                heatmap = cv2.applyColorMap(pre_mask, cv2.COLORMAP_JET)
                hmap_overlay_gt_img = heatmap * heatmap_alpha + gt_img * (
                    1.0 - heatmap_alpha
                )

                cv2.imwrite(
                    f"./qualitative/{category}/{file_name}_[0]mask_gt.jpg", gt_mask
                )
                cv2.imwrite(
                    f"./qualitative/{category}/{file_name}_[1]heatmap.jpg",
                    hmap_overlay_gt_img,
                )
                cv2.imwrite(
                    f"./qualitative/{category}/{file_name}_[2]img_gt.jpg", gt_img
                )

        image_level_auroc = roc_auc_score(
            np.array(image_level_gt_list), np.array(image_level_pred_list)
        )
        pixel_level_auroc = roc_auc_score(
            pixel_level_gt_list.astype(np.uint8), pixel_level_pred_list
        )

        image_level_auroc_all_categories.append(image_level_auroc)
        pixel_level_auroc_all_categories.append(pixel_level_auroc)

        if category in texture_types:
            image_level_auroc_texture_categories.append(image_level_auroc)
            pixel_level_auroc_texture_categories.append(pixel_level_auroc)
        elif category in object_types:
            image_level_auroc_object_categories.append(image_level_auroc)
            pixel_level_auroc_object_categories.append(pixel_level_auroc)

        print(f"Image Level AUROC - {category}:", image_level_auroc)
        print(f"Pixel Level AUROC - {category}:", pixel_level_auroc)
        output_file.write(f"Image Level AUROC - {category}: {image_level_auroc}\n")
        output_file.write(f"Pixel Level AUROC - {category}: {pixel_level_auroc}\n")
        print("===========")
        output_file.write("======================\n")

    output_file.write("\n\n\n")
    image_level_auroc_all_mean = np.mean(np.array(image_level_auroc_all_categories))
    pixel_level_auroc_all_mean = np.mean(np.array(pixel_level_auroc_all_categories))
    print("Image Level AUROC - Mean (15 classes):", image_level_auroc_all_mean)
    print("Pixel Level AUROC - Mean (15 classes):", pixel_level_auroc_all_mean)
    output_file.write(
        f"Image Level AUROC - Mean (15 classes): {image_level_auroc_all_mean}\n"
    )
    output_file.write(
        f"Pixel Level AUROC - Mean (15 classes): {pixel_level_auroc_all_mean}\n"
    )

    image_level_auroc_texture_mean = np.mean(
        np.array(image_level_auroc_texture_categories)
    )
    pixel_level_auroc_texture_mean = np.mean(
        np.array(pixel_level_auroc_texture_categories)
    )
    print("Image Level AUROC - Mean (Texture):", image_level_auroc_texture_mean)
    print("Pixel Level AUROC - Mean (Texture):", pixel_level_auroc_texture_mean)
    output_file.write(
        f"Image Level AUROC - Mean (Texture): {image_level_auroc_texture_mean}\n"
    )
    output_file.write(
        f"Pixel Level AUROC - Mean (Texture): {pixel_level_auroc_texture_mean}\n"
    )

    image_level_auroc_object_mean = np.mean(
        np.array(image_level_auroc_object_categories)
    )
    pixel_level_auroc_object_mean = np.mean(
        np.array(pixel_level_auroc_object_categories)
    )
    print("Image Level AUROC - Mean (Object):", image_level_auroc_object_mean)
    print("Pixel Level AUROC - Mean (Object):", pixel_level_auroc_object_mean)
    output_file.write(
        f"Image Level AUROC - Mean (Object): {image_level_auroc_object_mean}\n"
    )
    output_file.write(
        f"Pixel Level AUROC - Mean (Object): {pixel_level_auroc_object_mean}\n"
    )

    output_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        action="store",
        type=str,
        required=False,
        default="all",
        help="MVTec category type",
    )
    parser.add_argument(
        "--density",
        action="store",
        default="gde",
        choices=["kde", "gde"],
        type=str,
        help="choice of density estimator",
    )
    parser.add_argument(
        "--note",
        action="store",
        type=str,
        required=True,
        help="any personal note for this evaluation",
    )
    parser.add_argument(
        "--imagenet_resnet",
        action="store_true",
        required=False,
        help="use imageNet pretrain resnet",
    )
    parser.add_argument(
        "--qualitative",
        action="store_true",
        required=False,
        help="print qualitative images",
    )

    parsed_args = parser.parse_args()

    assert parsed_args.category in category_list + ["all"], "Invalid category option"

    picked_classes = (
        [parsed_args.category] if parsed_args.category != "all" else category_list
    )

    with torch.no_grad():
        eval_on_device(picked_classes, parsed_args)
