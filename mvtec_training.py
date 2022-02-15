from ast import parse
import os
import torch
import numpy as np
import sys
from torch.utils.data import DataLoader
from torch import optim
from contrast import resnet
from tensorboard_visualizer import TensorboardVisualizer
from mvtec_dataloader import MVTecDRAEMTrainDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

location_args = {
    "pretrained_resnet": "./pretrained_pixpro/pixpro_base_r50_100ep_md5_91059202.pth",
    "output_checkpoint": "./checkpoints/",
    "mvtec_dataset": "./datasets/mvtec/",
    "log": "./logs/",
}

# TODO: check with DA_contrastive paper, plus optimizer, refer to "run.sh" file in their repo
# --num_epoch=2048 \ --batch_size=64 \ --weight_decay=0.0003 \ --learning_rate=0.01 \ --net_type=ResNet18 \
# --input_shape=32,32,3 \ --sched_type=cos \  --sched_freq=epoch \
hyper_args = {"lr": 0.0001, "epochs": 100, "bs": 32}

##########################
# load pretrained resnet50 encoder
##########################
checkpoint = torch.load(
    location_args["pretrained_resnet"], map_location=device
)
checkpoint_obj = checkpoint["model"]
pretrained_model = {}

for k, v in checkpoint_obj.items():
    if not k.startswith("module.encoder."):
        continue
    k = k.replace("module.encoder.", "")
    pretrained_model[k] = v  # TODO: or =v.numpy()?


##########################
# mvtec dataset categories
##########################
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


def train_on_device(categories):
    if not os.path.exists(location_args["output_checkpoint"]):
        os.makedirs(location_args["output_checkpoint"])

    if not os.path.exists(location_args["log"]):
        os.makedirs(location_args["log"])

    for category in categories:
        run_name = (
            "DRAEM_test_"
            + str(hyper_args["lr"])
            + "_"
            + str(hyper_args["epochs"])
            + "_bs"
            + str(hyper_args["bs"])
            + "_"
            + category
            + "_"
        )

        visualizer = TensorboardVisualizer(
            log_dir=os.path.join(location_args["log"], run_name + "/")
        )

        encoder = resnet.__dict__["resnet50"](
            head_type="early_return").to(device)
        encoder.load_state_dict(pretrained_model)
        encoder.to(device)
        encoder.train()  # set model to training mode

        optimizer = torch.optim.Adam(
            params=encoder.parameters(), lr=hyper_args["lr"])

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            [hyper_args["epochs"] * 0.8, hyper_args["epochs"] * 0.9],
            gamma=0.2,
            last_epoch=-1,
        )

        dataset = MVTecDRAEMTrainDataset(
            location_args["mvtec_dataset"] + category + "/train/good/",
            # resize_shape=[256, 256],
            # resize_shape=[224, 224],
            resize_shape=[32, 32],
        )

        dataloader = DataLoader(
            dataset, batch_size=hyper_args["bs"], shuffle=True, num_workers=16
        )

        n_iter = 0
        for epoch in range(hyper_args["epochs"]):
            print("Epoch: " + str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):
                image_batch = sample_batched["image"].to(device)
                print("image.shape", image_batch.shape)

                outcome = encoder(image_batch)
                print(outcome[1, :, :, :])

                exit()
                # gray_batch = sample_batched["image"].cuda()
                # aug_gray_batch = sample_batched["augmented_image"].cuda()
                # anomaly_mask = sample_batched["anomaly_mask"].cuda()

                # gray_rec = model(aug_gray_batch)
                # joined_in = torch.cat((gray_rec, aug_gray_batch), dim=1)

                # out_mask = model_seg(joined_in)
                # out_mask_sm = torch.softmax(out_mask, dim=1)

                # l2_loss = loss_l2(gray_rec, gray_batch)
                # ssim_loss = loss_ssim(gray_rec, gray_batch)

                # segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                # loss = l2_loss + ssim_loss + segment_loss

                # optimizer.zero_grad()

                # loss.backward()
                # optimizer.step()

                # if args.visualize and n_iter % 200 == 0:
                #     visualizer.plot_loss(l2_loss, n_iter, loss_name="l2_loss")
                #     visualizer.plot_loss(ssim_loss, n_iter, loss_name="ssim_loss")
                #     visualizer.plot_loss(segment_loss, n_iter, loss_name="segment_loss")
                # if args.visualize and n_iter % 400 == 0:
                #     t_mask = out_mask_sm[:, 1:, :, :]
                #     visualizer.visualize_image_batch(
                #         aug_gray_batch, n_iter, image_name="batch_augmented"
                #     )
                #     visualizer.visualize_image_batch(
                #         gray_batch, n_iter, image_name="batch_recon_target"
                #     )
                #     visualizer.visualize_image_batch(
                #         gray_rec, n_iter, image_name="batch_recon_out"
                #     )
                #     visualizer.visualize_image_batch(
                #         anomaly_mask, n_iter, image_name="mask_target"
                #     )
                #     visualizer.visualize_image_batch(
                #         t_mask, n_iter, image_name="mask_out"
                #     )

                # n_iter += 1

            # scheduler.step()

            # torch.save(
            #     model.state_dict(),
            #     os.path.join(args.checkpoint_path, run_name + ".pckl"),
            # )
            # torch.save(
            #     model_seg.state_dict(),
            #     os.path.join(args.checkpoint_path, run_name + "_seg.pckl"),
            # )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", action="store", type=str, required=True)

    parsed_args = parser.parse_args()

    assert parsed_args.category in category_list + \
        ["all"], "Invalid category option"

    picked_classes = (
        [parsed_args.category] if parsed_args.category != "all" else category_list
    )

    if device=="cuda":
        with torch.cuda.device(parsed_args.gpu_id):
            train_on_device(picked_classes)
    else:
        train_on_device(picked_classes)

  