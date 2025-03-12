import torchvision


# torchvision.datasets.CelebA(
#     root="data/Datasets/",
#     split="all",
#     target_type=["attr", "identity", "bbox", "landmarks"],
#     download=True,
# )


torchvision.datasets.ImageNet(
    root="data/Datasets/imagenet/",
    split="train",
    download=False,
)
