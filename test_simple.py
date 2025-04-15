from torchvision import transforms
from datasets.aa2 import AA2

data_dir = "/data/Datasets/"
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
)

train_dataset = AA2(
    root=data_dir,
    split="train",
    transform=transform_train,
)
breakpoint()
