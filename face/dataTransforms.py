from torchvision import transforms


def get_transforms(model_name: str):

    if model_name == "ResnetRUL":
        train_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(scale=(0.02, 0.25)),
            ]
        )

        val_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    elif model_name == "MobileRUL":
        train_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                # transforms.RandomApply([
                # transforms.RandomAdjustSharpness(sharpness_factor=1),
                #     ], p=0.3),
                # transforms.RandomApply([
                #         transforms.RandomRotation(20),
                #         transforms.RandomCrop(224, padding=32)
                #     ], p=0.3),
                # transforms.RandomApply([
                #         transforms.ColorJitter(brightness=0.05, contrast=0.05,
                #                               saturation=0.05, hue=0.05)
                #     ], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.RandomErasing(scale=(0.02, 0.125)),
            ]
        )

        val_tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return (train_tf, val_tf)
