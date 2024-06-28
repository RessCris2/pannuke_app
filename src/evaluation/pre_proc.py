import albumentations as A
import torchvision as t


def aug_fn():
    trans = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.OneOf(
                [
                    A.GaussNoise(),  # 将高斯噪声应用于输入图像。
                ],
                p=0.2,
            ),  # 应用选定变换的概率
            A.OneOf(
                [
                    A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
                    A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
                    A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
                ],
                p=0.2,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2
            ),
            # 随机应用仿射变换：平移，缩放和旋转输入
            # A.RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
                always_apply=False,
                p=0.5,
            ),
        ]
    )
    return trans


def process_fn():
    process = t.Compose(
        [
            t.ToTensor(),
            t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return process
