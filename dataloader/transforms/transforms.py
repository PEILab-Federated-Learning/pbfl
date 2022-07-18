from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import (
    Normalize, RandomResizedCrop, 
    ToTensor, RandomHorizontalFlip,
    Compose, Resize, CenterCrop
)


INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}


def build_transform(cfg, is_train=True):
    """Build transformation function.
    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
    """
    if cfg.INPUT.NO_TRANSFORM:
        if cfg.VERBOSE:
            print("Note: no transform is applied!")
        return None

    target_size = f"{cfg.INPUT.SIZE[0]}x{cfg.INPUT.SIZE[1]}"
    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        return _build_transform_train(cfg, cfg.INPUT.TRANSFORMS, target_size, normalize)
    else:
        return _build_transform_test(cfg, cfg.INPUT.TRANSFORMS, target_size, normalize)


def _build_transform_train(cfg, choices, target_size, normalize):
    if cfg.VERBOSE:
        print("Building transform_train")
    tfm_train = []
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    if "random_resized_crop" in choices:
        s_ = cfg.INPUT.RRCROP_SCALE
        if cfg.VERBOSE:
            print(f"+ random resized crop (size={input_size}, scale={s_})")
        tfm_train += [
            RandomResizedCrop(input_size, scale=s_, interpolation=interp_mode)
        ]

    if "random_flip" in choices:
        if cfg.VERBOSE:
            print("+ random flip")
        tfm_train += [RandomHorizontalFlip()]
    
    if cfg.VERBOSE:
        print("+ to torch tensor of range [0, 1]")
    tfm_train += [ToTensor()]

    if "normalize" in choices:
        if cfg.VERBOSE:
            print(f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})")
        tfm_train += [normalize]

    tfm_train = Compose(tfm_train)

    return tfm_train


def _build_transform_test(cfg, choices, target_size, normalize):
    if cfg.VERBOSE:
        print("Building transform_test")
    tfm_test = []
    interp_mode = INTERPOLATION_MODES[cfg.INPUT.INTERPOLATION]
    input_size = cfg.INPUT.SIZE

    if cfg.VERBOSE:
        print(f"+ resize the smaller edge to {max(input_size)}")
    tfm_test += [Resize(max(input_size), interpolation=interp_mode)]

    if cfg.VERBOSE:
        print(f"+ {target_size} center crop")
    tfm_test += [CenterCrop(input_size)]
    
    if cfg.VERBOSE:
        print("+ to torch tensor of range [0, 1]")
    tfm_test += [ToTensor()]

    if "normalize" in choices:
        if cfg.VERBOSE:
            print(f"+ normalization (mean={cfg.INPUT.PIXEL_MEAN}, std={cfg.INPUT.PIXEL_STD})")
        tfm_test += [normalize]

    tfm_test = Compose(tfm_test)

    return tfm_test


