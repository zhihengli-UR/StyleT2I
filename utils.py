import torchvision.transforms.functional as tvf


def differentiable_clip_preprocess_from_stylegan(img, clip_visual_size):
    img = tvf.resize(
        img, size=clip_visual_size, interpolation=tvf.InterpolationMode.BICUBIC
    )
    img = (img + 1) / 2  # -1~1 to 0~1
    img = tvf.normalize(
        img, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    return img
