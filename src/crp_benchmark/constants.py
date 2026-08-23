from __future__ import annotations

CANONICAL_CLASSES = ("REAL", "PIXEL", "PATCH", "FAKE")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CANONICAL_CLASSES)}
INDEX_TO_CLASS = {index: name for name, index in CLASS_TO_INDEX.items()}

FAKE_SUBTYPES = ("FAKE-inpaint", "FAKE-StyleGAN2", "FAKE-GeoSynth")

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PRIMARY_SEED = 42

MAIN_TRAIN_PER_CLASS = 21_680
MAIN_VAL_PER_CLASS = 2_710
MAIN_TEST_PER_CLASS = 2_710

FSC_ZERO_ADAPT_PER_CLASS = 2_710
FSC_VAL_PER_CLASS = 542
FSC_TEST_PER_CLASS = 542
FSC_SHOTS_PER_CLASS = (50, 200, 1000)
