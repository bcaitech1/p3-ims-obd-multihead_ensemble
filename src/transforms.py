import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 512
# def set_image_size(size=512):
#     global IMAGE_SIZE
#     IMAGE_SIZE = size
#     print("IMAGE SIZE: ", IMAGE_SIZE)

# Normal image
trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(),
    ToTensorV2()
])

elastic_trfm = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.ElasticTransform(p=0.5, alpha=40, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.Normalize(),
        ToTensorV2()
])

grid_distort_trfm = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.GridDistortion(p=0.5),
        A.Normalize(),
        ToTensorV2()
])

random_grid_shuffle_trfm = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.RandomGridShuffle(p=0.5),
        A.Normalize(),
        ToTensorV2()
])

clahe_trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CLAHE(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

random_resize_trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.RandomSizedCrop(min_max_height=(150, 200), height=IMAGE_SIZE, width=IMAGE_SIZE, p=0.5),
    A.Normalize(),
    ToTensorV2()
])

# rotate 30 이상 넘어가면 병이 거꾸로 ?!
rotate_trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Rotate(limit=30, p=0.5),
    A.Normalize(),
    ToTensorV2()
])

cutout_trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Cutout(num_holes=4, max_h_size=20, max_w_size=20, p=0.5),
    A.Normalize(),
    ToTensorV2()
])

opt_trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
    A.Normalize(),
    ToTensorV2()
])

mix_trfm = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=1, alpha=40, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=1),
        A.CLAHE(p=1),
    ], p=0.6),
    A.RandomBrightness(p=0.5),
    # A.OneOf([
    #     A.RandomSizedCrop(min_max_height=(150, 200), height=IMAGE_SIZE, width=IMAGE_SIZE, p=1),
    #     A.Rotate(limit=30, p=1),
    # ], p=0.6),
    A.Rotate(limit=30, p=0.5),
    # A.Cutout(num_holes=4, max_h_size=20, max_w_size=20, p=0.5),
    A.Normalize(),
    ToTensorV2()
])