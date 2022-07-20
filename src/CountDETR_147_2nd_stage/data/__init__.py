from .fsc147 import FSC147Dataset, FSC147_Dataset_Val, FSC147_Dataset_Test

def build_dataset(args,):
    return FSC147Dataset(args,)

def build_test_dataset(args, image_set="val"):
    if image_set == "val":
        return FSC147_Dataset_Val(args,)
    else:
        return FSC147_Dataset_Test(args,)