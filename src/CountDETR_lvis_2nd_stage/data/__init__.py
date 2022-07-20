from .fscd_lvis import FSCD_LVISDataset, FSCD_LVIS_Dataset_Test


def build_dataset(args,):
    return FSCD_LVISDataset(args,)

def build_test_dataset(args, split="test"):
    if split=="val":
        return FSCD_LVIS_Dataset_Test(args,split="val")
    return FSCD_LVIS_Dataset_Test(args,)
