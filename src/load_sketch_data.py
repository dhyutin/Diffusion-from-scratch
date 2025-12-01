from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SketchDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        img = self.ds[idx]['image']
        txt = self.ds[idx]['text']
        img = self.transform(img)
        return img, txt




def data_loaders(BATCH_SIZE):
    # Load Dataset

    from datasets import load_dataset

    ds = load_dataset("zoheb/sketch-scene")

    split_ds = ds['train'].train_test_split(test_size = 0.1, seed=42)

    train_ds = split_ds['train']
    test_ds = split_ds['test']

    IMG_SIZE = 64

    sample = ds['train'][55]
    img = sample["image"]

    # define transform
    transforms_img = [transforms.Resize((64, 64)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]) if img.mode == 'L'
                        else transforms.Normalize([0.5]*3, [0.5]*3)
                ]

    transform = transforms.Compose(transforms_img)

    train_dataset = SketchDataset(train_ds, transform)
    test_dataset = SketchDataset(test_ds, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    return train_loader, test_loader

