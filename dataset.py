import torch
from torch.utils.data import Dataset
from sklearn import datasets
from torch.utils.data import random_split
import warnings

warnings.filterwarnings("ignore")

# i think this whole file can be done using torchvision
# templates but it's already too late.

# converting the sklearn data to torch dataset
digits = datasets.load_digits()

labels = digits["target"]
image_arr = digits["data"]

class DigitsDataSet(Dataset):
    def __init__(self, labels, image_data, transform=None):
        self.labels = labels
        self.image_data = image_data
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.image_data[idx]
        label = self.labels[idx]

        sample = (image, label)
        if self.transform:
            sample = self.transform(sample)

        return sample
# ToTensor sample was useless, ig torch does it automatically to ndarrays
# maybe could convert to float32 using a transform here

transformed_dataset = DigitsDataSet(labels, image_arr)

if __name__ == "__main__":
    # digit_dataset = DigitsDataSet(labels, image_arr)
    # i = 0
    # for sample in digit_dataset:
    #     print(sample["label"], sample["image"])
    #     i +=1
    #     if i == 10: break
    # train_dataloader = DataLoader(transformed_dataset, batch_size=64)
    # for x, y in enumerate(train_dataloader):
    #     # print(batch)
    #     print(x)
    #     print(y)
    #     print("\n")
    print(len(transformed_dataset))
    dataset = random_split(transformed_dataset, (800,800,197))
    print(dataset)