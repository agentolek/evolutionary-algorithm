import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
import warnings

warnings.filterwarnings("ignore")

# i think this whole file can be done using torchvision
# but it's already done so eeeee.


# converting the sklearn data to torch dataset
digits = datasets.load_digits()
# nd arrays
labels = digits["target"]
image_arr = digits["data"]


# basically copied from pytorch website
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

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    def __call__(self, sample):
        image = sample["image"]
        label = sample["label"]

        # # shifting the colour axis?
        # image = image.transpose((2, 0, 1)) # from np

        # return {'image': torch.from_numpy(image),
        #         "label": sample["label"]}
        return (image, label)


transformed_dataset = DigitsDataSet(labels, image_arr, ToTensor())

# idk if dataloader is needed but it's easy to do
# build the network later today

if __name__ == "__main__":
    # digit_dataset = DigitsDataSet(labels, image_arr)
    # i = 0
    # for sample in digit_dataset:
    #     print(sample["label"], sample["image"])
    #     i +=1
    #     if i == 10: break
    print(len())
    train_dataloader = DataLoader(transformed_dataset, batch_size=64)
    for x, y in enumerate(train_dataloader):
        # print(batch)
        print(x)
        print(y)
        print("\n")
