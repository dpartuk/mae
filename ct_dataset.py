import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from ct_visualizer import CTVisualizer

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from PIL import Image

class CTDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path # Ignore and use dataset folder under job_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.patient_ids = None
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        self.X_test = None
        self.Y_test = None

        # number_of_ct_patients = 5
        # self.images, self.labels, self.patient_ids = self.load_dataset(f"datasets/liver_dataset_{number_of_ct_patients}.npz")
        self.images, self.labels, self.patient_ids = self.load_dataset(self.path)

        print(self.patient_ids)
        print('Len (X, Y, Patients):', len(self.images), len(self.labels), len(self.patient_ids))
        print(f'Sample Patient Shapes ({self.patient_ids[2]}): X[2] Y[2]:', self.images[2].shape, self.labels[2].shape)
        self.print_samples()

        # Load all images and labels
        # for root, dirs, files in os.walk(path):
        #     for file in files:
        #         if file.endswith('.nii.gz'):
        #             self.images.append(os.path.join(root, file))
        #             # Assuming the label is stored in a separate file with the same name but different extension
        #             # You may need to modify this line based on your actual label storage
        #             self.labels.append(int(file.split('_')[0]))

    def load_dataset(self, name):

        data = np.load(name, allow_pickle=True)

        X_all = data["X_all"]

        Y_all = data["Y_all"]

        patient_ids = data["patient_ids"]

        return X_all, Y_all, patient_ids

    def print_samples(self):
        ctVisualizer = CTVisualizer()

        # Show Samples
        number_of_samples = 2
        number_of_slices = 4
        for i in range(number_of_samples):
            print(f"Sample {i}: liver_{i}.nii.gz")
            ctVisualizer.display_XY_samples_v2(self.images[i], self.labels[i], max_slices=number_of_slices)

    def split_train_test(self, number_of_ct_patients):
        # Split patients into train, validation, test (by index)
        # Create list of slices
        train_idx, test_idx = train_test_split(range(number_of_ct_patients), test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

        # Combine slices from selected patients
        self.X_train = np.concatenate([self.images[i] for i in train_idx])
        self.Y_train = np.concatenate([self.labels[i] for i in train_idx])

        self.X_val = np.concatenate([self.images[i] for i in val_idx])
        self.Y_val = np.concatenate([self.labels[i] for i in val_idx])

        self.X_test = np.concatenate([self.images[i] for i in test_idx])
        self.Y_test = np.concatenate([self.labels[i] for i in test_idx])

        print("Splits by patients:")

        print(len(train_idx), len(val_idx), len(test_idx))

        print("Train Shape: " ,self.X_train.shape, "Validation Shape: ", self.X_val.shape, "Test Shape: ", self.X_test.shape)

        # Test image
        # image_0 = X_train[[0], :, :, :]
        # ct = X[idx, ..., 0]

        # print("Image 0: ", image_0.shape)
        # plt.imshow(image_0.squeeze(), cmap='gray')
        # plt.show()

    def test_transform(self):
        index = 0
        # (256, 256)
        image = self.X_train[index, ..., 0]

        # For debug purposes, print current image
        plt.imshow(image, cmap='gray')
        # plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

        # Convert the ndarray to a PIL Image
        img = Image.fromarray(image, mode='L')
        # Now you can use img as a PIL Image
        img.show()

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, index):
        # (256, 256)
        global image
        image_slice = self.X_train[index, ..., 0]

        # During pre-train, the label is the original image
        # label = self.Y_train[index, ..., 0]
        label_slice = self.X_train[index, ..., 0]

        # (256, 256, 1)
        # image = self.X_train[index]
        # label = self.Y_train[index]

        # For debug purposes, print current image
        plt.imshow(image_slice, cmap='gray')
        # plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

        # Convert the ndarray to a PIL Image
        img = Image.fromarray(image_slice, mode='L')
        label = Image.fromarray(label_slice, mode='L')

        # For debug, print image after conversion to image
        # For debug purposes, print current image
        plt.imshow(img, cmap='gray')
        # plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

        # Apply any necessary preprocessing (e.g., normalization)
        if self.transform:
            image = self.transform(img)

        test_image = image.numpy().squeeze()
        # For debug purposes, print current image
        plt.imshow(test_image, cmap='gray')
        # plt.imshow(image.squeeze(), cmap='gray')
        plt.show()

        # Convert the image to a PyTorch tensor
        # image = torch.from_numpy(image).unsqueeze(0)  # Add a channel dimension

        return image, label

# Example usage:
# transform = lambda x: (x - np.mean(x)) / np.std(x)  # Simple normalization transform
# dataset = CTDataSet('path/to/ct_data/train', transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)