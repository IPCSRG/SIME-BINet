import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import torch.nn.functional as F
import cv2


def sobel_func(file_path,b,d):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    d = cv2.imread(d, cv2.IMREAD_GRAYSCALE)
    b = b.squeeze(0).numpy()
    image = image * (1 - b) + d * b
    ### sobel
    x = cv2.Sobel(image, cv2.CV_16S, 2, 0, ksize=3)
    y = cv2.Sobel(image, cv2.CV_16S, 0, 2, ksize=3)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 1, absY, 1, 0)
    image_pil = Image.fromarray(Sobel)
    # image_pil.save('./qweqw.png')
    return image_pil

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if opt.phase == 'train':
            self.dir_A = opt.train_imgroot  # create a path '/path/to/data/trainA'
            self.dir_B = opt.train_maskroot  # create a path '/path/to/data/trainB'
            # self.dir_C = opt.train_ddpm
            self.dir_D = opt.train_maskedimage_places2

        else:
            self.dir_A = opt.test_imgroot  # create a path '/path/to/data/trainA'
            self.dir_B = opt.test_maskroot  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        # self.C_paths = sorted(make_dataset(self.dir_C, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.D_paths = sorted(make_dataset(self.dir_D, opt.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transform_A = get_transform(self.opt, grayscale=(opt.input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=True)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
            index_D = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        D_path = self.D_paths[index_D]
        # C_path = self.C_paths[index % self.A_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('L')
        # C_img = Image.open(C_path).convert('RGB')
        D_img = Image.open(D_path).convert('RGB')
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        # C = self.transform_A(C_img)
        D = self.transform_A(D_img)
        E = self.transform_B(sobel_func(A_path,B,D_path))
        return {'A': A, 'B': B, 'D': D,'E':E, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return min(self.A_size, self.B_size)
