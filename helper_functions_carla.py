'''
Helper functions for loading and processing CARLA Images
'''

import numpy as np
from glob import glob
import scipy.misc
import os.path
import imageio
import random
import cv2


def apply_color(label):
    """
    Given a 2D label of argmax values, returns a 3D color image
    Assumes 10 classes in prediction:
        0   Background
        1   Vehicles
        2   Roads
        3   Fences
        4   Pedestrians
        5   Poles
        6   Sidewalks
        7   Vegetation
        8   Buildings and Walls
        9   Other
        
    :param label: 2D prediction label of argmax values [0-9]
    :return: 3D color image
    """
    
    colors = [(70, 130, 180), # Background
              (0, 0, 142), # Vehicles
              (128, 64, 128), # Roads
              (190, 153, 153), # Fences
              (220, 20, 60), # Pedestrians
              (255, 145, 20), # Poles
              (244, 35, 232), # Sidewalks
              (107, 142, 35), # Vegetation
              (70, 70, 70), # Buildings and Walls
              (220, 220, 0)] # Other
    
    image = np.zeros((*label.shape, 3), dtype=np.uint8)
    for i in range(len(colors)):
        image[label == i] = colors[i]
        
    return image


def image_preprocessing(image, hist=True, denoise=False):
    """
    Performs the following pre-processing techniques:
    1) Histogram equalization
    2) Denoising
    
    :param image: original image
    :param means: list of color channel means (R, G, B)
    :return: mean subtracted image
    """
    
    if hist:
        image = np.array(image, dtype=np.uint8)
        image[:,:,0] = cv2.equalizeHist(image[:,:,0])
        image[:,:,1] = cv2.equalizeHist(image[:,:,1])
        image[:,:,2] = cv2.equalizeHist(image[:,:,2])
        
    if denoise:   
        image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    
    return image


def pad_image(image, new_size=(640, 800)):
    """
    Pads (bottom and right padding) image with zeros based on new size
    
    :param image: image to pad
    :param new_size: 2-element tuple, at least one dimention must be greater than original image
    :return: padded image (bottom and/or right padding)
    """
    
    if len(image.shape) != 3:
        raise ValueError('image must be of shape (h,w,3)')
    if len(new_size) != 2:
        raise ValueError('new_size must be of shape (h,w)')
    org_size = image.shape[:-1]
    if not (org_size[0] < new_size[0] or org_size[1] < new_size[1]):
        raise ValueError('new_size must be larger than current size')

    new_image = np.zeros((*new_size, 3), dtype=np.uint8)
    new_image[:org_size[0],:org_size[1],:] = image
    return new_image


def trim_image(image, new_size=(512, 800)):
    """
    Trims image by removing pixels from bottom and/or right
    
    :param image: original image of shape (h,w,3)
    :param new_size: 2-element tuple, new size to trim too
    :return: trimmed image
    """
    
    if len(image.shape) != 3:
        raise ValueError('image must be of shape (h,w,3)')
    if len(new_size) != 2:
        raise ValueError('new_size must be of shape (h,w)')
    org_size = image.shape[:-1]
    if not (org_size[0] > new_size[0] or org_size[1] > new_size[1]):
        raise ValueError('new_size must be smaller than current size')
    
    return image[:new_size[0], :new_size[1],:]


def relabel_car_hood(label, new_val=0, height_limit=495):
    """
    Re-label the driving car's hood (y >= height_limit) from 10 to new_val. Re-labeling done in-place.  
    
    :param label: 2D array of shape (600, 800), containing segmentation labels
    :param new_val: new value to assign for car's hood
    :param height_limit: y-limit where car's hood starts
    """
    
    if not isinstance(label, np.ndarray):
        label = np.array(label)
    if len(label.shape) != 2:
        raise ValueError('input is not a 2D array')
    if label.shape[0] < height_limit-1:
        raise ValueError("height_limit must be less than label's height")
        
    np.place(label[height_limit:,:], label[height_limit:,:]==10, new_val)
    

def relabel_road_lines(label):
    """
    Re-label RoadLines(6) to Roads(7)
    
    :param label: 2D array containing segementation labels
    """
    
    if not isinstance(label, np.ndarray):
        label = np.array(label, dtype=np.uint8)
    
    np.place(label, label==6, 7)


def new_label_20(label):
    """
    Consolodate certian labels with ID = 20:
    Buildings (1), Walls (11)
    
    :param label: 2D array containing segementation labels
    """
    
    if not isinstance(label, np.ndarray):
        label = np.array(label, dtype=np.uint8)
    
    np.place(label, label==1, 20)
    np.place(label, label==11, 20)
    
    
def new_label_30(label):
    """
    Consolodate certian labels with ID = 30:
    Other (3), TrafficSigns (12)
    
    :param label: 2D array containing segementation labels
    """
    
    if not isinstance(label, np.ndarray):
        label = np.array(label, dtype=np.uint8)
    
    np.place(label, label==3, 30)
    np.place(label, label==12, 30)


def one_hot_label(label, values=list(range(1,13))):
    """
    One-Hot encodes 2D label array based on the following values:
    0   Background
    1   Buildings
    2   Fences
    3   Other
    4   Pedestrians
    5   Poles
    6   RoadLines
    7   Roads
    8   Sidewalks
    9   Vegetation
    10  Vehicles
    11  Walls
    12  TrafficSigns
    
    :param label: 2D label array
    :param values: 1D array of values to encode, excluding 0 (Background)
    :return: one-hot encoded np.array with channels = # of values + 1, first channel is background followed by order of values array
    """
        
    val = list(values)
    one_hot = np.where(label == val.pop(0), 1, 0)
    one_hot = one_hot.reshape(*one_hot.shape, 1)
    for i in val:
        new = np.where(label == i, 1, 0)
        new = new.reshape(*new.shape, 1)
        one_hot = np.concatenate((one_hot, new), axis=2)
    background = np.all(one_hot == 0, axis=2)
    background = background.reshape(*background.shape, 1)
    one_hot = np.concatenate((background, one_hot), axis=2)
    return one_hot


def train_batch_gen (data_dir, values=list(range(1,13)), shuffle=True, relabel=True, trim=True,
                     trim_ind=(121, 505), reshape=True, new_shape = (416, 224), preprocess=False, 
                     new_labels=False, denoise=False):
    """
    Generates batch function for training data
    
    :param data_dir: path to dataset
    :param values: 1D array of values to encode, excluding 0 (Background)
    :param shuffle: bool, if output should be shuffled
    :param relabel: bool, if car's hood and road lines should be relabelled
    :param trim: bool, if images and labels should be trimmed
    :param trim_ind: tuple (trim_start, trim_stop), trim indicies, trim must be true
    :param reshape: bool, if images and labels should be reshaped
    :param new_shape: tuple (width, height), new image shape, reshape must be true
    :return: get_train_batch
    """
    
    if len(values) < 1:
        raise ValueError('values array is empty')
    if 0 in values:
        raise ValueError('values array cannot contain 0, reserved for background')
    
    def get_train_batch(batch_size=12):
        """
        Generate batches of images and labels for training
        
        :param batch_size: size of batch
        :return: images, labels, names
        """
        
        image_paths = glob(os.path.join(data_dir, 'CameraRGB', '*.png'))
        if shuffle:
            random.shuffle(image_paths)
        for i in range(0, len(image_paths), batch_size):
            images = []
            labels = []
            names = []
            for path in image_paths[i:i+batch_size]:
                image_name = os.path.basename(path)
                names.append(image_name)
                label_path = os.path.join(data_dir, 'CameraSeg', image_name)
                label = imageio.imread(label_path)
                label = label[:,:,0]
                image = imageio.imread(path)
                if relabel:
                    relabel_car_hood(label)
                    relabel_road_lines(label)
                    if new_labels:
                        new_label_20(label)
                        new_label_30(label)
                if trim:
                    image = image[trim_ind[0]:trim_ind[1]]
                    label = label[trim_ind[0]:trim_ind[1]]
                if reshape:
                    image = cv2.resize(image, new_shape)
                    label = cv2.resize(label, new_shape, interpolation=cv2.INTER_NEAREST)
                if preprocess:
                    image = image_preprocessing(image, denoise=denoise)
                label = one_hot_label(label, values)
                images.append(image)
                labels.append(label)
                
            images = np.array(images, dtype=np.uint8)
            labels = np.array(labels, dtype=np.uint8)
            
            yield images, labels, names
        
    return get_train_batch


def test_batch_gen (data_dir, values=list(range(1,13)), shuffle=True, relabel=True, trim=False,
                    trim_ind=(121, 505), reshape=True, new_shape = (416, 224), preprocess=False, 
                    new_labels=False, denoise=False):
    """
    Generates batch function for test data. Labels are not trimed or reshaped
    revert_trim_reshape() converts argmax predictions back to original (label) size
    
    :param data_dir: path to dataset
    :param values: 1D array of values to encode, excluding 0 (Background)
    :param shuffle: bool, if output should be shuffled
    :param relabel: bool, if car's hood and road lines should be relabelled
    :param trim: bool, if images and labels should be trimmed
    :param trim_ind: tuple (trim_start, trim_stop), trim indicies, trim must be true
    :param reshape: bool, if images and labels should be reshaped
    :param new_shape: tuple (width, height), new image shape, reshape must be true
    :return: get_test_batch
    """

    if len(values) < 1:
        raise ValueError('values array is empty')
    if 0 in values:
        raise ValueError('values array cannot contain 0, reserved for background')
        
    def get_test_batch(batch_size=12):
        """
        Generate batches of images and labels for testing.
        
        :param batch_size: size of batch
        :return: images, labels, names
        """
        
        global original_size
        image_paths = glob(os.path.join(data_dir, 'CameraRGB', '*.png'))
        image = imageio.imread(image_paths[0])
        original_size = (image.shape[1], image.shape[0])
        
        if shuffle:
            random.shuffle(image_paths)
        for i in range(0, len(image_paths), batch_size):
            images = []
            labels = []
            names = []
            for path in image_paths[i:i+batch_size]:
                image_name = os.path.basename(path)
                names.append(image_name)
                label_path = os.path.join(data_dir, 'CameraSeg', image_name)
                label = imageio.imread(label_path)
                label = label[:,:,0]
                image = imageio.imread(path)
                if relabel:
                    relabel_car_hood(label)
                    relabel_road_lines(label)
                    if new_labels:
                        new_label_20(label)
                        new_label_30(label)
                if trim:
                    image = image[trim_ind[0]:trim_ind[1]]
                    new_label = np.zeros((original_size[1], original_size[0]), dtype=np.uint8)
                    new_label[trim_ind[0]:trim_ind[1]] = label[trim_ind[0]:trim_ind[1]]
                    label = new_label
                if reshape:
                    image = cv2.resize(image, new_shape)
                if preprocess:
                    image = image_preprocessing(image, denoise=denoise)
                label = one_hot_label(label, values)
                images.append(image)
                labels.append(label)

            images = np.array(images, dtype=np.uint8)
            labels = np.array(labels, dtype=np.uint8)
            yield images, labels, names
        
    def revert_trim_reshape (preds):
        """
        Batch generator only trims and resizes images. This function is used to revert
        predicted labels for comparison during evaluation.
        
        :param pred: batch of label prediction from network
        :return: predictions of original image size
        """
        
        if original_size == None:
            raise ValueError('original_size has not been set')
        if len(preds.shape) != 3:
            raise ValueError('preds array must be 3D argmax (batch_size, height, width)')
        if trim == False and reshape == False:
            return preds
        new_preds = np.zeros((preds.shape[0], original_size[1], original_size[0]), dtype=np.uint8)
        for i, pred in enumerate(preds):
            if reshape and trim:
                pred = cv2.resize(pred, (original_size[0], trim_ind[1]-trim_ind[0]), interpolation=cv2.INTER_NEAREST)
            elif reshape:
                pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_NEAREST)
            if trim:
                new_preds[i, trim_ind[0]:trim_ind[1]] = pred
            else:
                new_preds[i] = pred
        return new_preds
    
    return get_test_batch, revert_trim_reshape
