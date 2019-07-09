import cv2
import numpy as np
import random
import os

class singleimage_transforms:
    def __init__(self):
        pass
    def random_brightness(self,image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rand = np.random.uniform(0.9, 1.0)
        hsv[:, :, 2] = rand*hsv[:, :, 2]
        rand = np.random.uniform(0.9, 1.8)
        hsv[:, :, 0] =rand*hsv[:, :, 0]
        rand = np.random.uniform(0.9, 1.0)
        hsv[:, :, 1] = rand*hsv[:, :, 1]
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return image

    def zoom(self,image, rows, cols):
        zoom_pix = random.randint(5, 10)
        zoom_factor = 1 + (2 * zoom_pix) / rows
        image = cv2.resize(image, None, fx=zoom_factor,
                           fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        top_crop = (image.shape[0] - rows) // 2
        #     bottom_crop = image.shape[0] - top_crop - IMAGE_HEIGHT
        left_crop = (image.shape[1] - cols) // 2
        #     right_crop = image.shape[1] - left_crop - IMAGE_WIDTH
        image = image[top_crop: top_crop + rows,
                left_crop: left_crop + cols]
        return image

    def tensor_to_image(self,tensor):
        image = tensor.clone().detach().numpy()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
        image = image.clip(0, 1)
        return image

    def enhance_images_count(self,dirname, no_of_images):
        filename = os.listdir(dirname)
        filename = random.sample(filename, no_of_images)
        for images in filename:
                imagepath = dirname + images
                _,file_extension = os.path.splitext(imagepath)
                image = cv2.imread(imagepath)
                rows, cols, channel = image.shape
                image = np.fliplr(image)
                op1 = random.randint(0, 1)
                op2 = random.randint(0, 1)
                op3 = random.randint(0, 1)
                if op1:
                    image = self.random_brightness(image)
                if op2:
                    image = self.zoom(image, rows, cols)
                if op3:
                    image = cv2.flip(image, 1)
                newimagepath = dirname + images.split('.')[0] + '_enh'+file_extension
                try:
                    image = cv2.resize(image, (224, 224))
                    cv2.imwrite(newimagepath, image)
                except:
                    print("file {0} is not converted".format(images))

