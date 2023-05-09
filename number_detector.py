import os
import numpy as np
import cv2 as cv


# Get the list of images in the folder
def get_image_list(path):
    image_list = os.listdir(path)
    return image_list


# Read image from folder 2048_data
def read_image(path):
    image = cv.imread(path)
    return image


# Get the key points and descriptors of the image
def get_keypoints_descriptors(image):
    orb = cv.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


# Draw the key points of the all the images on list and save them in a new folder
def draw_keypoints(image_list, path):
    for image in image_list:
        img = read_image(path + image)
        kp, _ = get_keypoints_descriptors(img)
        img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        cv.imwrite('2048_data_orb/' + image, img2)

# get the descriptors of all the images in the list and return the descriptors in a dictionary, which key is the image name
def get_descriptors(image_list, path):
    descriptors = {}
    for image in image_list:
        img = read_image(path + image)
        _, des = get_keypoints_descriptors(img)
        descriptors[image[:-4]] = des
    return descriptors

#match the input image's descriptors with the descriptors of all the images in the dictionary using knnMatch, find the best match and return the image name
def match_image(descriptors, input_des):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    best_match = 0
    best_image = ''
    for image in descriptors:
        matches = bf.knnMatch(descriptors[image], input_des, k=5)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        if len(good) > best_match:
            best_match = len(good)
            best_image = image[:-4]
    return best_image

def number_detector():
    """
    return: a 4*4 matrix, each element is the number on the corresponding position

    """

    results = np.zeros((4, 4))
    path1 = '2048_data/'
    image_list = get_image_list(path1)
    descriptors = get_descriptors(image_list, path1)

    path2 = 'imgCache/'
    detect_list = get_image_list(path2)
    for image in detect_list:
        img = read_image(path2 + image)
        _, des = get_keypoints_descriptors(img)
        if des is None:
            results[int(image[0])][int(image[1])] = 0
        else:
            best_image = match_image(descriptors, des)
            results[int(image[0])][int(image[1])] = int(best_image[:-4])

    return results



# if __name__ == '__main__':
#     path = '2048_data/'
#     image_list = get_image_list(path)
#     draw_keypoints(image_list, path)
