import numpy as np
import cv2
import matplotlib.pyplot as plt

import helpers

IMAGE_DIR_TRAINING = "traffic_light_images/training"
IMAGE_LIST = helpers.load_dataset(IMAGE_DIR_TRAINING)

def one_hot_encode(label):
    # red, yellow, green
    one_hot_encoded = [0, 0, 0]
    if label == "red":
        one_hot_encoded[0] = 1
    elif label == "yellow":
        one_hot_encoded[1] = 1
    elif label == "green":
        one_hot_encoded = 1
    else:
        assert(False)
    return one_hot_encoded

def standardize_input(image):
    std_image = np.copy(image)
    return cv2.resize(std_image, (32, 32))

def crop_image(rgb_image, alpha=10, beta=220, thres=2):
    im = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    im = cv2.Canny(im, 10, 220)
    im[im > 0] = 1
    y = np.sum(im, axis=1)
    x = np.sum(im, axis=0)
    t, b, l, r = 0, 32, 0, 32
    
    if np.any(y[:5] > thres) and np.any(y[-5:] > thres):
        t = np.argwhere(y > thres)[0][0]
        b = np.argwhere(y > thres)[-1][0]
    if np.any(x[:5] > thres) and np.any(x[-5:] > thres):
        l = np.argwhere(x > thres)[0][0]
        r = np.argwhere(x > thres)[-1][0]
        
    return cv2.resize(rgb_image[t+2:b-2, l+5:r-5], (32, 32))

def create_feature(rgb_image):
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Extract brightness value
    v = hsv_image[:,:,2]

    height, width, channel = rgb_image.shape
    avg = np.sum(v) / float(height * width)

    print(hsv_image[:,:,1])

    # Create and return a feature value and/or vector
    feature = np.maximum(0, np.sum(v - avg, axis=1))
    return feature

def predict_label(image):
    predicted_label = [0, 0, 0]
    std_image = standardize_input(image)
    crp_image = crop_image(std_image)
    feature = create_feature(crp_image)

    v_red = np.sum(feature[2:10])
    v_yellow = np.sum(feature[10:20])
    v_green = np.sum(feature[20:30])

    i = np.argmax([v_red, v_yellow, v_green])
    predicted_label[i] = 1

    return predicted_label

def main():
    # sample one traffic light image

    image_index = 0

    for i, (image, label) in enumerate(IMAGE_LIST):
        if label == "yellow":
            image_index = i
            break

    sample_image, sample_label = IMAGE_LIST[image_index]
    std_image = standardize_input(image)
    ohe_label = one_hot_encode(label)
    crp_image = crop_image(std_image)
    predicted_label = predict_label(sample_image)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.imshow(sample_image)
    ax1.set_title("original")
    ax2.imshow(std_image)
    ax2.set_title("resized")
    ax3.imshow(crp_image)
    ax3.set_title("cropped")
    plt.show()

    print(predicted_label)

if __name__ == "__main__":
    main()