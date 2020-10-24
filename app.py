from datetime import datetime
import random
# from random import randint, seed
import matplotlib.pyplot as plt
import numpy as np
import cv2

def generate_initial_seed(cipher):
    value = 0
    for c in cipher:
        value = value + ord(c)
    value = value * value

    string = str(value)[1:4]
    size = len(str(int(string)))

    # print('size:', size)
    # print('string:', string)
    if size < 3:
        # print(string)
        string = string + (3 - size) * '1'
        # print(string)

    return int(string)

def generate_seed(initial_seed):
    value = initial_seed * initial_seed
    # print('value:', value)

    string = str(value)[1:4]
    size = len(str(int(string)))

    # print('size:', size)
    # print('string:', string)
    if size < 3:
        # print(string)
        string = string + (3 - size) * '1'
        # print(string)

    return int(string)

def open_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def save_image_array(image, description):
    img = open(description, 'w')
    img.write(str(image))
    img.close()     

def generate_grayscale_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            image[y, x] = [g, g, g]

    return image

def generate_binary_image(image, level=0.5):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    level_value = int(level * 255)

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            if g >= level_value:
                g = 255
            elif g < level_value:
                g = 0

            image[y, x] = [g, g, g]

    return image       

def generate_red_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            image[y, x] = [g, 0, 0]

    return image    

def generate_green_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            image[y, x] = [0, g, 0]

    return image

def generate_blue_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            image[y, x] = [0, 0, g]

    return image

def generate_inverted_color_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            image[y, x] = [255 - r, 255 - g, 255 - b]

    return image

def generate_shift_right_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            image[y, x] = [r >> 1, g >> 1, b >> 1]

    return image

def generate_shift_left_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            image[y, x] = [r << 1, g << 1, b << 1]

    return image

def generate_or_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            random_number = randint(0, 255)
            image[y, x] = [r | random_number, g | random_number, b | random_number]

    return image

def generate_and_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            random_number = randint(0, 255)
            image[y, x] = [r & random_number, g & random_number, b & random_number]

    return image            

def generate_xor_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    
    random.seed(30)

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            random_number = random.randint(0, 255)
            image[y, x] = [r ^ random_number, g ^ random_number, b ^ random_number]

    return image

def generate_xor_image_with_cipher(image, cipher='robinson'):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    random.seed(30)

    # initial = 0
    # for c in cipher:
    #     initial = initial + ord(c)
    # initial = initial * initial

    # initial = str(initial)[:-1]
    
    # value = generate_initial_seed(cipher)
    index = 0
    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            value = random.randint(0, 255)
            # value = generate_seed(value)
            # _value = value & 255
            image[y, x] = [r ^ value, g ^ value, b ^ value]

            if index < len(cipher) - 1:
                index = index + 1
            else:
                index = 0

    return image    

def generate_not_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            image[y, x] = [~r, ~g, ~b]

    return image

def generate_test1_image(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]

    for x in range(width):
        for y in range(height):
            r, g, b = image[y, x]
            if (r + g + b) < 10:
                r = 0
                g = 0
                b = 0                              
            image[y, x] = [r, g, b]

    return image                                              

def show_image(image):
    plt.imshow(image)
    plt.show()

def save_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./output/{}.jpg'.format(datetime.now()), image)    

    
# In fact, OpenCV by default reads images in BGR format. 
# You can use the cvtColor(image, flag) and the flag we looked at above to fix this:
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# np.savetxt("nxx", array.reshape((3,-1)), fmt="%s", header=str(array.shape))

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# print('{}'.format(image))

# image = process_image(image)
# np.savetxt('zero2.txt', image.reshape((3, -1)), fmt="%s", header=str(image.shape))


# for x in range(width):
#     for y in range(height):
#         r, g, b = image[y, x]
#         # value = (r + g + b) / 3
#         image[y, x] = [r * 0.299, g * 0.587, b * 0.114]
#         # image[y, x] = [r * 0, g * 0, b]

# https://stackoverflow.com/questions/19181323/what-grayscale-conversion-algorithm-does-opencv-cvtcolor-use
# The color to grayscale algorithm is stated in the cvtColor() documentation. (search for RGB2GRAY). The formula used is the same as for CCIR 601:

# Y = 0.299 R + 0.587 G + 0.114 B

# The luminosity formula you gave is for ITU-R Recommendation BT. 709. If you want that you can specify CV_RGB2XYZ (e.g.) in the third parameter to cvtColor() then extract the Y channel.

# You can get OpenCV to to do the "lightness" method you described by doing a CV_RGB2HLS conversion then extract the L channel. I don't think that OpenCV has a conversion for the "average" method, but if you explore the documentation you will see that there are a few other possibilities


# cv2.imshow('image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def main():
    image = open_image('input/Dogao.jpg')
    image = generate_binary_image(image)
    # save_image_array(image, 'original.txt')
    show_image(image)
    save_image(image)


if __name__ == '__main__':
    main()