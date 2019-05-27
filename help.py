import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.exposure import histogram
from skimage.util import random_noise
from skimage import img_as_float
from scipy.ndimage.filters import median_filter
from PIL import Image, ImageFilter, ImageChops
import cv2 as cv


def rgb_gray(pic):
    x_form = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    y_cbcr = pic.dot(x_form.T)
    y_cbcr[:, :, [1, 2]] += 128

    return y_cbcr.astype('uint8')[:, :, 0]

    gw_pic = gray_world(pic)
    imsave('gw_pic.jpg', gw_pic)
    imshow(gw_pic)
    
    sp_gray_pic = random_noise(gray_pic, mode="s&p", seed=None, clip=True)
    imshow(sp_gray_pic)
    
    imsave('sp_gray_pic.jpg', sp_gray_pic)

     
    
    


def create_hist(pic):

    plt.hist(pic.flatten(), 256, [0, 256], color='black')
    plt.xlim([0, 256])
    plt.show()


def brle(pic, pct):
    y, x = histogram(pic)
    xmin, xmax = int(np.percentile(x, pct)), int(np.percentile(x, 100 - pct))
    new_pic = np.array([np.array([
        np.clip(np.uint8((px - xmin) * (255 / (xmax - xmin))), 0, 255)
        for px in row]) for row in pic])
    return new_pic


def diff_map(pic1, pic2, name="pic1_pic2"):
    _diff_map = np.abs(img_as_float(pic1) - img_as_float(pic2))
    imsave('diff_map_' + str(name) + '.jpg', _diff_map)
    return _diff_map


def docked_map(pic1, pic2, name="pic1_pic2"):
    new_pic = pic1.copy()
    for px in range(len(new_pic) // 2, len(new_pic)):
        new_pic[px] = pic2[px]
    imsave('docked_map_' + str(name) + '.jpg', new_pic.astype('uint8'))
    return new_pic.astype('uint8')


def ext(pic):
    y, x = histogram(pic)
    xmin, xmax = int(np.percentile(x, 0)), int(np.percentile(x, 100))
    new_pic = np.array([np.array([np.uint8((px - xmin) * (255 / (xmax - xmin))) for px in row]) for row in pic])
    return new_pic


def foreach_channel(pic):
    r = ext(pic[:, :, 0])
    g = ext(pic[:, :, 1])
    b = ext(pic[:, :, 2])
    return np.dstack((r, g, b))


def gray_world(pic):
    av = np.mean(pic)
    r, g, b = np.clip((av / np.mean(pic[:, :, 0])) * pic[:, :, 0], 0, 255), np.clip(
        (av / np.mean(pic[:, :, 1])) * pic[:, :, 1], 0, 255), np.clip((av / np.mean(pic[:, :, 2])) * pic[:, :, 2], 0,
                                                                      255)
    new_pic = np.dstack((r, g, b)).astype('uint8')
    return new_pic


def convolution(pic, kernel):
    kernel_shape = kernel.shape[0]
    kernel = np.flipud(np.fliplr(kernel))

    output = np.zeros_like(pic)

    pic_padded = np.zeros((pic.shape[0] + kernel_shape, pic.shape[1] + kernel_shape))

    pic_padded[1: - kernel_shape + 1, 1: - kernel_shape + 1] = pic
    for x in range(pic.shape[1]):
        for y in range(pic.shape[0]):
            output[y, x] = (kernel * pic_padded[y: y + kernel_shape, x: x + kernel_shape]).sum()

    return output


if __name__ == "__main__":
    pic = imread('picture.jpg')
    pic2 = imread('picture2.jpg')
    pic3 = imread('picture3.jpg')
    gray_pic = rgb_gray(pic2)

    imsave("gray_pic.jpg", gray_pic)

    # gaussian_3_9x9 = cv.GaussianBlur(gray_pic, (9, 9), 10.0, 3)
    # cv.imwrite('gaussian_3_9x9.jpg', gaussian_3_9x9)
    # gaussian_6_9x9 = cv.GaussianBlur(gray_pic, (9, 9), 10.0, 6)
    # cv.imwrite('gaussian_6_9x9.jpg', gaussian_6_9x9)
    # gaussian_3_15x15 = cv.GaussianBlur(gray_pic, (15, 15), 10.0, 3)
    # cv.imwrite('gaussian_3_15x15.jpg', gaussian_3_15x15)


    # print(pic.shape)

    create_hist(pic)

    brle_pic = brle(pic, 5)
    imsave('brle_pic.jpg', brle_pic)
    

    create_hist(brle_pic)

    diff_map_brle = diff_map(pic, brle_pic, name="pic_brle-pic")

    docked_map_brle = docked_map(pic, brle_pic, name="pic_brle-pic")
    ext_pic = foreach_channel(pic)
    imsave('ext_pic.jpg', ext_pic)

    docked_map_ext = docked_map(pic, ext_pic, "pic_ext-pic")


    create_hist(ext_pic)

    diff_map_ext = diff_map(pic, ext_pic, name="pic_ext-pic")
    pic = pic2
    gray_pic = rgb_gray(pic)
    imsave('gray_pic.jpg', gray_pic)

    gw_pic = gray_world(pic3)
    imsave('gw_pic.jpg', gw_pic)

    docked_map_gw = docked_map(pic3, gw_pic, "pic_gw-pic")


    sp_gray_pic = random_noise(gray_pic, mode="s&p", seed=None, clip=True)
    imsave('sp_gray_pic.jpg', sp_gray_pic)

    # docked_map_sp = docked_map(gray_pic, sp_gray_pic, "gray-pic_sp-gray-pic")

    diff_map_sp = diff_map(gray_pic, sp_gray_pic, "gray-pic_sp-gray-pic")

    filtered_pic = median_filter(sp_gray_pic, 3)
    imsave('filtered_pic.jpg', filtered_pic)

    diff_map_filtered = diff_map(img_as_float(gray_pic), img_as_float(filtered_pic), "gray-pic_filtered-pic")

    # Усреднение
    kernel = np.array([[1 / 9, 1 / 9, 1 / 9],[1 / 9, 1 / 9, 1 / 9],[1 / 9, 1 / 9, 1 / 9]])
    # kernel = np.array([[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25], [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]])
    av_pic = convolution(gray_pic, kernel)
    imsave('av_pic.jpg', av_pic)
    # Автоусреднение
    pilgray = Image.open('gray_pic.jpg')
    autoav_pic = pilgray.filter(ImageFilter.SMOOTH)
    autoav_pic.save('autoav.jpg')
    autoav_pic = imread('autoav.jpg')

    docked_map_av = docked_map(av_pic, autoav_pic, "av-pic_autoav-pic")

    diff_map_av = diff_map(img_as_float(av_pic), img_as_float(autoav_pic), "av-pic_autoav-pic")

 
    kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    shift_1 = convolution(gray_pic, kernel)


    pilgray = Image.open('gray_pic.jpg')
    autoshift = ImageChops.offset(pilgray, 1, 0)
    autoshift.save('autoshift.jpg')
    autoshift = imread('autoshift.jpg')

    docked_map_shift = docked_map(shift_1, autoshift, "shift_autoshift")

    diff_map_shift = diff_map(img_as_float(shift_1), img_as_float(autoshift), "shift_autoshift")


    kernel = np.array([[0.07511361, 0.1238414, 0.07511361], [0.1238414, 0.20417996, 0.1238414], [0.07511361, 0.1238414, 0.07511361]])
    gauss = convolution(gray_pic, kernel)
    imsave("gauss.jpg", gauss)


    pilgray = Image.open('gray_pic.jpg')
    autogauss = pilgray.filter(ImageFilter.GaussianBlur)
    autogauss.save('autogauss.jpg')
    autogauss = imread('autogauss.jpg')

    docked_map_gauss = docked_map(gauss, autogauss, "gauss_autogauss")

    diff_map_gauss = diff_map(img_as_float(gauss), img_as_float(autogauss), "gauss_autogauss")

  
    kernel = np.array([[0, -0.04, 0], [-0.04, 2, -0.04], [0, -0.04, 0]])
    sharp = convolution(gray_pic, kernel)
    imsave('sharp.jpg', sharp)

  
    pilgray = Image.open('gray_pic.jpg')
    autosharp = pilgray.filter(ImageFilter.SHARPEN)
    autosharp.save('autosharp.jpg')
    autosharp = imread('autosharp.jpg')

    docked_map_sharp = docked_map(sharp, autosharp, "sharp_autosharp")

    diff_map_sharp = diff_map(img_as_float(sharp), img_as_float(autosharp), "sharp_autosharp")

 
    gaussian_3 = cv.GaussianBlur(gray_pic, (9, 9), 10.0)
    unsharp_pic = cv.addWeighted(gray_pic, 1.5, gaussian_3, -0.5, 0, gray_pic)
    cv.imwrite('unsharp_pic.jpg', unsharp_pic)

    unsharp_pic = imread('unsharp_pic.jpg')

    docked_map_unsharp = docked_map(unsharp_pic, autosharp, "unsharp_autosharp")

    diff_map_unsharp = diff_map(img_as_float(unsharp_pic), img_as_float(autosharp), "unsharp_autosharp")
