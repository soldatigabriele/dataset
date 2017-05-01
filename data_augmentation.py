from imgaug import augmenters as iaa
import imgaug as ia
import os
from skimage import io
import cv2
import scipy.misc
import numpy as np
from scipy.misc import imsave
from random import randint

'''
seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
])
'''
st = lambda aug: iaa.Sometimes(0.3, aug)
'''
seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5), # vertically flip 50% of all images

        st(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
        st(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
        st(iaa.GaussianBlur((0, 3.0))), # blur images with a sigma between 0 and 3.0
        st(iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))), # sharpen images
        st(iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))), # emboss images
        # search either for all edges or for directed edges
        st(iaa.Sometimes(0.5,
            iaa.EdgeDetect(alpha=(0, 0.7)),
            iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
        )),
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)), # add gaussian noise to images
        st(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
        st(iaa.Invert(0.25, per_channel=True)), # invert color channels
        st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        st(iaa.Multiply((0.5, 1.5), per_channel=0.5)), # change brightness of images (50-150% of original value)
        st(iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=ia.ALL, # use any of scikit-image's interpolation methods
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        st(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)) # apply elastic transformations with random strengths
    ],
    random_order=False # do all of the above in random order
)
'''

rotate1 = iaa.Affine(rotate=(-90, -90)) # rotate by -45 to +45 degrees)
rotate2 = iaa.Affine(rotate=(180, 180)) # rotate by -45 to +45 degrees)
rotate3 = iaa.Affine(rotate=(90, 90)) # rotate by -45 to +45 degrees)
#bright = iaa.Add((-10, 10), per_channel=0.5) # change brightness of images (by -10 to 10 of original value)
scale1 = iaa.Affine(scale={"x": (0.8, 0.8), "y": (0.8, 0.8)})
scale2 = iaa.Affine(scale={"x": (1.4, 1.4), "y": (1.4, 1.4)})
flip1 = iaa.Fliplr(1) # horizontally flip all images
flip2 = iaa.Flipud(1) # vertically flip all images
#suppix = iaa.Superpixels(p_replace=(0.9, 1.0), n_segments=(20, 200)) # superpixel representation
multiply = iaa.Multiply((0.5, 1.5), per_channel=1) # change brightness of images (50-150% of original value)
blur = iaa.GaussianBlur((1.0, 3.0)) # blur images with a sigma between 0 and 3.0
contrast = iaa.ContrastNormalization((0.2, 0.9), per_channel=0.5) # improve or worsen the contrast
shear1 = iaa.Affine(shear=(-16, -16)) # shear by -16 to +16 degrees)
shear2 = iaa.Affine(shear=(16, 16)) # shear by -16 to +16 degrees)
shear3 = iaa.Affine(shear=(-26, -26)) # shear by -16 to +16 degrees)
shear4 = iaa.Affine(shear=(26, 26)) # shear by -16 to +16 degrees)
shear5 = iaa.Affine(shear=(-36, -36)) # shear by -16 to +16 degrees)
shear6 = iaa.Affine(shear=(36, 36)) # shear by -16 to +16 degrees)


i = 0
images = []
images_aug = []

directory = "originali/test/"
save_directory = "modified/"

for file in os.listdir(directory):
    #load images
    if not file.startswith('.') and os.path.isfile(os.path.join(directory, file)):
        img = io.imread(directory+file)
        images.append(img)
        print(file)
    
#for batch_idx in range(10):
#perform the transformations
'''
#rotation
images_aug.append(rotate3.augment_images(images))
images_aug.append(rotate2.augment_images(images))
images_aug.append(rotate1.augment_images(images))
#brightness
#images_aug.append(bright.augment_images(images))
#scale
images_aug.append(scale1.augment_images(images))
images_aug.append(scale2.augment_images(images))
#flip
images_aug.append(flip1.augment_images(images))
images_aug.append(flip2.augment_images(images))
# superpixel
#images_aug.append(suppix.augment_images(images))
#moltiplica
images_aug.append(multiply.augment_images(images))
#blur
images_aug.append(blur.augment_images(images))

images_aug.append(contrast.augment_images(images))
'''
images_aug.append(shear1.augment_images(images))
images_aug.append(shear2.augment_images(images))
images_aug.append(shear3.augment_images(images))
images_aug.append(shear4.augment_images(images))
images_aug.append(shear5.augment_images(images))
images_aug.append(shear6.augment_images(images))

#save the images
for _ in images_aug:
    for img in _:
        rescaled = (255.0 / img.max() * (img - img.min())).astype(np.uint8)
        imsave(save_directory+str(i)+'.png',rescaled)        
        i = i+1


'''
#images = np.random.randint(0, 255, (16, 128, 128, 3), dtype=np.uint8)
seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur((0, 3.0))])
aug = iaa.WithChannels(
  channels=[0],
  children=iaa.Add((-30, 30))
)
images_aug = aug.augment_images(images)

print(seq)

# show an image with 8*8 augmented versions of image 0
#seq.show_grid(images[0], cols=8, rows=8)
plt.grid()
plt.show(images[0])

# Show an image with 8*8 augmented versions of image 0 and 8*8 augmented
# versions of image 1. The identical augmentations will be applied to
# image 0 and 1.
#seq.show_grid([images[0], images[1]], cols=8, rows=8)
'''
