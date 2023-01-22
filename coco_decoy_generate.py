seed_value = 0
import random
import cv2
import numpy as np
import os

def generate_decoy(category):
    # loop over ms coco images, and draw square confounding regions on them

    #%% four corners; confounder size: 16x16
    # top left, bottom left, bottom right, top right

    confounders_loc = [[(0,0), (16, 16)], [(0,208), (16, 224)], 
                       [(208,208), (224,224)], [(208,0), (224,16)]]

    im_size = 224
    src_path = "./MSCOCO/images/" + category +"/"
    target_path = "./MSCOCO/confounded/" + category +"/"

    mask_path = "./MSCOCO/confounded_mask/" + category +"/"
    mask = 255 * np.zeros(shape=[224, 224, 3], dtype=np.uint8)

    random.seed(seed_value)
    for i, img_name in enumerate(os.listdir(src_path)):
        # add confounding regions only to training dataset of 1000 images from each category
        if(i==1000): break

        img_pth = src_path + img_name
        img = cv2.imread(img_pth)
        img = cv2.resize(img, (im_size, im_size))

        pixel_intensity_red = random.randint(50, 255)
        pixel_intensity_blue = random.randint(50, 255)
        confound_loc = random.randint(0,3)

        cv2.rectangle(img, 
                      pt1=confounders_loc[confound_loc][0], 
                      pt2=confounders_loc[confound_loc][1], 
                      color=(pixel_intensity_red,0,pixel_intensity_blue), thickness=-1)

        # mask of confounding regions in the same location as above
        # pixel intensity doesn't matter for mask because it will be between 0/1
        cv2.rectangle(mask,
                      pt1=confounders_loc[confound_loc][0], 
                      pt2=confounders_loc[confound_loc][1], 
                      color=(255,255,255), thickness=-1)

        cv2.imwrite(target_path + img_name, img)
        cv2.imwrite(mask_path + img_name, mask)
        # remove rectangle from mask for next iteration
        mask = 255 * np.zeros(shape=[224, 224, 3], dtype=np.uint8)


def main():
    categories = ['zebra', 'train']
    for c in categories:
        generate_decoy(c)
    print('decoy images generated successsfully.')

if __name__ == "__main__":
    main()
