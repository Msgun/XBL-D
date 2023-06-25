from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import requests
import os

# gets all images containing given categories and save them to pre-created subdirectories
def save_masks_images(coco, category, category_id):
    catIds = coco.getCatIds(catNms=[category]) #zebra train
    imgIds = coco.getImgIds(catIds=catIds)
    images = coco.loadImgs(imgIds)
    
    # save annotaions
    for ctr, img_id in enumerate(imgIds):
        cat_ids = coco.getCatIds()
        img = coco.imgs[img_id]

        anns_ids = coco.getAnnIds(imgIds=img_id, catIds=category_id, iscrowd=None) 
        anns = coco.loadAnns(anns_ids)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])
        plt.imsave('./MSCOCO/annotations/' + category + '/' + img['file_name'] , mask)
        if(ctr>1298): break # stop after reading 1300 annotations

    # Save images into a local folder
    for ctr, im in enumerate(images):
        img_data = requests.get(im['coco_url']).content
        with open('./MSCOCO/images/' + category + '/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)
        if(ctr>1298): break # stop after reading 1300 annotations

def main():
    # prepare directories and subdirectories
    os.mkdir('./MSCOCO')
    os.mkdir('./MSCOCO/images')
    os.mkdir('./MSCOCO/annotations')
    
    coco = COCO('./instances_train2014.json')
    categories = ['zebra', 'train']
    category_id = [24, 7] # #24 for zebra; 7 is train
    for ctr, c in enumerate(categories):
        os.mkdir(os.path.join('./MSCOCO/annotations', c))
        os.mkdir(os.path.join('./MSCOCO/images', c))
        
        save_masks_images(coco, c, category_id[ctr])
    print('annotations and images stored successsfully.')

if __name__ == "__main__":
    main()
