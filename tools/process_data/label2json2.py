import json
from tqdm import tqdm
from glob import glob
import os
from PIL import Image

label_file = '/tcdata/tile_round2_train_20210208/train_annos.json'
target_file = 'data/annotations/train_round2_2.json'
with open(label_file, 'r', encoding='utf-8') as f:
    labels = json.load(f)
categoryid2name = {
    #   "0": "背景",
    "1": "边异常",
    "2": "角异常",
    "3": "白色点瑕疵",
    "4": "浅色块瑕疵",
    "5": "深色点块瑕疵",
    "6": "光圈瑕疵",
    "7": "记号笔",
    "8": "划伤"
}
categories = []
for cateid, name in categoryid2name.items():
    categories.append({"id": int(cateid), "name": name})
images = []
imgnames = []
annotations = []

image_id = 1
anno_id = 1
imgname2anno = {}
# imgname2id = {}
for label in labels:
    file_name = label['name']
    if file_name not in imgname2anno:
        imgname2anno[file_name] = []
    category_id = label['category']
    box = label['bbox']
    imgname2anno[file_name].append([box, category_id])

nobox = 0
for imagepath in tqdm(
        glob('/tcdata/tile_round2_train_20210208/train_imgs/*.jpg')):
    file_name = os.path.basename(imagepath)
    # for imgname, annos in imgname2anno.items():
    #     file_name = imgname
    #     image_id = imgname2id[imgname]
    width, height = Image.open(imagepath).size
    images.append({"file_name": file_name, "id": image_id, "height": height, "width": width})
    if file_name not in imgname2anno:
        image_id += 1
        nobox += 1
        continue
    for anno in imgname2anno[file_name]:
        box, category_id = anno
        xmin, ymin, xmax, ymax = box
        w = xmax - xmin
        h = ymax - ymin
        annotations.append(
            {"id": anno_id, "image_id": image_id, "bbox": [xmin, ymin, w, h], "area": w * h, "segmentation": [[]],
             "category_id": category_id,
             "iscrowd": 0})
        anno_id += 1
    image_id += 1
instances = {"images": images, "annotations": annotations, "categories": categories}
with open(target_file, "w") as f:
    json.dump(instances, f, indent=1)

print(nobox)