Detectron平台做检测的应该都知道，网上关于配置的教程也很多，应该使用的挺多的。该平台对COCO数据集支持良好。
Cityperson数据集，在16年CVPR上被提出，是张姗姗一波人在cityscapes数据集上进行标注得到的行人检测数据集。做行人检测的应该都不陌生。[论文地址](https://arxiv.org/abs/1702.05693)，[数据库地址](https://www.cityscapes-dataset.com/)，[张姗姗提供的数据库地址](https://bitbucket.org/shanshanzhang/citypersons)
这篇文章中，我将详细介绍如何在Detectron平台下实现Cityperson数据集的训练和测试。

首先下载Cityperson数据集，网站上提供了很多的下载选项，下载需要注册。对于行人检测任务，下载这两个文件就行了：
1. [train+val图片](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
2. [cityperson标注](https://www.cityscapes-dataset.com/file-handling/?packageID=28)
需要说明的是，目前cityperson只公布了训练和验证数据集，没有公布测试数据。希望尽早公布吧。

接下来要做两个工作，一个是数据库的转换，另外一个是detectron平台相应代码的修改。
首先说数据库的转换吧：

这里我是将cityperson数据集转换成coco集之后用来训练的。虽然detectron平台对citysapes数据集有支持，但是似乎不是行人检测这一块的。没有过多的研究，如果有同样在做这类似工作的小伙伴，欢迎通过各种方法联系我，因为我确实也没弄得太清楚。

cityperson数据集的标注文件，是每一个图片对应一个标注json文件，而coco的标注格式，可以参考[官网](http://cocodataset.org/)上的说明，需要强调的就是他是整个集对应一个json文件，每一个图片对应一个唯一的图片编号。

因为不能看到coco集下具体的形式，为了快速解决这个问题，我决定用现有的代码实现。在github上面搜索了一下做数据库转换的代码。发现只有这个：[cityperson2voc](https://github.com/Microos/citypersons2voc)，拿下来试了一下，确实可以用。具体的使用方法参照代码中的说明即可。现在我们得到了一个voc形式的cityperson数据集。其实detectron平台对voc也支持，但是我看了一下代码。他支持的voc数据集仍然需要一个总体的json标注文件，而不是传统的voc那样每一个图片对应的xml文件。这时需要使用voc到coco的转换工具了，这样的转换工具，github上也有现成的，可以拿来直接用。不过对于转换过来的cityperson数据集，需要对代码进行一点点修改：
[代码地址](https://github.com/shiyemin/voc2coco)
修改部分如下面注释;
```python
#!/usr/bin/python

# pip install lxml

import sys
import os
import json
import xml.etree.ElementTree as ET


START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}
# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
                         #  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
                         #  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
                         #  "motorbike": 14, "person": 15, "pottedplant": 16,
                         #  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
######################这里加一行###########################
        filename = filename.split('_')[1] + filename.split('_')[2]
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')
    json_dict = {"images":[], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        print("Processing %s"%(line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
########################下面部分改成######################
            #xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            #ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            #xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            #ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text)) - 1
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text)) - 1
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox':[xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('3 auguments are need.')
        print('Usage: %s XML_LIST.txt XML_DIR OUTPU_JSON.json'%(sys.argv[0]))
        exit(1)

    convert(sys.argv[1], sys.argv[2], sys.argv[3])
```
理由说明：
修改1：原代码针对的是voc数据库，读取了图片文件名的编号作为coco集下图片的id，但是对于cityperson数据集，文件名的格式和voc不同，所以需要稍作处理。
修改2：cityperson和voc中bbox的数据格式不同，在这里会报数据转换的错，先将其转换为float再转为int可以快速解决问题。

*运行该代码还需要提供一个xml文件的目录txt，很多方法实现了。如果不想写程序的话，去imageset文件下找到train和test文件的目录txt，使用文本编辑器替换的方式，再每一个项目末尾加上.xml即可。*

至此，cityperson数据集的转换工作已经完成。将其链接到Detectron文件夹下的detectron/datasets/data下，我们开始修改Detectron平台的代码，这里参照了博客[Caffe2 - (十九) 基于 Detectron 的 DeepFashion 服装 bbox 检测实现](https://blog.csdn.net/zziahgf/article/details/79488025)

首先修改detectron/datasets/dataset_catalog.py文件，用于添加新的数据集：
按照之前的格式，在DATASETS中增加两项：
```python
    'cityperson_train': {
        IM_DIR:
            _DATA_DIR + '/cityperson/data/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/cityperson/data/cityperson_train.json'
    },
    'cityperson_val': {
        IM_DIR:
            _DATA_DIR + '/cityperson/data/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/cityperson/data/cityperson_val.json'
    },
    },
```
然后修改网络文件：
configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml
只需要修改三项：
```yaml
NUM_CLASSES: 2 # 一个类别 + 一个background 类
TRAIN：
DATASETS: ('cityperson_train',)
 TEST：
DATASETS: ('cityperson_val',)
```
这时应该已经可以开始训练了，
```bash
python tools/train_net.py --cfg ./configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml OUTPUT_DIR ./detectron-output
```
我印象中前面某个位置会报错，需要将cityperson中的jpg图片转换为png图片，用下面的指令即可：
```bash
$ ls -1 *.jpg | xargs -n 1 bash -c 'convert "$0" "${0%.jpg}.png"'
```

在1080ti下单张卡训练大概是两个小时，不得不感叹相比与caffe，caffe2很快了。
训练完成后，会报没有验证函数的错。对啊，就是没有。这一块我目前正在解决，不过可以使用coco默认的测试函数算一下，虽然和行人检测中常用的MR-FPPI不相同，不过也可以反应一下训练效果。不过从我的实验来看，训练算成功了，但是效果不太理想。接下来会去找问题。也欢迎大家和我讨论。
将验证函数默认为coco，需要修改detectron/core/config.py下的项
```python
_C.TEST.FORCE_JSON_DATASET-EVAL = Ture
```
当然你也可以和demo一样，可视化一下检测结果：
修改tools/infer_simple.py
```
dummy_coco_dataset = dummy_datasets.get_cityperson_dataset()
```
然后在detectron/datasets/dummy_datasets.py中增加函数：
```python
def get_cityperson_dataset():
    ds = AttrDict()
    classes = ['__background__', 'ped',  ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds
```
然后仿照demo运行infer_simple.py就行了。

