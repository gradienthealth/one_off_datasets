"""objectcxr dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os
import collections
import pandas as pd
from PIL import Image, ImageDraw

_URL = """https://web.archive.org/web/20201127235812/https://jfhealthcare.github.io/object-CXR/"""

# TODO(objectcxr): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """Automatic detection of foreign objects on chest X-rays"""

_CITATION = """\
@article{,
title= {object-CXR - Automatic detection of foreign objects on chest X-rays},
keywords= {radiology},
author= {JF Healthcare},
abstract= {## Data
5000 frontal chest X-ray images with foreign objects presented and 5000 frontal chest X-ray images without foreign objects were filmed and collected from about 300 township hosiptials in China. 12 medically-trained radiologists with 1 to 3 years of experience annotated all the images. Each annotator manually annotates the potential foreign objects on a given chest X-ray presented within the lung field. Foreign objects were annotated with bounding boxes, bounding ellipses or masks depending on the shape of the objects. Support devices were excluded from annotation. A typical frontal chest X-ray with foreign objects annotated looks like this:

https://i.imgur.com/SFUZy80.jpg


## Annotation

Object-level annotations for each image, which indicate the rough location of each foreign object using a closed shape.

Annotations are provided in csv files and a csv example is shown below.

```csv
image_path,annotation
/path/#####.jpg,ANNO_TYPE_IDX x1 y1 x2 y2;ANNO_TYPE_IDX x1 y1 x2 y2 ... xn yn;...
/path/#####.jpg,
/path/#####.jpg,ANNO_TYPE_IDX x1 y1 x2 y2
...
```

Three type of shapes are used namely rectangle, ellipse and polygon. We use `0`, `1` and `2` as `ANNO_TYPE_IDX` respectively.

- For rectangle and ellipse annotations, we provide the bounding box (upper left and lower right) coordinates in the format `x1 y1 x2 y2` where `x1` < `x2` and `y1` < `y2`.

- For polygon annotations, we provide a sequence of coordinates in the format `x1 y1 x2 y2 ... xn yn`.

> ### Note:
> Our annotations use a Cartesian pixel coordinate system, with the origin (0,0) in the upper left corner. The x coordinate extends from left to right; the y coordinate extends downward.

## Organizers
[JF Healthcare](http://www.jfhealthcare.com/) is the primary organizer of this challenge.
},
terms= {},
license= {https://creativecommons.org/licenses/by-nc/4.0/},
superseded= {},
url= {https://web.archive.org/web/20201127235812/https://jfhealthcare.github.io/object-CXR/}
}
"""

ANNO_TYPE_IDX = [
    'rectangle',
    'ellipse', 
    'polygon'
]




class Objectcxr(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for objectcxr dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
        manual_dir should contain images. Use the wasabi s3://gradient-scratch/abhi/datasets/
    """

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(objectcxr): Specifies the tfds.core.DatasetInfo object
    anno_type_label = tfds.features.ClassLabel(names=ANNO_TYPE_IDX)

    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "image" : tfds.features.Image(shape=(None, None, 1)),
            "image/filename" : tfds.features.Text(),
            "bboxes": tfds.features.Sequence(tfds.features.BBoxFeature()),
            "anno_type": tfds.features.Sequence(anno_type_label)
            #"segmask" : tfds.features.Image(shape=(None, None, 1))
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'bboxes'),  # e.g. ('image', 'label')
        homepage=_URL,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    manual_dir = "s3://gradient-scratch/abhi/datasets/"
    # manual_dir = "/content/drive/MyDrive/Data/object-CXR"
    train_labels = os.path.join(manual_dir, 'train.csv')
    test_labels = os.path.join(manual_dir, 'dev.csv')
    train_dir = os.path.join(manual_dir, 'train')
    test_dir = os.path.join(manual_dir, 'dev')


    if not (tf.io.gfile.exists(train_labels) or 
                        tf.io.gfile.exists(test_labels) or
                        tf.io.gfile.exists(train_dir) or
                        tf.io.gfile.exists(test_dir)):
            msg = "You must download the dataset files manually and unzip them in: {}, {}, {}, and {}".format(train_labels, test_labels, train_dir, test_dir)
            raise AssertionError(msg)
    
    train_labels_df = pd.read_csv(train_labels)
    test_labels_df = pd.read_csv(test_labels)
    

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
              'images_dir' : train_dir,
              'labels_df' : train_labels_df
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
              'images_dir' : test_dir,
              'labels_df' : test_labels_df
            },
        ),

    ]

  def _generate_examples(self, images_dir=None, labels_df=None):
    """Yields examples."""
    # TODO(objectcxr): Yields (key, example) tuples from the dataset
    for idx, row in labels_df.iterrows():
      img = Image.open(os.path.join(images_dir, row['image_name']))
      width = img.width
      height = img.height 
      del(img)
      segmask = Image.new("L", (width, height))
      draw = ImageDraw.Draw(segmask)
      bboxes = []
      annotypes = []
      if (str(row["annotation"]) != "nan"):
        raw_boxes = row["annotation"].split(";")
        for box in raw_boxes: 
          coords = box.split()
          if (coords[0] == 0 or coords[0] == 1):
            if (coords[0] == '0'): 
              annotype = 'rectangle'
            else:
              annotype = 'ellipse'
            bboxes.append(tfds.features.BBox(ymin=float(coords[2])/height, xmin=float(coords[1])/width, ymax=float(coords[4])/height, xmax=float(coords[3])/width))
            annotypes.append(annotype)
          else:
            annotype = 'polygon'
            it = iter(coords[1:])
            xmin, ymin = float('inf'), float('inf')
            xmax, ymax = float('-inf'), float('-inf')
            for val in it:
              x = float(val)
              if xmin > x and x >= 0 and x < width:
                xmin = x
              if xmax < x and x >= 0 and x < width:
                xmax = x
              y = float(next(it))
              if ymin > y and y >= 0 and y < height:
                ymin = y
              if ymax < y and y >= 0 and y < height:
                ymax = y
            bboxes.append(tfds.features.BBox(ymin=float(ymin)/height, xmin=float(xmin)/width, ymax=float(ymax)/height, xmax=float(xmax)/width))
            annotypes.append(annotype)
      yield idx, {
        "image" : os.path.join(images_dir, row['image_name']),
        "image/filename" : row['image_name'],
        "bboxes": bboxes,
        "anno_type": annotypes
        #"segmask": np.asarray(segmask)
      }
      #if idx == 100: break # remove after debugging 


