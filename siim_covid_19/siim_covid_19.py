"""siim-covid-19 dataset."""
import ast
import csv
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pydicom import dcmread

# TODO(siim-covid-19): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
"""

# TODO(siim-covid-19): BibTeX citation
_CITATION = """
"""


class Siim_covid_19(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for siim-covid-19 dataset."""

    VERSION = tfds.core.Version('1.0.1')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        '1.0.1': 'Added supervised target'
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(None, None, 1)),
                'bboxes': tfds.features.Sequence({'bbox': tfds.features.BBoxFeature()}),
                'image_id': tfds.features.Text(),
                'series_id': tfds.features.Text(),
                'study_id': tfds.features.Text(),
                'category': tfds.features.ClassLabel(names=['negative', 'typical', 'atypical', 'indeterminate'])
            }),
            supervised_keys=('image', 'category'),
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        files_dir = 's3://gradient-scratch/kerry/siim-covid19-detection'
        download_dir = dl_manager.download_and_extract(
            'https://s3.wasabisys.com/gradient-scratch/kerry/siim-covid19-detection/train.tar.gz')
        return {
            'train': self._generate_examples(
                images=download_dir / 'train',
                image_csv=files_dir + '/train_image_level.csv',
                study_csv=files_dir + '/train_study_level.csv'
            )
        }

    def _generate_examples(self, images, image_csv, study_csv):
        """Yields examples."""
        # Set AWS environment keys here or comment out and use aws configure
        os.environ["AWS_ACCESS_KEY_ID"] = ""
        os.environ["AWS_SECRET_ACCESS_KEY"] = ""
        os.environ["AWS_REGION"] = ""
        os.environ["S3_ENDPOINT"] = ""
        os.environ["S3_USE_HTTPS"] = ""
        os.environ["S3_VERIFY_SSL"] = ""
        raise NotImplementedError('Set AWS keys or use aws configure, then comment out this line')

        study_dict = {}
        with tf.io.gfile.GFile(study_csv) as f:
            for row in csv.DictReader(f):
                category = ''
                if row['Negative for Pneumonia'] == '1':
                    category = 'negative'
                elif row['Typical Appearance'] == '1':
                    category = 'typical'
                elif row['Indeterminate Appearance'] == '1':
                    category = 'indeterminate'
                elif row['Atypical Appearance'] == '1':
                    category = 'atypical'
                else:
                    raise ValueError('Malformed data')
                study_dict[row['id'][:-6]] = category
        print(f'Read {len(study_dict)} studies from csv')

        image_dict = {}
        with tf.io.gfile.GFile(image_csv) as f:
            for row in csv.DictReader(f):
                boxes = list()
                if row['boxes'] != '':
                    boxes = ast.literal_eval(row['boxes'])
                for box in boxes:
                    box['xmin'] = box['x']
                    box['ymin'] = box['y']
                    box['xmax'] = box['x'] + box['width']
                    box['ymax'] = box['y'] + box['height']
                    box.pop('x')
                    box.pop('y')
                    box.pop('width')
                    box.pop('height')
                image_dict[row['id'][:-6]] = {
                    'boxes': boxes,
                    'study': row['StudyInstanceUID']
                }
        print(f'Read {len(image_dict)} images from csv')

        for study in os.listdir(images):
            for series in os.listdir(f'{images}/{study}'):
                for file in os.listdir(f'{images}/{study}/{series}'):
                    with dcmread(f'{images}/{study}/{series}/{file}') as ds:
                        image_id, e = os.path.splitext(file)
                        boxes = image_dict[image_id]['boxes']
                        category = study_dict[study]
                        image = ds.pixel_array
                        image -= np.amin(image)
                        image = image / np.amax(image)
                        image *= 255
                        image = image.astype(np.uint8)
                        y, x = image.shape
                        tf_boxes = [{
                            'bbox': tfds.features.BBox(
                                    xmin=np.clip(box['xmin'] / x, 0, 1),
                                    xmax=np.clip(box['xmax'] / x, 0, 1),
                                    ymin=np.clip(box['ymin'] / y, 0, 1),
                                    ymax=np.clip(box['ymax'] / y, 0, 1)
                                    )
                        } for box in boxes]
                        if ds['PhotometricInterpretation'].value == 'MONOCHROME1':
                            image = np.invert(image)
                        image = image.reshape(y, x, 1)

                        yield image_id, {
                            'image': image,
                            'bboxes': tf_boxes,
                            'image_id': image_id,
                            'series_id': series,
                            'study_id': study,
                            'category': category
                        }
