# siim-covid-19
Dataset built from Kaggle competition [here](https://www.kaggle.com/c/siim-covid19-detection/overview):

Example Colab notebook [here](https://colab.research.google.com/drive/1uKIIWXMN6_smJLlsTrK49XxQOezjmzUf?usp=sharing)

Requires ~220GB disk space to build from scratch

Remember to set AWS keys either in the script or using `aws configure`, then remove the `NotImplementedError` line in the script

Dependencies:
- tensorflow_datasets==4.3.0
- numpy==1.21.0
- pydicom
- pylibjpeg
- pylibjpeg-libjpeg
- pylibjpeg-openjpeg
- pylibjpeg-rle