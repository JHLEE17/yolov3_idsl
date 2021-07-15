# ReadMe

## **This repository is forked version of ultralytics/yolov3.**

# **IDEC 실습용 자료**

[https://camo.githubusercontent.com/93b386d6287ad65a71643520f0339bf84d1aaf7259c6b379136ff8743bc9754c/68747470733a2f2f6964736c2e73656f756c746563682e61632e6b722f696d6167652f4944534c2d6c6f676f322e706e67](https://camo.githubusercontent.com/93b386d6287ad65a71643520f0339bf84d1aaf7259c6b379136ff8743bc9754c/68747470733a2f2f6964736c2e73656f756c746563682e61632e6b722f696d6167652f4944534c2d6c6f676f322e706e67)

[https://camo.githubusercontent.com/a39a561c67903f765357b73ae6b228c3ce4d80709027f476e30debac34a1895f/68747470733a2f2f696d672e65746e6577732e636f6d2f70686f746f6e6577732f313630362f3831373632395f32303136303632393135343230355f3637335f303030312e6a7067](https://camo.githubusercontent.com/a39a561c67903f765357b73ae6b228c3ce4d80709027f476e30debac34a1895f/68747470733a2f2f696d672e65746e6577732e636f6d2f70686f746f6e6577732f313630362f3831373632395f32303136303632393135343230355f3637335f303030312e6a7067)

This repository represents Ultralytics open-source research into future object detection methods, and incorporates lessons learned and best practices evolved over thousands of hours of training and evolution on anonymized client datasets. **All code and models are under active development, and are subject to modification or deletion without notice.** Use at your own risk.

![https://user-images.githubusercontent.com/26833433/114424655-a0dc1e00-9bb8-11eb-9a2e-cbe21803f05c.png](https://user-images.githubusercontent.com/26833433/114424655-a0dc1e00-9bb8-11eb-9a2e-cbe21803f05c.png)

- Figure Notes (click to expand)
    - GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS.
    - EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.
    - **Reproduce** by `python test.py --task study --data coco.yaml --iou 0.7 --weights yolov3.pt yolov3-spp.pt yolov3-tiny.pt yolov5l.pt`

## **Branch Notice**

The [ultralytics/yolov3](https://github.com/ultralytics/yolov3) repository is now divided into two branches:

- [Master branch](https://github.com/ultralytics/yolov3/tree/master): Forward-compatible with all [YOLOv5](https://github.com/ultralytics/yolov5) models and methods (**recommended** ✅).

`$ git clone https://github.com/ultralytics/yolov3  # master branch (default)`

- [Archive branch](https://github.com/ultralytics/yolov3/tree/archive): Backwards-compatible with original [darknet](https://pjreddie.com/darknet/) *.cfg models (**no longer maintained** ⚠️).

`$ git clone https://github.com/ultralytics/yolov3 -b archive  # archive branch`

## **Pretrained Checkpoints**

[Untitled](https://www.notion.so/d13c3b9faaeb4cd5a4997ac3753399ad)

- Table Notes (click to expand)

## **Requirements**

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov3/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:

`$ pip install -r requirements.txt`

## **Tutorials**

- [Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data) 🚀 RECOMMENDED
- [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results) ☘️ RECOMMENDED
- [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289) 🌟 NEW
- [Supervisely Ecosystem](https://github.com/ultralytics/yolov5/issues/2518) 🌟 NEW
- [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
- [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) ⭐ NEW
- [TorchScript, ONNX, CoreML Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
- [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
- [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
- [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
- [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
- [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314) ⭐ NEW
- [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)

## **Environments**

YOLOv3 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab and Kaggle** notebooks with free GPU:

    [https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667](https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667)

    [https://camo.githubusercontent.com/a08ca511178e691ace596a95d334f73cf4ce06e83a5c4a5169b8bb68cac27bef/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667](https://camo.githubusercontent.com/a08ca511178e691ace596a95d334f73cf4ce06e83a5c4a5169b8bb68cac27bef/68747470733a2f2f6b6167676c652e636f6d2f7374617469632f696d616765732f6f70656e2d696e2d6b6167676c652e737667)

- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/GCP-Quickstart)
- **Amazon** Deep Learning AMI. See [AWS Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/AWS-Quickstart)
- **Docker Image**. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov3/wiki/Docker-Quickstart)

    [https://camo.githubusercontent.com/dd3bcb559ac1acab72d1f935da34727783cf30e8217bdd61af28141f2d005415/68747470733a2f2f696d672e736869656c64732e696f2f646f636b65722f70756c6c732f756c7472616c79746963732f796f6c6f76333f6c6f676f3d646f636b6572](https://camo.githubusercontent.com/dd3bcb559ac1acab72d1f935da34727783cf30e8217bdd61af28141f2d005415/68747470733a2f2f696d672e736869656c64732e696f2f646f636b65722f70756c6c732f756c7472616c79746963732f796f6c6f76333f6c6f676f3d646f636b6572)

## **Inference**

`detect.py` runs inference on a variety of sources, downloading models automatically from the [latest YOLOv3 release](https://github.com/ultralytics/yolov3/releases) and saving results to `runs/detect`.

`$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream`

To run inference on example images in `data/images`:

`$ python detect.py --source data/images --weights yolov3.pt --conf 0.25`

![https://user-images.githubusercontent.com/26833433/100375993-06b37900-300f-11eb-8d2d-5fc7b22fbfbd.jpg](https://user-images.githubusercontent.com/26833433/100375993-06b37900-300f-11eb-8d2d-5fc7b22fbfbd.jpg)

### **PyTorch Hub**

To run **batched inference** with YOLOv3 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36):

`import torch

# Model
model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or 'yolov3_spp', 'yolov3_tiny'

# Image
img = 'https://ultralytics.com/images/zidane.jpg'

# Inference
results = model(img)
results.print()  # or .show(), .save()`

## **Training**

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov3/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv3/YOLOv3-SPP/YOLOv3-tiny are 6/6/2 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).

`$ python train.py --data coco.yaml --cfg yolov3.yaml      --weights '' --batch-size 24
                                         yolov3-spp.yaml                            24
                                         yolov3-tiny.yaml                           64`

![https://user-images.githubusercontent.com/26833433/100378028-af170c80-3012-11eb-8521-f0d2a8d021bc.png](https://user-images.githubusercontent.com/26833433/100378028-af170c80-3012-11eb-8521-f0d2a8d021bc.png)

## **Citation**

[https://camo.githubusercontent.com/a840127e9461d7325ca3ca2cae13e9163104a8d00f3b6a05b17fb70f5d994e53/68747470733a2f2f7a656e6f646f2e6f72672f62616467652f3134363136353838382e737667](https://camo.githubusercontent.com/a840127e9461d7325ca3ca2cae13e9163104a8d00f3b6a05b17fb70f5d994e53/68747470733a2f2f7a656e6f646f2e6f72672f62616467652f3134363136353838382e737667)

## **About Us**

Ultralytics is a U.S.-based particle physics and AI startup with over 6 years of expertise supporting government, academic and business clients. We offer a wide range of vision AI services, spanning from simple expert advice up to delivery of fully customized, end-to-end production solutions, including:

- **Cloud-based AI** systems operating on **hundreds of HD video streams in realtime.**
- **Edge AI** integrated into custom iOS and Android apps for realtime **30 FPS video inference.**
- **Custom data training**, hyperparameter evolution, and model exportation to any destination.

For business inquiries and professional support requests please visit us at [https://ultralytics.com](https://ultralytics.com/).

## **Contact**

**Issues should be raised directly in the repository.** For business inquiries or professional support requests please visit [https://ultralytics.com](https://ultralytics.com/) or email Glenn Jocher at [glenn.jocher@ultralytics.com](mailto:glenn.jocher@ultralytics.com).