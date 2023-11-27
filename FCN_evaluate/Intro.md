This is the modified one convert 'caffe version' to 'pytorch version', like a reimplementation. But it requires a pre-trained .pth file and other components are already done.

1.use FCN8s_cityscapes.pth as the pre-trained model, but there is no pre-trained FCN8s model on cityscapes online and due to time limitation, we do not train the FCN8s on the cityscapes.

2.evaluate.py is to test and get the score. parameters are required to run .py file and instructions are in itself.

tips:


align.py try to align the model structure between the model and .pth file


scripts are the must files to handle with the cityscapes datasets.


The normData.py aims to rename the images into right format and move the translated images into a folder.
