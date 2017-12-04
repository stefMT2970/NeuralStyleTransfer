# NeuralStyleTransfer
## Art Generation with Neural Style Transfer 

### Work instructions

The code is tested on a Windows 10 Anaconda python3.5 distribution, using tensorflow.

1. Clone to your desktop.
2. Keep directory structure
    /images
    /output
    /pretrained-model
    
3. modify Class CONFIG in nst_utils to enter your parameters
4. download imagenet-vgg-verydeep-19.mat from the web and store in a directory ./pretrained-model
5. copy the instructions in main.py to your IPython console.
6. The output (generated images) is in ./output directory.


On a laptop equipped wiht NVIDIA GTX960M (2GB RAM) it takes 90" to train (using 100 iterations).
There is no error checking, make sure your images are at least as large as
CONFIG.IMAGE_HEIGHT and CONFIG.IMAGE_WIDTH.

The purpose of this code is to play with it, learn, modify etc ...
It is not an application you can start from a command line.

Ref : https://www.coursera.org/learn/convolutional-neural-networks
