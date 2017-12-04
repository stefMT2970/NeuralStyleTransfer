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
4. copy the instructions in main.py to your IPython console
5. The output is in output dir

On a laptop equipped wiht NVIDIA GTX960M (2GB RAM) it takes 90" to train.
There is no error checking, make sure your images are at least as large as
CONFIG.IMAGE_HEIGHT and CONFIG.INMAGE_WIDTH.


Ref : https://www.coursera.org/learn/convolutional-neural-networks
