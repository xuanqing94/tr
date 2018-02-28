# tr
code for *Better Generalization by Efficient Trust Region Method*, Liu et al. 2018

## How-to
0. `cd tr/` 
1. Create directory for storing model `mkdir vgg16`
2. Create directories for stroing different kind of batch size b: `mkdir [vgg16/b128 | vgg16/b256 | vgg16/b512 | vgg16/b1024 ... ]`
3. Start training `./run.sh`, you might need to change the arguments (most of them are self-explainatory).
