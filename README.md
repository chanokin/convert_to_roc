# Utility scripts to convert different image datasets into rank-order coded spiking representations. 


**Usage:**

    python convert.py dataset path/to/input/files [options]

    --timestep         Timestep which will be used in the simulations. How many spikes will be emmited 
                       at each timestep can be set with --spikes_per_bin
    --percent          How many of the possible spikes (number of pixels) should we 
                       output. Percent (0.0 < p <= 1.0)
    --output_dir       Path to the output location of the generated spike files
    --skip_existing    Whether to skip database entries corresponding to files already found in the 
                       output directory
    --spikes_per_bin   How many spikes per timestep will be emmited. Note that more than one is not 
                       standard rank-order encoding
    --scaling          Scaling applied to the input image (only supported by the Omniglot dataset)

    
---    

## About the datasets and conversion method:

1. ___Databases___ in this repository are the property of their authors:

   * __[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)__: [Alex Krizhevsky, (2009). Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
      * This database has color images which get transformed into a YUV encoding. The conversion outpus rank-order coded spikes for each of the images (grayscale [Y], blue-yellow [U],  red-green [V])
   * __[MNIST](http://yann.lecun.com/exdb/mnist/)__: [Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition.](http://yann.lecun.com/exdb/publis/index.html#lecun-98) Proceedings of the IEEE, 86(11):2278-2324, November 1998.
      * The images in the dataset are converted so that background value is 0 and digit regions values are 255. The output is rank-order coded spikes from the 28x28 images.
   * __[OMNIGLOT](https://github.com/brendenlake/omniglot)__: [Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction.](http://www.sciencemag.org/content/350/6266/1332.short) _Science_, 350(6266), 1332-1338.
      * The images in the dataset are converted so that background value is 0 and digit regions values are greater than 0; furthermore, the images are scaled using the `--scaling` parameter. The output is rank-order coded spikes from the scaled images.

2. The (grayscale) ___transformation method___ was:
   * Created by __[Basabdatta Sen Bhattacharya](https://sites.google.com/site/bsenbhattacharya/)__
   * Implemented for the __[NE15 Database](https://github.com/NEvision/NE15)__ reported in [Liu Qian, Pineda García Garibaldi, Stromatias Evangelos, Serrano-Gotarredona Teresa, Furber Steve B. (2016) Benchmarking Spike-Based Visual Recognition: A Dataset and Evaluation](https://www.frontiersin.org/article/10.3389/fnins.2016.00496) Frontiers in Neuroscience
