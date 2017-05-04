# tensorFeatureExtraction
This is a MATLAB script for feature extraction of multi-dimensional data.
This repository contains two kinds of algorithms, feature extraction with the higher-order orthogonal iteration [1] and with tensor-train decomposition [2]. 
I implemented these feature extraction algorithms and experimented their accuracy with MNIST hand-written digit dataset. 

Installation:  
1. Install [pmtk3](https://github.com/probml/pmtk3).  
2. Install [Tensor Toolbox](http://www.sandia.gov/~tgkolda/TensorToolbox/).  
3. Install [TT-Toolbox](https://github.com/oseledets/TT-Toolbox).
4. clone our code:  
$git clone git@github.com:YoshiHotta/tensorFeatureExtraction.git  
5. Run script files (src/*_script.m)  

Credits.  
The algorithms were proposed by the following articles and are NOT my results.  
[1] Phan, Anh Huy, and Andrzej Cichocki. "Tensor decompositions for feature extraction and classification of high dimensional datasets." Nonlinear theory and its applications, IEICE 1.1 (2010): 37-68.    
[2] Bengua, Johann A., Ho N. Phien, and Hoang D. Tuan. "Optimal feature extraction and classification of tensors via matrix product state decomposition." Big Data (BigData Congress), 2015 IEEE International Congress on. IEEE, 2015.   
The script uses the libraries above in addition to the function that is included as appendix of the following paper:  
[3] [ncon](https://arxiv.org/abs/1402.0939)   

