NTDS 2019 Project Team 7 - Movie Recommendation
===
[![Movie_Recommendation](https://img.shields.io/badge/Movie-Recommendation-orange?labelColor=0f4c81&color=8d0045)](https://github.com/dinotuku/ntds-2019-project-team-7)

A movie recommendation is important to our life because of its strength in providing enhanced entertainment. Such a system can suggest a set of movies to users based on their interest, or the popularity of the movies. In this work, we emphasize on building a recommendation system using graph based machine learning. Besides, we also analyze data from Movielens 100k to find out the hidden network structures of movies and users.

## Requirements
* `Python 3.6` for Matrix Factorization and `Python 2.7` for Graph Convolutional Matrix Completion
* `Keras == 2.2.0`
* `Pandas == 0.24.2`
* `Numpy == 1.14.0`
* `Tensorflow == 1.4.0`
* `h5py == 2.10.0`

## Usage
We have two recommendation systems. Here are the steps to reproduce their results:

#### Matrix Factorization (+ DNN) (MF-DNN)

1. Download the data through this Google Drive [links](https://drive.google.com/open?id=1Ppm-Z4BkKFNamBjZnH3K7cvK7tpBsL1s) and put them in `recommenders/mf-dnn/data` 
2. Download the trained models through this Google Drive [links](https://drive.google.com/open?id=1cxkr2ni2F-It0mRf5Bc9vkfptLKfc5bE) and put them in `recommenders/mf-dnn/models`
3. `cd recommenders/mf-dnn`
4. Run one of the three scripts to get our testing results: 
   1. `bash mf.sh` - run the testing code with MF model (latent dimension=16, with ratings normalization)
   2. `bash dnn.sh` - run the testing code with MF + DNN model (latent dimension=64, no ratings normalization, 3 layers with 256 hidden size, dropout=0.5)
   3. `bash dnn_w_info.sh` - run the testing code with MF + DNN with features model (latent dimension=64, no ratings normalization, 3 layers with 256 hidden size, dropout=0.5)
5. You can also train the three models from scratch: 
   1. `python train.py --normal --dim 16` - train the Matrix Factorizaton model
   2. `python train.py --dim 64 --dnn` - train the Matrix Factorization + DNN model
   3. `python train.py --dim 64 --dnn_w_info` - train the Matrix Factorization + DNN with features model

#### Graph Convolutional Matrix Completion (GC-MC)

1. The data should be automatically download if you run the training or testing script. But if it was not downloaded, you can download the data through this Google Drive [links](https://drive.google.com/open?id=1zhxvGefe-fEQS8rDyN6LHqS6cRcaJMrp) and put the folder in `gc-mc/data`
2. Download the trained models through this Google Drive [links](https://drive.google.com/open?id=129mWle-cRLuEVeXFwGKqHGsVTzrdkUiH) and put them in `gc-mc/models`
3. `cd recommenders/gc-mc`
4. Run one of the two scripts to get our testing results:
   1. `bash test_no_features.sh` - run the testing code with the GC-MC model (no additional features)
   2. `bash test_with_features.sh` - run the testing code with the GC-MC model (with features)
5. You can also train (and test) the two models from scratch:
   1. `train_test_no_features.sh` - train and test the GC-MC model (no additional features)
   2. `train_test_with_features.sh` - train and test the GC-MC model (with features)


## Files Description
```
recommenders
|_  gc-mc
    |_  data/: folder for dataset files.
    |_  logs/: folder for log files.
    |_  models/: folder for models.
    |_  data_utils.py: data utility functions, like downloading datasets from the internet.
    |_  initializations.py: different initialization methods for layers.
    |_  layers.py: handles the computations of graph layers.
    |_  metrics.py: different metrics for model evaluation.
    |_  model.py: handles model related tasks, like saving and loading models.
    |_  plot_rmse.py: plots history of training and validation rmse.
    |_  preprocessing.py: preprocessing helper functions.
    |_  test_no_features.sh: script to run the testing code with the GC-MC model (no additional features).
    |_  test_with_features.sh: script to run the testing code with the GC-MC model (with features).
    |_  test.py: testing codes for GC-MC models.
    |_  train_test_no_features.sh: script to train and test the GC-MC model (no additional features).
    |_  train_test_with_features.sh: script to train and test the GC-MC model (with features).
    |_  train.py: experiment runner for GC-MC models.
    |_  utils.py: utility function for constructing feed dict for tensorflow model.
|_  mf-dnn
    |_  data/: folder for dataset files.
    |_  logs/: folder for log files.
    |_  models/: folder for models.
    |_  utils/: folder for utility functions codes.
    |_  dnn_w_info.sh: script to run the testing code with MF + DNN with features model.
    |_  dnn.sh: script to run the testing code with MF + DNN model.
    |_  mf.sh: script to run the testing code with MF model.
    |_  model.py: builds model and create history class.
    |_  plot_loss.py: plots history of training and validation loss.
    |_  test.py: testing codes for MF-DNN models. 
    |_  train.py: experiment runner for MF-DNN models.
|_  parse_data.ipynb: parse Movielens 100k data for mf-dnn codes
```

## Team
| Kuan Tung | Chun-Hung Yeh | Hiroki Hayakawa | Jinhui Guo |
| :---: |:---:| :---:| :---: |
| <img src="https://scontent.ftpe7-3.fna.fbcdn.net/v/t1.0-1/p320x320/44598597_2395336093814687_5861457721299042304_o.jpg?_nc_cat=108&_nc_ohc=S9RMSb64YhoAQkGyn-scFiV2xMyg6XZIv2dDWvzZXFz29QswtojFaU-Ww&_nc_ht=scontent.ftpe7-3.fna&oh=5f0d1fd5c995b718238bd81a7d123faf&oe=5E9D09A9" width=80> | <img src="https://scontent.ftpe7-1.fna.fbcdn.net/v/t1.0-1/p320x320/79498686_2761353167417628_1246618539746394112_o.jpg?_nc_cat=106&_nc_ohc=x3z4iSKGAwwAQkbehCuNPDegEk_Y0iRYHs2Y4V7_QWQ4RO5kCRNPOC55A&_nc_ht=scontent.ftpe7-1.fna&_nc_tp=1&oh=390d2c783d4cbef5e6bd6fb3b3787d82&oe=5E99A09A" width=80> | <img src="https://scontent.ftpe7-1.fna.fbcdn.net/v/t1.0-9/995456_408847005882967_842797001_n.jpg?_nc_cat=100&_nc_ohc=1imVOW7QilEAQkNRtC9TiMAJEkxOELjVATHLoYPgRBD2wOBi5TSGharzw&_nc_ht=scontent.ftpe7-1.fna&oh=72f7972b8339d12c728923f89091f3a9&oe=5EA82A2A" width=80>  | <img src="https://scontent.ftpe7-2.fna.fbcdn.net/v/t1.0-1/p320x320/76762604_477483546200564_68076086340091904_n.jpg?_nc_cat=104&_nc_ohc=ljf__qrYM5EAQmYdsWhQP7SU8_SsGGC_qJ7lIcCVV37yfj588biL06gZQ&_nc_ht=scontent.ftpe7-2.fna&_nc_tp=1&oh=b03d1d0c9ce7c95c0c3809d1c05d3255&oe=5EA1FF48" width=80> |
| <a href="https://github.com/dinotuku" target="_blank">`dinotuku`</a> | <a href="https://github.com/yehchunhung" target="_blank">`yehchunhung`</a> | <a href="https://github.com/hirokihayakawa07" target="_blank">`hirokihayakawa07`</a> | <a href="https://github.com/eternalbetty233" target="_blank">`eternalbetty233`</a> |

## References
* [Graph Convolutional Matrix Completion (paper)](https://arxiv.org/abs/1706.02263)
* [Graph Convolutional Matrix Completion (GitHub repository)](https://github.com/riannevdberg/gc-mc)
* [Matrix Factorization (lecture given by Hung-yi Lee)](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2017/Lecture/MF.pdf)

## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details
