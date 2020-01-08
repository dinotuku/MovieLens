NTDS 2019 Project Team 7 - Movie Recommendation
===
[![Movie_Recommendation](https://img.shields.io/badge/Movie-Recommendation-orange?labelColor=0f4c81&color=8d0045)]()

A movie recommendation is important to our life because of its strength in providing enhanced entertainment. Such a system can suggest a set of movies to users based on their interest, or the popularity of the movies. In this work, we emphasize on building a recommendation system using graph based machine learning. Besides, we also analyze data from Movielens 100k to find out the hidden network structures of movies and users.

## Requirements
* `Python 3.6` for mf-dnn and `Python 2.7` for gc-mc
* `Keras == 2.2.0`
* `Pandas == 0.24.2`
* `Numpy == 1.14.0`
* `Tensorflow == 1.4.0`
* `h5py == 2.10.0`

## Files Description
```
recommenders
|_  gc-mc
    |_  data/: 
    |_  logs/:
    |_  models/:
    |_  data_utils.py: 
    |_  initializations.py:
    |_  layers.py: 
    |_  metrics.py: 
    |_  model.py: 
    |_  preprocessing.py: 
    |_  train_test.sh: 
    |_  train.py: 
    |_  utils.py:
|_  mf-dnn
    |_  data/: 
    |_  logs/: 
    |_  models/: 
    |_  utils/: 
    |_  dnn_w_info.sh: 
    |_  dnn.sh: 
    |_  mf.sh: 
    |_  model.py: 
    |_  test.py: 
    |_  train.py: 
```

## Team
| Kuan Tung | Chun-Hung Yeh | Hiroki Hayakawa | Jinhui Guo |
| :---: |:---:| :---:| :---: |
| <img src="https://scontent.ftpe7-3.fna.fbcdn.net/v/t1.0-1/p320x320/44598597_2395336093814687_5861457721299042304_o.jpg?_nc_cat=108&_nc_ohc=S9RMSb64YhoAQkGyn-scFiV2xMyg6XZIv2dDWvzZXFz29QswtojFaU-Ww&_nc_ht=scontent.ftpe7-3.fna&oh=5f0d1fd5c995b718238bd81a7d123faf&oe=5E9D09A9" width=80> | <img src="https://scontent.ftpe7-1.fna.fbcdn.net/v/t1.0-1/p320x320/79498686_2761353167417628_1246618539746394112_o.jpg?_nc_cat=106&_nc_ohc=x3z4iSKGAwwAQkbehCuNPDegEk_Y0iRYHs2Y4V7_QWQ4RO5kCRNPOC55A&_nc_ht=scontent.ftpe7-1.fna&_nc_tp=1&oh=390d2c783d4cbef5e6bd6fb3b3787d82&oe=5E99A09A" width=80> | <img src="https://scontent.ftpe7-1.fna.fbcdn.net/v/t1.0-9/995456_408847005882967_842797001_n.jpg?_nc_cat=100&_nc_ohc=1imVOW7QilEAQkNRtC9TiMAJEkxOELjVATHLoYPgRBD2wOBi5TSGharzw&_nc_ht=scontent.ftpe7-1.fna&oh=72f7972b8339d12c728923f89091f3a9&oe=5EA82A2A" width=80>  | <img src="https://scontent.ftpe7-2.fna.fbcdn.net/v/t1.0-1/p320x320/76762604_477483546200564_68076086340091904_n.jpg?_nc_cat=104&_nc_ohc=ljf__qrYM5EAQmYdsWhQP7SU8_SsGGC_qJ7lIcCVV37yfj588biL06gZQ&_nc_ht=scontent.ftpe7-2.fna&_nc_tp=1&oh=b03d1d0c9ce7c95c0c3809d1c05d3255&oe=5EA1FF48" width=80> |
| <a href="https://github.com/dinotuku" target="_blank">`dinotuku`</a> | <a href="https://github.com/yehchunhung" target="_blank">`yehchunhung`</a> | <a href="https://github.com/hirokihayakawa07" target="_blank">`hirokihayakawa07`</a> | <a href="https://github.com/eternalbetty233" target="_blank">`eternalbetty233`</a> |
