# Keras Practice

Note: This is some practices of Keras


## 0. Config Environment

### 0.1 Create Virtualenv

cd into the dir of keras_practice

```
$ cd PATH/keras_practice/
```

Create virtualenv

```
$ virtualenv .env
```

### 0.2 Installing Libraries

Activate virtualenv and install some necessary libraries by pip

```
$ source .env/bin/activate
$ pip install -r requirements.txt
```

Wait about 20min for libraries installing, and you can deactivate the virtualenv if you like.

```
$ deactivate
```

### 0.3 Download dataset

cd to kps/dataset/, and run the download scrpt

```
$ cd dlwp/dataset/
$ ./get_pima_indians_diabetes_data.sh
``` 

the dataset will be downloaded to this dir

### 0.4 Start

Now, you can open the jupyter notebook by

```
$ ./start_ipython_notebook.sh
```

or 

```
$ ./env/bin/ipython notebook
```
