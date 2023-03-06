# Model Trainer for [gl-activation-map-viewer](https://github.com/joshmurr/gl-activation-map-viewer)

This is a fork of *Alantian's* [ganshowcase](https://github.com/alantian/ganshowcase). I pinched their original trained DCGAN model and so adapted the code to train a few more models under the same architecture. I probably wouldn't have used Chainer otherwise tbh..

This repo differs in a number of ways:

- Addition of 128x128 and 256x256 DCGAN models.
- Adapted *datatool.py* with a few more arguments.
- TODO: finish this list..

## What is in This Repo

### Step 1 - Prepare data

Dataset is stored as an [npz](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) file, and can be converted from either a folder containing images or the CelebAHQ dataset.

For using the folder of images, use

```bash
DIR_PATH=...
DATA_FILE=...
SIZE=... # can be 64, 128 or 256
./datatool.py --task dir_to_npz \
  --dir_path $DIR_PATH --npz_path $DATA_FILE --size $SIZE
```

For using the CelebAHQ dataset which can be obtained by consulting [its GitHub repo](https://github.com/tkarras/progressive_growing_of_gans):

```bash
CELEBAHQ_PATH=...  # should be an h5 file
DATA_FILE=...
SIZE=... # can be 64, 128 or 256

../scripts/run_docker.sh \
./datatool.py --task multisize_h5_to_npz \
  --multisize_h5_path $CELEBAHQ_PATH  --npz_path $DATA_FILE --size $SIZE
```

### Step 2 - Training the model

```bash
# Training DCGAN64 model
DATA_FILE_SIZE_64=...  # Data file of size 64
DCGAN64_OUT=... # Output directory
./chainer_dcgan.py \
  --arch dcgan64 \
  --image_size 64 \
  --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.999 --lambda_gp 1.0 --learning_rate_anneal 0.9 --learning_rate_anneal_trigger 0 --learning_rate_anneal_interval 5000 --max_iter 100000 --snapshot_interval 5000 --evaluation_sample_interval 100 --display_interval 10 \
  --npz_path $DATA_FILE_SIZE_64 \
  --out $DCGAN64_OUT \
  ;


# Training ReSNet128 model
DATA_FILE_SIZE_128=...  # Data file of size 64
RESNET128_OUT=... # Output directory
./chainer_dcgan.py \
  --arch dcgan64 \
  --image_size 64 \
  --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.999 --lambda_gp 1.0 --learning_rate_anneal 0.9 --learning_rate_anneal_trigger 0 --learning_rate_anneal_interval 5000 --max_iter 100000 --snapshot_interval 5000 --evaluation_sample_interval 100 --display_interval 10 \
  --npz_path $DATA_FILE_SIZE_128 \
  --out $RESNET128_OUT \
  ;


# Training ReSNet256 model
DATA_FILE_SIZE_256=...  # Data file of size 64
RESNET256_OUT=... # Output directory
./chainer_dcgan.py \
  --arch dcgan64 \
  --image_size 64 \
  --adam_alpha 0.0001 --adam_beta1 0.5 --adam_beta2 0.999 --lambda_gp 1.0 --learning_rate_anneal 0.9 --learning_rate_anneal_trigger 0 --learning_rate_anneal_interval 5000 --max_iter 100000 --snapshot_interval 5000 --evaluation_sample_interval 100 --display_interval 10 \
  --npz_path $DATA_FILE_SIZE_256 \
  --out $RESNET256_OUT \
  ;

```

### Step 3 - Convert from Chainer model to Keras/Tensorflow.js model

Note that due to difficulty in training GANs,
you may want to select a proper snapshot by specifying `ITER` below.
This script also samples a few images serving as a sanity check and providing clue for picking the correct snapshot.

```bash
# DCGAN64

ITER=50000
./dcgan_chainer_to_keras.py \
  --arch dcgan64 \
  --chainer_model_path $DCGAN64_OUT/SmoothedGenerator_${ITER}.npz \
  --keras_model_path $DCGAN64_OUT/Keras_SmoothedGenerator_${ITER}.h5 \
  --tfjs_model_path $DCGAN64_OUT/tfjs_SmoothedGenerator_${ITER} \
  ;

# ResNet128

ITER=20000
./dcgan_chainer_to_keras.py \
  --arch resnet128 \
  --chainer_model_path $RESNET128_OUT/SmoothedGenerator_${ITER}.npz \
  --keras_model_path $RESNET128_OUT/Keras_SmoothedGenerator_${ITER}.npz.h5 \
  --tfjs_model_path $RESNET128_OUT/tfjs_SmoothedGenerator_${ITER} \
  ;

# ResNet256

ITER=45000
./dcgan_chainer_to_keras.py \
  --arch resnet256 \
  --chainer_model_path $RESNET256_OUT/SmoothedGenerator_${ITER}.npz \
  --keras_model_path $RESNET256_OUT/Keras_SmoothedGenerator_${ITER}.npz.h5 \
  --tfjs_model_path $RESNET256_OUT/tfjs_SmoothedGenerator_${ITER} \
  ;

```

### Step 4 - Present the generative as a web page

This step is covered by a web project under `./webcode/ganshowcase`.
Now it is assumeed that you are in `./webcode/ganshowcase` directory.

First you need to copy TensorFlow.js model (specified as argument to `--tfjs_model_path` in previous step) to a public accessible place, and
modify `model_url` in `all_model_info` which is in the beginning of `index.js`.

Then run the following:

```
yarn
yarn build
```

Finally, copy `./dist/`, which is the built web page and js file,
to whatever suitable for web hosting.

As an example, [deploy.sh](./deploy.sh) does the compilation and put everthing to [docs](./docs),
since the GitHub Pages site is currently being built from the `/docs` folder in the master branch.
