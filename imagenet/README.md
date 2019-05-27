## Scaleable input gradient regularization for ImageNet-1k

This folder contains training code for ImageNet-1k models, trained with squared
L2 gradient norm regularization (Tikhonov regularization).

The code is modified from [ImageNet in 18 minutes](https://github.com/diux-dev/imagenet18). The code trains on smaller images for the first 15 epochs or so. To get these smaller images, use the following
```
wget https://s3.amazonaws.com/yaroslavvb/imagenet-data-sorted.tar
wget https://s3.amazonaws.com/yaroslavvb/imagenet-sz.tar

tar -xvf imagenet-data-sorted.tar -C /path/to/data/
tar -xvf imagenet-sz.tar -C /path/to/data

cd /path/to/data
mv raw-data imagenet
```

An example training script is provided, `example.sh`. Regularization strength
is controlled by the `--tikhonov` flag, which is set to 0.1 in the script.

The script `test_statistics.py` is used for computing model statistics on the
test set, including lower bounds on the minimum adversarial distance, on a per
image basis. Statistics are written to a .pkl file.

To download one of the models used in the paper, use the script `fetch_model.py`. To see available models, run `python fetch_model.py -h`.
