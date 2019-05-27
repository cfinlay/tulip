## Scaleable input gradient regularization for CIFAR-10

This folder contains training code for CIFAR-10 models, trained with squared
gradient norm regularization.
The main training script is `train.py`;  the strength of the regularizer is controlled with the `--penalty` flag.

An example training bash script is provided, see `example.sh`.

For comparison, multi-step adversarial training (with hyper-parameters from Madry et al [1]) has been implemenetd in `adversarial-training.py`.

The script `test_statistics.py` is used for computing model statistics on the
test set, including lower bounds on the minimum adversarial distance, on a per
image basis. Statistics are written to a .pkl file.

To download one of the models used in the paper, use the script `fetch_model.py`. To see available models, run `python fetch_model.py -h`.

#### References
[1]   Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." [arXiv preprint arXiv:1706.06083](https://arxiv.org/abs/1706.06083) (2017).
