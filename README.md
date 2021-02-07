# 🕵️‍♀️ ganx

Let's try and implement a wgan-gp using jax and haiku.

⚠️ _This repo is still just an experiment, it probably shouldn't be used for anything yet_ 

## 💾  Install
```
$ pip install git+https://github.com/rosshemsley/ganx
```


## 🚅  Train
```
$ python -m ganx.cli.train --root-dir </path/to/celeba/img_align_celeba/>
```

Or, from a virtualenv with the package installed,
```
$ train --root-dir </path/to/celeba/img_align_celeba/>
```

During training, logs are written to a tensorflow events file at `runs/`. You may visualize progress using tensorboard.
