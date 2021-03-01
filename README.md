# Implicit Image Fitting

## Getting Started

<details><summary><b>Install</b></summary>
<p>

* `python3.8`
* `pytorch`: 1.7.0+ (GPU support preferable).

Then,

* `make install`
</p>
</details>

<details><summary><b>W&B API Key</b></summary>
<p>

Copy your WandB API key to `wandb_api.key`.
Will be used to login to your dashboard for visualisation. 
Alternatively, you can skip W&B visualisation, 
and set `wandb.use=False` while running the python code or `USE_WANDB=False` while running make commands.
</p>
</details>

<details><summary><b>Google Drive Links</b></summary>
<p>

* [Project Folder](https://drive.google.com/open?id=1sDWa0notYql5KZZfG4wkbxMwzwDgxgdS&authuser=vsundar4%40wisc.edu&usp=drive_fs)
* [Image Dataset](https://drive.google.com/open?id=1sjXxggKV2Yn2KC7LCknJwGJIRHdRMcXv&authuser=vsundar4%40wisc.edu&usp=drive_fs): we'll be mainly using 16-bit images from `img/rgb16bit`. 
These are all sourced from the [image compression benchmark](https://imagecompression.info).
* [Research Papers](https://drive.google.com/open?id=1SPozBvSU1w---OK0j0Ltr-tV_c_N6zkk&authuser=vsundar4%40wisc.edu&usp=drive_fs): saved as `<name of paper>[<conference> <year> <author>].pdf`. 
Look out for top-tier conference papers (CVPR, ECCV, ICCV, NeurIPS, ICLR) and journals (TPAMI).
* [Output folder](https://drive.google.com/open?id=1MaVgu-Tu9vIPq6c9vEaPfCZC_8kPHL3W&authuser=vsundar4%40wisc.edu&usp=drive_fs): will contain logs of important experiments and their config files.

</p>
</details>

<details><summary><b>Using Colab</b></summary>
<p>

We can't run our codebase directly on colab, since hydra relies on config injection.
Instead, we'll make use of [colabcode](https://github.com/abhishekkrthakur/colabcode/blob/master/colab_starter.ipynb). 
Just click the button below, and setup your port and password.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/varun19299/implicit-image-compression/blob/main/colab_starter.ipynb)

This should open up a familiar VSCode environment, use the terminal to run.

Steps:

* Change to your drive home folder.
File -> Open -> `/content/drive/MyDrive/`.

* Clone repository (only the first time):
`git clone https://github.com/varun19299/implicit-image-compression.git`.

* Copy the `img/` folder from the shared drive here.

* Install all dependencies with: `make colab_install` 

**Please do not use the shared folder as the location for your code, it will cause conflicts.**

</p>
</details>

<details><summary><b>Recommended Workflow</b></summary>
<p>


* Use meaningful experiment names, via `exp_name`. 
Hydra allows you to use other config values in any command line variable.

Eg: `python main.py exp_name='siren-width-${mlp.width}-depth-${mlp.depth}' mlp.width=256,512 mlp.depth=6,8`.

This will run 4 experiments (cartesian product of {256,512} x {6,8}), 
with experiment names as siren-width-256-depth-6, siren-width-256-depth-8, etc. 

* Please create a new project on [W&B](https://wandb.ai/implicit-image/), and change `wandb.project` accordingly.

Eg: `python main.py wandb.project=siren-width-depth`.  

* Copy important output folders (see under `outputs/`) to `Drive/code/outputs/`. 

* W&B is pretty flexible when it comes to plotting, so you should be able to compare methods on the dashboard itself.
Use their [API](https://docs.wandb.ai/library/public-api-guide) in case you need to do some post-processing before making plots.
</p>
</details>

<details><summary><b>Rsync Outputs</b></summary>
<p>

</p>
</details>


## View all configs

`python main.py --cfg job`
``

We use hydra for configs. YAML files present under `conf/`.

## Main Script

`make fit`

## MakeFile Help

`make help`, will display commented usage for each command.
