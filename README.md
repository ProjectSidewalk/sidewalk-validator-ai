# sidewalk-validator-ai

This is the code for training and testing validator models for Project Sidewalk. The models are based on the dinov2 architecture.

## Setup
First, please setup the conda environment:

```bash
conda env create -f environment.yml
```

Now, download the depth anything checkpoint that we use for crop extraction:

```bash
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth
```

## ❗An important note
Please don't forget to activate your conda environment before running any slurm scripts or python files!

## Dataset Generation
Before we can train any validator models, we need to create a dataset. To do this for all five label types (Crosswalk, CurbRamp, NoCurbRamp, Obstacle, SurfaceProblem), simply run:
```bash
sbatch data_generation.slurm
```
Once this slurm job starts, you should see the creation of a folder for each label type. It might take some time for the scripts to initialize, though, so don't be alarmed if it doesn't appear instantly. You can check the logs of your slurm job by looking in the log file that was generated immediately after the slurm job started.

Once the job is finished (which will take many hours, possibly even days), each label type folder should have a subfolder `ds` (which stands for dataset). This is the dataset we will use for the next step, training!

## Model Training

Now that we have the datasets for each label type, we want to train a validator model for each label type.

To do this for all label types, simply run:

```bash
sbatch fine_tune.slurm
```

This will take a long time. By the end of it, each label type folder should have a `dinov2` folder with model checkpoints for both the best model and the latest model.

## Model Evaluation

To evaluate model performance for all label types on the `test` split, simply run:
```bash
sbatch get_stats.slurm
```

After the job is finished, you'll have some files generated in each label type folder:

- `confidence_precision_map.json`
    - This maps model confidence to actual precision.
- `pvr.png`
    - The precision-recall curve for the model (Remember, each label type gets its own binary classification model).
- `report.png`
    - Basically the precision recall curve, but with some additional stats included in text.

## Model Deployment

To use the newly trained models in production, upload the contents of the dinov2 folder into a new Hugging Face model. Example [here](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-crosswalk/tree/main). In the future, it might be helpful to have a script to automatically upload to Hugging Face instead of manually doing it.