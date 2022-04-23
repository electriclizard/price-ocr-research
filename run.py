import os
import yaml
import shutil


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

params = yaml.safe_load(open("params.yaml"))

# remove aocr log
try:
    os.remove("aocr.log")
    shutil.rmtree(f"{params['model']['exported_model']}", ignore_errors=True)
    shutil.rmtree(f"{params['model']['model_dir']}", ignore_errors=True)
except OSError:
    pass

os.system(
    f"aocr train {params['dataset']['train_filepath']} \
    --model-dir {params['model']['model_dir']} \
    --steps-per-checkpoint {params['model']['steps-per-checkpoint']} \
    --initial-learning-rate {params['train']['learning_rate']} \
    --batch-size {params['train']['batch_size']} \
    --num-epoch {params['train']['epochs']} \
    "
)

os.system(
    f"aocr test \
    --visualize {params['dataset']['test_filepath']}"
)

# export model
os.system(
    f"aocr export {params['model']['exported_model']}"
)

# export frozen graph
os.system(
    f"aocr export --format=frozengraph {params['model']['exported_model']}"
)