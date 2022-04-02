import json
import yaml

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.integrations import TensorBoardCallback
from transformers import default_data_collator
from datasets import load_metric

from dataset import IAMDataset


def string_accuracy(st1: str, st2: str):
    if st1 == st2:
        return True
    else:
        return False


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    st = {"cer": cer, "string_accuracy": (np.array(pred_str) == np.array(label_str)).mean()}
    with open('metrics.json', 'a') as file:
        file.write(f"{st}")
    return st


params = yaml.safe_load(open("params.yaml"))


processor = TrOCRProcessor.from_pretrained(params['model']['model_name'])
model = VisionEncoderDecoderModel.from_pretrained(params['model']['model_name'])

train_df = pd.read_csv(params['dataset']['train_filepath'], sep=' ', header=None)
train_df = train_df[:int(train_df.shape[0] * float(params['dataset']['train_size']))]
train_df[0] = train_df[0].apply(lambda x: "/".join(x.split('/')[-1:]))
train_df[1] = train_df[1].apply(lambda x: str(x))

test_df = pd.read_csv(params['dataset']['test_filepath'], sep=' ', header=None)
test_df[0] = test_df[0].apply(lambda x: "/".join(x.split('/')[-1:]))
test_df[1] = test_df[1].apply(lambda x: str(x))

train_dataset = IAMDataset(root_dir=params['dataset']['images_path'],
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir=params['dataset']['images_path'],
                           df=test_df,
                           processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))


# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = int(params['model']['max_length'])
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    save_strategy='epoch',
    logging_strategy='epoch',
    per_device_train_batch_size=int(params['train']['batch_size']),
    per_device_eval_batch_size=int(params['train']['batch_size']),
    fp16=True, 
    output_dir="./",
    logging_steps=2,
    num_train_epochs=float(params['train']['epochs']),
#     save_steps=10000,
#     eval_steps=5000,
    save_total_limit=2,
    dataloader_num_workers=4,
    learning_rate=float(params['train']['learning_rate']),
)

cer_metric = load_metric("cer")
st_acc = load_metric('accuracy')

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
    callbacks=[TensorBoardCallback]
)

# trainer.train("./checkpoint-20000")
trainer.train()

trainer.evaluate()
model.save_pretrained(params['train']['save_name'])
