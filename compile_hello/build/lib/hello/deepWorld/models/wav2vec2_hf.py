from transformers import AutoFeatureExtractor,AutoModelForAudioClassification,TrainingArguments, Trainer
from transformers.trainer import Trainer
from torch import tensor
import evaluate
import numpy as np

BASE_TRAINING_ARGUMENTS=\
{    
    "output_dir":"./models_ckpt",
    "logging_dir":"./models_log",
    "overwrite_output_dir":True,
    "evaluation_strategy":"steps",
    "eval_steps":10,
    "save_strategy":"steps",
    "save_step":10,
    "learning_rate":3e-5,
    "weight_decay":0.00001,
    "num_train_epochs":10,
    "per_device_train_batch_size":32,
    "per_device_eval_batch_size":32,
    "logging_strategy":"steps",
    "logging_steps":10,
    "save_total_limit":2
}



def get_wav2vec_model(num_labels,label2id,id2label):

    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label)

    return model

def compute_metrics(eval_preds):
    metric = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits.detach().numpy(), axis=-1)
    return metric.compute(predictions=predictions, references=labels)

feature_extractor=AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

def preprocess_function(examples):

    sig,label=examples
    inputs = feature_extractor(
        sig[0], sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
    )
    inputs["input_values"]=tensor(inputs["input_values"][0])
    inputs["label"]=label
    return inputs
