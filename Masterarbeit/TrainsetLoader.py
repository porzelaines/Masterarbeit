from absl import flags
import transformers
from datasets import Dataset, load_dataset
import json
import random


flags.DEFINE_boolean('augmentation', False, '')
flags.DEFINE_string('augmentation_type', 'EDA', '')
flags.DEFINE_boolean('k_fold', False, '')
FLAGS = flags.FLAGS

def _load_dataset(concept, model):
    def _tokenize(x):
        x['input_ids'] = tokenizer.encode(
            x['item'],
            max_length=85,
            pad_to_max_length=True
        )
        return x

    tokenizer = transformers.BertTokenizer.from_pretrained(model)
    list_of_ds_train = []
    list_of_ds_val = []
    if FLAGS.k_fold == True:
        ds_val = load_dataset('json', data_files=r'./Trainsets/%s.json' %concept, field='data', split=[f'train[{k}%:{k + 10}%]' for k in range(0, 100, 10)])
        ds_train = load_dataset('json', data_files=r'./Trainsets/%s.json' %concept, field='data', split=[f'train[:{k}%]+train[{k + 10}%:]' for k in range(0, 100, 10)])
        for split in ds_train:
            split = split.map(_tokenize)
            split.set_format(type='torch', columns=['input_ids', 'label'])
            list_of_ds_train.append(split)
        for split in ds_val:
            split = split.map(_tokenize)
            split.set_format(type='torch', columns=['input_ids', 'label'])
            list_of_ds_val.append(split)
        return list_of_ds_train, list_of_ds_val
    else:
        if FLAGS.augmentation == True:
            if FLAGS.augmentation_type == 'gpt2':
                ds_val = load_dataset('json', data_files=r'./Augmentation/%s/Train/Original/%s_%s.json' %(FLAGS.augmentation_type, concept, FLAGS.augmentation_type), field='data', split=f'train[:10%]')
                ds_train = load_dataset('json', data_files=r'./Augmentation/%s/Train/Original/%s_%s.json' %(FLAGS.augmentation_type, concept, FLAGS.augmentation_type), field='data', split=f'train[-90%:]')
            else:
                ds_val = load_dataset('json', data_files=r'./Augmentation/%s/Val/Original/%s_%s.json' %(FLAGS.augmentation_type, concept, FLAGS.augmentation_type), field='data', split=f'train')
                ds_train = load_dataset('json', data_files=r'./Augmentation/%s/Train/Original/%s_%s.json' %(FLAGS.augmentation_type, concept, FLAGS.augmentation_type), field='data', split=f'train')
        else:
            ds_val = load_dataset('json', data_files=r'./Trainsets/Original/%s.json' %concept, field='data', split=f'train[:10%]')
            ds_train = load_dataset('json', data_files=r'./Trainsets/Original/%s.json' %concept, field='data', split=f'train[-90%:]')

        ds_val = ds_val.map(_tokenize)
        ds_val.set_format(type='torch', columns=['input_ids', 'label'])

        ds_train = ds_train.map(_tokenize)
        ds_train.set_format(type='torch', columns=['input_ids', 'label'])

        return ds_train, ds_val