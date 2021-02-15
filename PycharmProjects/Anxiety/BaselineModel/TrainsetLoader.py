from absl import flags
import transformers
from datasets import Dataset, load_dataset
from Augmentation import get_augmentation

flags.DEFINE_boolean('augmentation', True, '')
FLAGS = flags.FLAGS

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

def _tokenize(x):
    x['input_ids'] = tokenizer.encode(
        x['item'],
        max_length=32,
        pad_to_max_length=True
    )
    return x


def _load_dataset(concept):
    ds_val = load_dataset('json', data_files=r'C:\Users\Anne\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\%s.json' %concept, field='data', split=[f'train[{k}%:{k + 10}%]' for k in range(0, 100, 10)])
    ds_train = load_dataset('json', data_files=r'C:\Users\Anne\AppData\Roaming\JetBrains\PyCharmCE2020.2\scratches\%s.json' %concept, field='data', split=[f'train[:{k}%]+train[{k + 10}%:]' for k in range(0, 100, 10)])
    list_of_ds_train = []
    list_of_ds_val = []
    for split in ds_train:
        if FLAGS.augmentation:
            split = Dataset.from_dict(get_augmentation(split))
        split = split.map(_tokenize)
        split.set_format(type='torch', columns=['input_ids', 'label'])
        list_of_ds_train.append(split)
    for split in ds_val:
        split = split.map(_tokenize)
        split.set_format(type='torch', columns=['input_ids', 'label'])
        list_of_ds_val.append(split)

    return list_of_ds_train, list_of_ds_val