import json
import statistics
import time
from absl import app, flags, logging
import pytorch_lightning as pl
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from textaugment import Wordnet, EDA
import transformers
import nltk
import torch as th
from datasets import Dataset, load_dataset, list_metrics, load_metric
from random import randint
from transformers import pipeline
import re

flags.DEFINE_integer('epochs', 6, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('lr', 1e-2, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('augmentation', False, '')
flags.DEFINE_string('concept', 'Anxiety', '' )
flags.DEFINE_boolean('classification_task', True, '')
flags.DEFINE_boolean('balance', True, '')

FLAGS = flags.FLAGS

class BertModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        if FLAGS.classification_task:
            self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
            self.loss = th.nn.CrossEntropyLoss(reduction="none")
        else:
            self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model, num_labels=5)
            self.loss = th.nn.MSELoss(reduction="none")

    def prepare_data(self):
        pass

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits = self.model(input_ids, mask).logits
        return logits

    def training_step(self, batch, batch_idx):
        if FLAGS.classification_task == False:
            batch['label'] = batch['label'].type(th.FloatTensor)
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        if FLAGS.classification_task == False:
            batch['label'] = batch['label'].type(th.FloatTensor)
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits,batch['label'])
        if FLAGS.classification_task:
            TP = sum(((logits.argmax(-1) == 1) & (batch['label'] == 1)).int())
            FP = sum(((logits.argmax(-1) == 1) & (batch['label'] == 0)).int())
            FN = sum(((logits.argmax(-1) == 0) & (batch['label'] == 1)).int())
            acc = (logits.argmax(-1) == batch['label']).float()
            prec = TP / (FP + TP)
            prec[th.isnan(prec)] = 0.
            rec = TP / (TP + FN)
            rec[th.isnan(rec)] = 0.
            f1 = (2 * prec * rec) / (prec + rec)
            f1[th.isnan(f1)] = 0.
        else:
            acc = 0.
            prec = 0.
            rec = 0.
            f1 = 0.
        return {'loss': loss, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

    def validation_epoch_end(self, outputs):
        print("outputs: ", outputs)
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        if FLAGS.classification_task:
            acc = th.cat([o['acc'] for o in outputs], 0).mean()
            prec = [o['prec'] for o in outputs]
            prec = th.mean(th.stack(prec))
            rec = [o['rec'] for o in outputs]
            rec = th.mean(th.stack(rec))
            f1 = [o['f1'] for o in outputs]
            f1 = th.mean(th.stack(f1))
        else:
            acc = 0.
            prec = 0.
            rec = 0.
            f1 = 0.
        #acc = [o['acc'] for o in outputs]

        self.out = {'val_loss': loss, 'val_acc': acc, 'val_prec': prec, 'val_rec': rec, 'val_f1': f1}
        print("out: ", self.out)
        return {**self.out, 'log': self.out}

    def train_dataloader(self):
        # take the training dataset and construct a dataloader from it
        return th.utils.data.DataLoader(
            self.ds_train,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.ds_test,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=False,
        )

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=FLAGS.lr,
            momentum=FLAGS.momentum,
        )



def _prepare_ds():
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    fillmask = pipeline('fill-mask', top_k=3)
    mask_token = fillmask.tokenizer.mask_token
    w = EDA()

    def _get_mask_pipeline(words):
        if len(words) > 3:
            K = randint(1, len(words) - 1)
            masked_sentence = " ".join(words[:K] + [mask_token] + words[K + 1:])
            predictions = fillmask(masked_sentence)
            augmented_sequences = [re.sub('<s>|</s>', '', predictions[i]['sequence']) for i in range(3)]
        else:
            augmented_sequences = []
        return augmented_sequences

    def _get_eda(sequence):
        augmented_sequences = []
        # synonym replacement
        new_item = w.synonym_replacement(sequence)
        augmented_sequences.append(new_item)
        # random insertion
        new_item = w.random_insertion(sequence)
        augmented_sequences.append(new_item)

        words = sequence.split(' ')
        if len(words) > 2:
            # random deletion
            new_item = w.random_deletion(sequence)
            augmented_sequences.append(new_item)
            # random swap
            new_item = w.random_swap(sequence)
            augmented_sequences.append(new_item)
        return augmented_sequences

    def _get_nlp_aug(sequence):
        augmented_sequences = []
        # character level
        new_item = nac.OcrAug().augment(sequence)
        augmented_sequences.append(new_item)
        new_item = nac.KeyboardAug().augment(sequence)
        augmented_sequences.append(new_item)
        new_item = nac.RandomCharAug(action="insert").augment(sequence)
        augmented_sequences.append(new_item)
        new_item = nac.RandomCharAug(action="delete").augment(sequence)
        augmented_sequences.append(new_item)
        # word level
        new_item = naw.SpellingAug().augment(sequence)
        augmented_sequences.append(new_item)
        #i = 0
        #while i < 5:
        #    new_item = naw.ContextualWordEmbsAug(
        #        model_path='bert-base-uncased', action="insert").augment(sequence)
        #    augmented_sequences.append(new_item)
        #    i += 1
        #i = 0
        #while i < 5:
        #    new_item = naw.ContextualWordEmbsAug(
        #        model_path='bert-base-uncased', action="substitute").augment(sequence)
        #    augmented_sequences.append(new_item)
        #    i += 1
        return augmented_sequences


    def _get_augmentation(split):
        augmented = {}
        augmented['item'] = []
        augmented['label'] = []
        counter = len(split)
        for item in split:
            sequence = re.sub('\\n', ' ', item['item'])
            label = item['label']
            words = sequence.split(' ')
            masked_sequences = _get_mask_pipeline(words)
            eda_sequences = _get_eda(sequence)
            nlpaug_sequences = _get_nlp_aug(sequence)
            sequence = list(set([sequence] + masked_sequences + eda_sequences + nlpaug_sequences))
            label = [label for i in range(len(sequence))]
            augmented['item'] += sequence
            augmented['label'] += label
            counter -= 1
            print("added ", len(sequence), " augmentations to item.", counter, "to go")
        return augmented

    def _get_more_control_samples(split):
        added_control_samples={}
        added_control_samples['item'] = []
        added_control_samples['label'] = []
        # count existing case and control samples
        count_case = 0
        count_control = 0
        for item in split:
            added_control_samples['item'].append(item['item'])
            added_control_samples['label'].append(item['label'])
            if item['label'] == 0:
                count_control += 1
            else:
                count_case += 1
        # compute missing control samples
        missing_control = count_case - count_control
        for i in range(missing_control):
            with open(r'/mount/studenten5/projects/kreuteae/BaselineModel/Control_Samples.json', encoding="utf8") as control_items:
                control_items = json.load(control_items)
                K = randint(0, len(control_items["data"]) - 1)
                control_item = control_items["data"][K]
                added_control_samples['item'].append(control_item['item'])
                added_control_samples['label'].append(0)

        print("Successfully added ", missing_control, " control samples.")
        return added_control_samples


    def _load_dataset():
        ds_test = load_dataset('json', data_files=r'/mount/studenten5/projects/kreuteae/BaselineModel/%s.json' %FLAGS.concept, field='data', split=[f'train[{k}%:{k+20}%]' for k in range(0, 100, 20)])
        ds_train = load_dataset('json', data_files=r'/mount/studenten5/projects/kreuteae/BaselineModel/%s.json' %FLAGS.concept, field='data', split=[f'train[:{k}%]+train[{k+20}%:]' for k in range(0, 100, 20)])
        list_of_ds_train = []
        list_of_ds_test = []
        for split in ds_train:
            if FLAGS.augmentation:
                split = Dataset.from_dict(_get_augmentation(split))
            if FLAGS.balance:
                split = Dataset.from_dict(_get_more_control_samples(split))
            split = split.map(_tokenize)
            split.set_format(type='torch', columns=['input_ids', 'label'])
            list_of_ds_train.append(split)
        for split in ds_test:
            if FLAGS.augmentation:
                split = Dataset.from_dict(_get_augmentation(split))
            if FLAGS.balance:
                split = Dataset.from_dict(_get_more_control_samples(split))
            split = split.map(_tokenize)
            split.set_format(type='torch', columns=['input_ids', 'label'])
            list_of_ds_test.append(split)
        return list_of_ds_train, list_of_ds_test

    def _tokenize(x):
        x['input_ids'] = tokenizer.encode(
            x['item'],
            max_length=32,
            pad_to_max_length=True
        )
        return x

    list_of_ds_train, list_of_ds_test = _load_dataset()
    return list_of_ds_train, list_of_ds_test

def main(_):
    start = time.time()
    list_of_ds_train, list_of_ds_test = _prepare_ds()
    performance = {}
    performance['mse'] = []
    if FLAGS.classification_task == True:
        performance['acc'] = []
        performance['prec'] = []
        performance['rec'] = []
        performance['f1'] = []

    for i in range(len(list_of_ds_train)):
        print("******************* Round No. ", i+1, " *******************")
        model = BertModel()
        model.ds_train = list_of_ds_train[i]
        print(model.ds_train)
        model.ds_test = list_of_ds_test[i]
        print(model.ds_test)
        trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(1 if th.cuda.is_available() else 0),
            max_epochs=FLAGS.epochs,
            fast_dev_run=FLAGS.debug,
            logger=pl.loggers.TensorBoardLogger('logs/', name=FLAGS.concept)
        )

        trainer.fit(model)
        mse = model.out['val_loss']
        performance['mse'].append(mse.item())
        if FLAGS.classification_task == True:
            acc = model.out['val_acc']
            prec = model.out['val_prec']
            rec = model.out['val_rec']
            f1 = model.out['val_f1']
            performance['acc'].append(acc.item())
            performance['prec'].append(prec.item())
            performance['rec'].append(rec.item())
            performance['f1'].append(f1.item())
        print("performance summary until now:", performance)

    for key, values in performance.items():
        print(key, ": ", statistics.mean(values))
    ende = time.time()
    print('{:5.3f}s'.format(ende - start))


if __name__ == '__main__':
    app.run(main)