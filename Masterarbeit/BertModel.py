import datetime
from absl import flags
import pytorch_lightning as pl
import transformers
import torch as th
#from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report


flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_float('lr', 1e-5, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')
flags.DEFINE_boolean('multiclass', False, '')

FLAGS = flags.FLAGS

class BertModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.batch_size = FLAGS.batch_size
        self.counter_epoch = 0
        self.conc = None
        self.tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)
        self.model_type = FLAGS.model
        self.multiclass = FLAGS.multiclass
        self.lr = FLAGS.lr
        if self.multiclass:
            self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model, num_labels=3)
            self.loss = th.nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
            self.loss = th.nn.CrossEntropyLoss(reduction="none")

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits = self.model(input_ids, mask).logits
        #for i in range(len(input_ids)):
        #    print(self.tokenizer.decode(input_ids[i]))
        #    print(logits[i].argmax(-1))
        #    print(logits[i])
        return logits

    def training_step(self, batch, batch_idx):
        print(self.lr)
        if self.multiclass:
            batch['label'] = batch['label'].type(th.FloatTensor).to(device = 'cuda')
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        #self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['label']
        if self.multiclass:
            batch['label'] = batch['label'].type(th.FloatTensor).to(device = 'cuda')
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits,batch['label'])
        preds = logits.argmax(-1)
        return {'loss': loss, 'pred': preds, 'target': y}

    def validation_epoch_end(self, outputs):
            loss = th.cat([o['loss'] for o in outputs], 0).mean()
            preds = th.cat([o['pred'] for o in outputs], 0)
            print(preds)
            targets = th.cat([o['target'] for o in outputs], 0)
            print(targets)
            if self.multiclass:
                targets = targets.argmax(-1)
            report = classification_report(targets.cpu(), preds.cpu())
            print(report)
            print(loss)
            with open('./Ergebnisse/%s/training/%s_epoch_no_%s.txt' % (
                    FLAGS.concept, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), self.counter_epoch),
                      'w') as outfile:
                outfile.write(str(report))
            report = classification_report(targets.cpu(), preds.cpu(), output_dict=True)
            self.counter_epoch += 1
            self.log = {'val_loss': loss}
            return self.log

    def test_step(self, batch, batch_idx):

        if self.multiclass:
            logits = self.forward(batch['input_ids'])
            pred_negative = sum(logits.argmax(-1) == 0)
            pred_neutral = sum(logits.argmax(-1) == 1)
            pred_positive = sum(logits.argmax(-1) == 2)
            return {'pred_negative': pred_negative, 'pred_neutral': pred_neutral, 'pred_positive': pred_positive, 'logits': logits}
        else:
            logits = self.forward(batch['input_ids'])
            pred = sum(logits.argmax(-1) == 1)
            print(pred)
            return {'logits': logits, 'pred': pred}

    def test_epoch_end(self, outputs):
        print(outputs)
        if self.multiclass:
            logits = [o['logits'] for o in outputs]
            conc = logits[0]
            for logit in logits[1:]:
                conc= th.cat((conc,logit), 0)
            self.conc = conc
            predictions = {}
            pred_negative = [o['pred_negative'] for o in outputs]
            print(pred_negative)
            pred_negative = th.sum(th.stack(pred_negative))
            print(pred_negative)
            predictions['negative'] = pred_negative
            pred_neutral = [o['pred_neutral'] for o in outputs]
            pred_neutral = th.sum(th.stack(pred_neutral))
            predictions['neutral'] = pred_neutral
            pred_positive = [o['pred_positive'] for o in outputs]
            pred_positive = th.sum(th.stack(pred_positive))
            predictions['positive'] = pred_positive
            highest_prediction = max(predictions, key=predictions.get)
            self.log = {'test_pred': highest_prediction}
            return self.log
        else:
            logits = [o['logits'] for o in outputs]
            pred = [o['pred'] for o in outputs]
            conc = logits[0]
            for logit in logits[1:]:
                conc= th.cat((conc,logit), 0)
            pred = th.sum(th.stack(pred))            
            pred = pred / len(self.ds_test)
            self.log = {'test_pred': pred}
            self.conc = conc
            return self.log



    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return th.utils.data.DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=4
        )

    def configure_optimizers(self):
        print(self.lr)
        return th.optim.Adam(
            self.parameters(),
            lr=(self.lr or self.learning_rate)
        )