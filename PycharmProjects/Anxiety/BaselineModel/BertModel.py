from absl import flags
import pytorch_lightning as pl
import transformers
import torch as th


flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_float('lr', 5e-3, '')
flags.DEFINE_float('momentum', .9, '')
flags.DEFINE_string('model', 'bert-base-uncased', '')

FLAGS = flags.FLAGS

class BertModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = th.nn.CrossEntropyLoss(reduction="none")

    def prepare_data(self):
        pass

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits = self.model(input_ids, mask).logits
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits,batch['label'])
        TP = sum(((logits.argmax(-1) == 1) & (batch['label'] == 1)).int())
        FP = sum(((logits.argmax(-1) == 1) & (batch['label'] == 0)).int())
        FN = sum(((logits.argmax(-1) == 0) & (batch['label'] == 1)).int())
        acc = (logits.argmax(-1) == batch['label']).float()
        prec = TP / (FP + TP)
        prec[th.isnan(prec)] = 0.
        rec = TP / (TP + FN)
        rec[th.isnan(rec)] = 0.
        pred = sum(logits.argmax(-1) == 1)
        f1 = (2 * prec * rec) / (prec + rec)
        f1[th.isnan(f1)] = 0.
        return {'loss': loss, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'pred': pred}

    def validation_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        prec = [o['prec'] for o in outputs]
        prec = th.mean(th.stack(prec))
        rec = [o['rec'] for o in outputs]
        rec = th.mean(th.stack(rec))
        f1 = [o['f1'] for o in outputs]
        f1 = th.mean(th.stack(f1))
        pred = [o['pred'] for o in outputs]
        pred = th.sum(th.stack(pred))
        pred = pred / len(self.ds_val)
        self.out = {'val_loss': loss, 'val_acc': acc, 'val_prec': prec, 'val_rec': rec, 'val_f1': f1, 'val_pred': pred}
        print("out: ", self.out)
        return {**self.out, 'log': self.out}

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        TP = sum(((logits.argmax(-1) == 1) & (batch['label'] == 1)).int())
        FP = sum(((logits.argmax(-1) == 1) & (batch['label'] == 0)).int())
        FN = sum(((logits.argmax(-1) == 0) & (batch['label'] == 1)).int())
        acc = (logits.argmax(-1) == batch['label']).float()
        prec = TP / (FP + TP)
        prec[th.isnan(prec)] = 0.
        rec = TP / (TP + FN)
        rec[th.isnan(rec)] = 0.
        pred = sum(logits.argmax(-1) == 1)
        f1 = (2 * prec * rec) / (prec + rec)
        f1[th.isnan(f1)] = 0.
        return {'loss': loss, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'pred': pred}

    def test_epoch_end(self, outputs):
        loss = th.cat([o['loss'] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        prec = [o['prec'] for o in outputs]
        prec = th.mean(th.stack(prec))
        rec = [o['rec'] for o in outputs]
        rec = th.mean(th.stack(rec))
        f1 = [o['f1'] for o in outputs]
        f1 = th.mean(th.stack(f1))
        pred = [o['pred'] for o in outputs]
        pred = th.sum(th.stack(pred))
        pred = pred / len(self.ds_test)
        self.out = {'test_loss': loss, 'test_acc': acc, 'test_prec': prec, 'test_rec': rec, 'test_f1': f1, 'test_pred': pred}
        print("out: ", self.out)

    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.ds_train,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.ds_val,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=True,
        )

    def test_dataloader(self):
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