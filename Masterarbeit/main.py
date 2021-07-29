import time
from absl import app, flags
import pytorch_lightning as pl
import torch as th
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from BertModel import BertModel
from TrainsetLoader import _load_dataset
from TestsetLoader import load_users
from datasets import concatenate_datasets
from sklearn.metrics import classification_report

flags.DEFINE_integer('epochs', 200, '')
flags.DEFINE_string('concept', 'Depression', '')
flags.DEFINE_string('model_name', 'Depression', '')
flags.DEFINE_boolean('debug', False, '')
flags.DEFINE_boolean('ahypo', False, '')
flags.DEFINE_boolean('kfold', False, '')
flags.DEFINE_boolean('multi_class', False, '')

FLAGS = flags.FLAGS

def main(_):
    seed_everything(42)
    start = time.time()
    model = BertModel()
    model_type = model.model_type
    if FLAGS.kfold == True:
       gesamt_performance = ''
       list_of_ds_train, list_of_ds_val = _load_dataset(FLAGS.concept, model_type)
       for i in range(len(list_of_ds_train)):
           print("******* Train Model No. ", i + 1, " *******")
           model.ds_train = list_of_ds_train[i]
           model.concept = FLAGS.concept
           model.ds_val = list_of_ds_val[i]
           early_stop_callback = EarlyStopping(
               monitor='val_loss',
               min_delta=0.0001,
               patience=5,
               verbose=False,
               mode='min'
               )
           trainer = pl.Trainer(
               default_root_dir='logs',
               gpus=(4 if th.cuda.is_available() else 0),
               max_epochs=FLAGS.epochs,
               logger=pl.loggers.TensorBoardLogger('logs/', name=FLAGS.concept),
               callbacks=[early_stop_callback],
               accumulate_grad_batches=2,
               deterministic=True,
               fast_dev_run=FLAGS.debug,
               gradient_clip_val=1.0,
               auto_select_gpus=(True if th.cuda.is_available() else False),
               accelerator='dp',
               auto_lr_find=False
               )
           trainer.tune(model)
           trainer.fit(model)
           gesamt_performance += " " + str(model.log)
           print("Performance after training ", i,": ", model.log)
       print(gesamt_performance)



   
    list_of_users = load_users(FLAGS.concept, model_type)
    total_len = len(list_of_users)
    print(total_len)
    train_len = int(0.5 * total_len)
    test_len = int(0.5 * total_len)

    ds_train = list_of_users[:train_len]
    ds_test = list_of_users[-test_len:]
    print(ds_test)

    if FLAGS.ahypo == True:
        merge = ds_train[0]
        for user in ds_train[1:]:
            merge = concatenate_datasets([merge, user])
        merge = merge.shuffle(seed=42)
        merge = merge.train_test_split(test_size=0.1)
        print(merge['train']['label'][:50])
        print(merge['test']['label'][:50])
        model.ds_train = merge['train']
        model.ds_val = merge['test']
        model.concept = FLAGS.concept

    else:
        ds_train, ds_val = _load_dataset(FLAGS.concept, model_type)
        model.ds_train = ds_train
        model.concept = FLAGS.concept
        model.ds_val = ds_val

    checkpoint_callback = ModelCheckpoint(
            dirpath='./models/%s/' %FLAGS.concept,
            filename='%s_%s' % (FLAGS.model_name,FLAGS.ahypo),
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
        )
    early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0001,
            patience=5,
            verbose=False,
            mode='min'
        )
    trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(1 if th.cuda.is_available() else 0),
            max_epochs=FLAGS.epochs,
            logger=pl.loggers.TensorBoardLogger('logs/', name=FLAGS.concept),
            callbacks=[early_stop_callback,checkpoint_callback],
            accumulate_grad_batches=2,
            deterministic=True,
            fast_dev_run=FLAGS.debug,
            gradient_clip_val=1.0,
            auto_select_gpus=(True if th.cuda.is_available() else False),
            accelerator='dp',
            auto_lr_find=False
            )
    trainer.tune(model)
    trainer.fit(model)

    print("Performance after training: ", model.log)
    ende = time.time()
    print('{:5.3f}s'.format(ende - start))



    preds = []
    targets = []
    counter = 0
    if FLAGS.multi_class:
        for user in ds_test:
            print("user no. ", counter + 1, "/", len(ds_test))
            counter += 1            
            y_true = user["label"][0]
            if y_true == 0:
                y_true = 'No %s' %FLAGS.concept
            elif y_true == 1:
                y_true = '%s' %FLAGS.concept
            model.ds_test = user
            trainer.test(model)
            targets.append(y_true)
            print("y_true: ", y_true)
            y_pred = model.log['test_pred'].item()
            print("y_pred: ",y_pred)
            if y_pred > 0.5:
                y_pred = '%s' %FLAGS.concept
            else:
                y_pred = 'No %s' %FLAGS.concept
            preds.append(y_pred)
            print("y_pred: ", y_pred)
    else:
        for user in ds_test:
            print(FLAGS.concept)
            print("user no. ", counter + 1, "/", len(ds_test))
            counter += 1
            print(user)
            model.ds_test = user
            trainer.test(model)
            y_pred = model.log['test_pred'].item()
            print("y_pred: ",y_pred)
            if y_pred > 0.5:
                y_pred = '%s' %FLAGS.concept
            else:
                y_pred = 'No %s' %FLAGS.concept
            y_true = user["label"][0].item()
            if y_true == 1:
                y_true = '%s' %FLAGS.concept
            else:
                y_true = 'No %s' %FLAGS.concept
            print("y_true: ", y_true)
            preds.append(y_pred)
            targets.append(y_true)

    ergebnis_of_model = classification_report(targets, preds)
    print(ergebnis_of_model)

if __name__ == '__main__':
    app.run(main)