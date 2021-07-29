from sklearn.metrics import classification_report
from pytorch_lightning import seed_everything
from TestsetLoader import load_users
from BertModel import BertModel
from absl import app, flags
import pytorch_lightning as pl
import datetime
import json
import torch as th

flags.DEFINE_string('concept', 'Depression', '')
flags.DEFINE_string('test_art', 'all', '')
flags.DEFINE_float('threshold', 0.5, '')
flags.DEFINE_string('model_name', 'Depression', '')
flags.DEFINE_boolean('multi_class', False, '')

FLAGS = flags.FLAGS


def main(_):
    seed_everything(42)
    trainer = pl.Trainer(
        gpus=(4 if th.cuda.is_available() else 0),
        accelerator='dp',
        auto_select_gpus=(True if th.cuda.is_available() else False)
    )

    counter = 0
    if not FLAGS.multi_class:
        model_test = BertModel.load_from_checkpoint('./models/%s/%s.ckpt' % (FLAGS.concept, FLAGS.model_name))
        user_profiles = load_users(FLAGS.concept, model_test.model_type)
        print("Now testing model on ", len(user_profiles), " user profiles")
        preds = []
        targets = []
        for user in user_profiles:
            print(FLAGS.concept)
            print("user no. ", counter + 1, "/", len(user_profiles))
            counter += 1
            print(user)
            model_test.ds_test = user
            trainer.test(model_test)
            y_pred = model_test.log['test_pred'].item()
            print("y_pred: ",y_pred)
            if y_pred > FLAGS.threshold:
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

        target_names = ['%s' %FLAGS.concept, 'No %s' %FLAGS.concept]
        ergebnis_of_model = classification_report(targets, preds, target_names=target_names)
        print(ergebnis_of_model)

        if FLAGS.test_art == "case":
            with open('./Ergebnisse/%s/case/%s_model.json' % (
                    FLAGS.concept, FLAGS.model_name), 'w') as outfile:
                json.dump(ergebnis_of_model, outfile)
        if FLAGS.test_art == "control":
            with open('./Ergebnisse/%s/control/%s_model.json' % (
                    FLAGS.concept, FLAGS.model_name), 'w') as outfile:
                json.dump(ergebnis_of_model, outfile)
        if FLAGS.test_art == "all":
            with open('./Ergebnisse/%s/all/%s_model.json' % (
                    FLAGS.concept, FLAGS.model_name), 'w') as outfile:
                json.dump(ergebnis_of_model, outfile)

    else:
        model_test = BertModel.load_from_checkpoint('./models/%s/%s.ckpt' % (FLAGS.concept, FLAGS.model_name), num_labels=3)
        user_profiles = load_users(FLAGS.concept, model_test.model_type)
        print("Now testing model on ", len(user_profiles), " user profiles")
        preds = []
        targets = []
        for user in user_profiles:
            print(FLAGS.concept)
            print("user no. ", counter + 1, "/", len(user_profiles))
            counter += 1            
            y_true = user["label"][0].numpy()
            if y_true.argmax(-1) == 0:
                y_true = 'negative'
            elif y_true.argmax(-1) == 1:
                continue
            elif y_true.argmax(-1) == 2:
                y_true = 'positive'                
            targets.append(y_true)
            print("y_true: ", y_true)
            model_test.ds_test = user
            trainer.test(model_test)
            y_pred = model_test.log['test_pred']
            preds.append(y_pred)
            print("y_pred: ", y_pred)

        ergebnis_of_model = classification_report(targets, preds)
        print(ergebnis_of_model)

        with open('./Ergebnisse/%s/all/%s_model.txt' % (
                FLAGS.concept, FLAGS.model_name), 'w') as outfile:
            json.dump(ergebnis_of_model, outfile)



if __name__ == '__main__':
    app.run(main)