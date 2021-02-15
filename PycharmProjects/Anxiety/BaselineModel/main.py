import statistics
import time
from absl import app, flags
import pytorch_lightning as pl
import torch as th
from BertModel import BertModel
from TrainsetLoader import _load_dataset
from TestsetLoader import load_users

flags.DEFINE_float('threshold', 0.4, '')
flags.DEFINE_boolean('test', True, '')
flags.DEFINE_integer('epochs', 3, '')
flags.DEFINE_string('concept', 'Depression', '')

FLAGS = flags.FLAGS

def main(_):
    def test_on_user_profiles(model):
        user_profiles = load_users(FLAGS.concept)
        print("Now testing model on ", len(user_profiles), " user profiles")
        TP = 0
        FP = 0
        FN = 0
        list_of_pred = []
        counter = 0
        for user in user_profiles:
            print("k fold cross: ", i + 1, "/10")
            print("user no. ", counter + 1, " of ", len(user_profiles), "user profiles.")
            print("user labeled with: ", user["label"][0].item())
            counter += 1
            model.ds_test = user
            trainer.test()
            prediction_of_user = model.out['test_pred'].item()
            print("model labeled ", prediction_of_user, "% as ", FLAGS.concept)
            list_of_pred.append(prediction_of_user)
            if prediction_of_user > FLAGS.threshold and user["label"][0] == 1:
                TP += 1
            elif prediction_of_user > FLAGS.threshold and user["label"][0] == 0:
                FP += 1
            elif prediction_of_user < FLAGS.threshold and user["label"][0] == 1:
                FN += 1
        if TP > 0:
            prec = TP / (TP + FP)
            rec = TP / (TP + FN)
            f1 = (2 * prec * rec) / (prec + rec)
        else:
            prec = rec = f1 = 0
        return prec, rec, f1, list_of_pred

    start = time.time()
    list_of_ds_train, list_of_ds_val = _load_dataset(FLAGS.concept)
    ergebnisse = {}
    ergebnisse["prec"] = []
    ergebnisse["rec"] = []
    ergebnisse["f1"] = []
    list_of_pred = {}
    for i in range(len(list_of_ds_train)):
        print("******* Train Model No. ", i + 1, " *******")
        model = BertModel()
        model.ds_train = list_of_ds_train[i]
        model.ds_val = list_of_ds_val[i]
        trainer = pl.Trainer(
            default_root_dir='logs',
            gpus=(1 if th.cuda.is_available() else 0),
            max_epochs=FLAGS.epochs,
            logger=pl.loggers.TensorBoardLogger('logs/', name=FLAGS.concept)
        )
        trainer.fit(model)
        print("Performance during training: ", model.out)
        if FLAGS.test == True:
            prec, rec, f1, list_of_pred_i = test_on_user_profiles(model)
            ergebnisse["prec"].append(prec)
            ergebnisse["rec"].append(rec)
            ergebnisse["f1"].append(f1)
            list_of_pred[i] = list_of_pred_i
        else:
            ergebnisse["prec"].append(model.out["val_prec"].item())
            ergebnisse["rec"].append(model.out["val_rec"].item())
            ergebnisse["f1"].append(model.out["val_f1"].item())
    print("Ergebnisse der ", FLAGS.concept, " models (auf den user profiles):", ergebnisse)
    print(list_of_pred)

    for key, values in ergebnisse.items():
        print(key, ": ", statistics.mean(values))
        
    ende = time.time()
    print('{:5.3f}s'.format(ende - start))



if __name__ == '__main__':
    app.run(main)