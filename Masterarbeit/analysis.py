from pytorch_lightning import seed_everything
from BertModel import BertModel
from absl import app, flags
import pytorch_lightning as pl
import torch as th
import transformers
import lime
import numpy as np
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
from datasets import Dataset
import json
import os


flags.DEFINE_string('concept', 'Depression', '')
flags.DEFINE_string('model_name', 'Depression', '')
flags.DEFINE_boolean('multi_class', False, '')
flags.DEFINE_boolean('test_control_users', True, '')
flags.DEFINE_boolean('test_case_users', True, '')

FLAGS = flags.FLAGS

def main(_):
    def _tokenize(x):
        x['input_ids'] = tokenizer.encode(
            x['tweet'],
            max_length=85,
            pad_to_max_length=True
        )
        return x

    def predictor(texts):
        text_dict = {}
        text_dict["tweet"] = []
        for text in texts:
            text_dict["tweet"].append(text)
        ds_test = Dataset.from_dict(text_dict)
        ds_test = ds_test.map(_tokenize)
        ds_test.set_format(type='torch', columns=['input_ids'])
        model.ds_test = ds_test
        trainer.test(model)    
        outputs = model.conc
        probas = F.softmax(outputs.cpu()).detach().numpy()
        return probas

    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    seed_everything(42)
    trainer = pl.Trainer(
        deterministic=True,
        gpus=(4 if th.cuda.is_available() else 0),
        accelerator='dp',
        auto_select_gpus=(True if th.cuda.is_available() else False)
    )


    if not FLAGS.multi_class:
        model = BertModel.load_from_checkpoint('./models/%s/%s.ckpt' % (FLAGS.concept, FLAGS.model_name))
        if FLAGS.test_case_users == True:
            for subdir, dirs, files in os.walk(r'./Test_Testsets/%s' %FLAGS.concept):
                for file in files:
                    ergebnis_tweets = []
                    path_profile = os.path.join(subdir, file)
                    print(path_profile)
                    profile = open(path_profile, encoding='utf8')
                    profile = json.load(profile)
                    counter = 0
                    for k, v in profile.items():
                        counter += 1
                        ergebnis_tweet = {}
                        str_to_predict = v['0']
                        explainer = LimeTextExplainer(class_names=["No %s" %FLAGS.concept, "%s" %FLAGS.concept])
                        exp = explainer.explain_instance(str_to_predict, predictor, num_features=10)
                        print("********************", str_to_predict, "************************")
                        print(exp.as_list())
                        ergebnis_tweet["tweet"] = str_to_predict
                        ergebnis_tweet["label"] = str(model.log['test_pred'])
                        ergebnis_tweet["counter"] = counter
                        ergebnis_tweets.append(ergebnis_tweet)
                        exp.save_to_file('./Ergebnisse/%s/analysis/%s_model_%s_ahypo.html' %(FLAGS.concept, FLAGS.model_name, counter))                        
                    with open('./Ergebnisse/%s/analysis/%s_model_ahypo.txt' % (
                        FLAGS.concept, FLAGS.model_name), 'w') as outfile:
                        json.dump(ergebnis_tweets, outfile)

        if FLAGS.test_control_users == True:
            for subdir, dirs, files in os.walk(r'./Test_Testsets/Control'):
                for file in files:
                    ergebnis_tweets = []
                    path_profile = os.path.join(subdir, file)
                    print(path_profile)
                    profile = open(path_profile, encoding='utf8')
                    profile = json.load(profile)
                    counter = 0
                    for k, v in profile.items():
                        counter += 1
                        ergebnis_tweet = {}
                        str_to_predict = v['0']
                        explainer = LimeTextExplainer(class_names=["No %s" %FLAGS.concept, "%s" %FLAGS.concept])
                        exp = explainer.explain_instance(str_to_predict, predictor, num_features=10)
                        print("********************", str_to_predict, "************************")
                        print(exp.as_list())
                        ergebnis_tweet["tweet"] = str_to_predict
                        ergebnis_tweet["label"] = str(model.log['test_pred'])
                        ergebnis_tweet["counter"] = counter
                        ergebnis_tweets.append(ergebnis_tweet)
                        exp.save_to_file('./Ergebnisse/%s/analysis/%s_model_control_%s_ahypo.html' %(FLAGS.concept, FLAGS.model_name, counter))             
                    with open('./Ergebnisse/%s/analysis/%s_model_control_ahypo.txt' % (
                        FLAGS.concept, FLAGS.model_name), 'w') as outfile:
                        json.dump(ergebnis_tweets, outfile)

    else:
        model = BertModel.load_from_checkpoint('./models/%s/%s.ckpt' %(FLAGS.concept, FLAGS.model_name), num_labels=3)
        for subdir, dirs, files in os.walk(r'./Test_Testsets/Personality'):
            for file in files:
                ergebnis_tweets = []
                path_profile = os.path.join(subdir, file)
                print(path_profile)
                profile = open(path_profile, encoding='utf8')
                profile = json.load(profile)
                counter = 0
                for v in profile['item']:
                    counter += 1
                    ergebnis_tweet = {}
                    str_to_predict = v
                    explainer = LimeTextExplainer(class_names=["Negative","Neutral","Positive"])
                    exp = explainer.explain_instance(str_to_predict, predictor, num_features=10, top_labels =3)
                    print("********************", str_to_predict, "************************")
                    print(exp.as_list())
                    ergebnis_tweet["tweet"] = str_to_predict
                    ergebnis_tweet["label"] = str(model.log['test_pred'])
                    ergebnis_tweet["counter"] = counter
                    ergebnis_tweets.append(ergebnis_tweet)
                    exp.save_to_file('./Ergebnisse/%s/analysis/%s_model_%s_ahypo.html' %(FLAGS.concept, FLAGS.model_name, counter))             
                with open('./Ergebnisse/%s/analysis/%s_model_ahypo.txt' % (
                     FLAGS.concept, FLAGS.model_name), 'w') as outfile:
                     json.dump(ergebnis_tweets, outfile)

if __name__ == '__main__':
    app.run(main)