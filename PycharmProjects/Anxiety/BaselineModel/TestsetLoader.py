from datasets import Dataset
import json
import os
import random
import transformers
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('test_control_users', True, '')
flags.DEFINE_boolean('test_case_users', True, '')
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

def _tokenize(x):
    x['input_ids'] = tokenizer.encode(
        x['item'],
        max_length=32,
        pad_to_max_length=True
    )
    return x

def load_users(concept):
    print("*****Laden der Social Media Profiles*****")
    user_profiles = []
    if FLAGS.test_case_users == True:
        user_profiles += load_case_users(concept)
    if FLAGS.test_control_users == True:
        user_profiles += load_control_users()
    random.shuffle(user_profiles)
    return user_profiles

def load_case_users(concept):
    user_profiles = []
    for subdir, dirs, files in os.walk(r'C:\Users\Anne\Documents\Masterarbeit\Datasets\Twitter_API\%s' %concept):
        for file in files:
            profile_dict = {}
            profile_dict["item"] = []
            profile_dict["label"] = []
            path_profile = os.path.join(subdir, file)
            print(path_profile)
            profile = open(path_profile, encoding='utf8')
            profile = json.load(profile)
            for k, v in profile.items():
                profile_dict["item"].append(v['0'])
                profile_dict["label"].append(1)
            ds_test = Dataset.from_dict(profile_dict)
            ds_test = ds_test.map(_tokenize)
            ds_test.set_format(type='torch', columns=['input_ids', 'label'])
            user_profiles.append(ds_test)
    return user_profiles

def load_control_users():
    user_profiles = []
    for subdir, dirs, files in os.walk(r'C:\Users\Anne\Documents\Masterarbeit\Datasets\Twitter_API\Control'):
        for file in files:
            profile_dict = {}
            profile_dict["item"] = []
            profile_dict["label"] = []
            path_profile = os.path.join(subdir, file)
            print(path_profile)
            profile = open(path_profile, encoding='utf8')
            profile = json.load(profile)
            for k, v in profile.items():
                profile_dict["item"].append(v['0'])
                profile_dict["label"].append(0)
            ds_test = Dataset.from_dict(profile_dict)
            ds_test = ds_test.map(_tokenize)
            ds_test.set_format(type='torch', columns=['input_ids', 'label'])
            user_profiles.append(ds_test)
    return user_profiles




