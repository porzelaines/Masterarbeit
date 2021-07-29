from datasets import Dataset
import json
import os
import random
import transformers
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean('test_control_users', True, '')
flags.DEFINE_boolean('test_case_users', True, '')


def load_users(concept, model):
    def _tokenize(x):
        x['input_ids'] = tokenizer.encode(
            x['item'],
            max_length=85,
            pad_to_max_length=True
        )
        return x

    tokenizer = transformers.BertTokenizer.from_pretrained(model)
    print("*****Laden der Social Media Profiles*****")
    user_profiles = []
    if concept == 'Openness' or concept == 'Conscientiousness' or concept == 'Extraversion' or concept == 'Agreeableness' or concept == 'Neuroticism':
        user_profiles = []
        for subdir, dirs, files in os.walk(r'./Testsets/Personality/%s' % concept):
            for file in files:
                path_profile = os.path.join(subdir, file)
                print(path_profile)
                profile_dict = open(path_profile, encoding='utf8')
                profile_dict = json.load(profile_dict)
                if profile_dict["label"][0] != [0,1,0]:          
                    for i in range(len(profile_dict["label"])):
                        if profile_dict["label"][i] == [1,0,0]:
                            profile_dict["label"][i] = 0
                        else:
                            profile_dict["label"][i] = 1           
                    ds_test = Dataset.from_dict(profile_dict)
                    ds_test = ds_test.map(_tokenize)
                    ds_test.set_format(type='torch', columns=['input_ids', 'label'])
                    user_profiles.append(ds_test)
                else:
                    continue
    else:
        if FLAGS.test_case_users == True:
            user_profiles += load_case_users(concept,model)
        if FLAGS.test_control_users == True:
            user_profiles += load_control_users(model)
    random.shuffle(user_profiles)
    return user_profiles

def load_case_users(concept,model):
    def _tokenize(x):
        x['input_ids'] = tokenizer.encode(
            x['tweet'],
            max_length=85,
            pad_to_max_length=True
        )
        return x
    tokenizer = transformers.BertTokenizer.from_pretrained(model)
    counter=0
    user_profiles = []
    for subdir, dirs, files in os.walk(r'./Testsets/%s' %concept):
        for file in files:
            counter += 1
            print(counter)
            profile_dict = {}
            profile_dict["tweet"] = []
            profile_dict["label"] = []
            path_profile = os.path.join(subdir, file)
            print(path_profile)
            profile = open(path_profile, encoding='utf8')
            profile = json.load(profile)
            for k, v in profile.items():
                profile_dict["tweet"].append(v['0'])
                profile_dict["label"].append(1)
            ds_test = Dataset.from_dict(profile_dict)
            ds_test = ds_test.map(_tokenize)
            ds_test.set_format(type='torch', columns=['input_ids', 'label'])
            user_profiles.append(ds_test)
    return user_profiles

def load_control_users(model):
    def _tokenize(x):
        x['input_ids'] = tokenizer.encode(
            x['tweet'],
            max_length=85,
            pad_to_max_length=True
        )
        return x
    tokenizer = transformers.BertTokenizer.from_pretrained(model)
    counter = 0
    user_profiles = []
    for subdir, dirs, files in os.walk(r'./Testsets/Control'):
        for file in files:
            counter += 1
            print(counter)
            profile_dict = {}
            profile_dict["tweet"] = []
            profile_dict["label"] = []
            path_profile = os.path.join(subdir, file)
            print(path_profile)
            profile = open(path_profile, encoding='utf8')
            profile = json.load(profile)
            for k, v in profile.items():
                profile_dict["tweet"].append(v['0'])
                profile_dict["label"].append(0)
            ds_test = Dataset.from_dict(profile_dict)
            ds_test = ds_test.map(_tokenize)
            ds_test.set_format(type='torch', columns=['input_ids', 'label'])
            user_profiles.append(ds_test)
    return user_profiles




