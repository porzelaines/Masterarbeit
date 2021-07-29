import transformers
import torch

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import json
import re
from sklearn.model_selection import train_test_split

def t5(sentence):
  text =  "paraphrase: " + sentence + " </s>"

  max_len = 85

  encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

  # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
  beam_outputs = model.generate(
      input_ids=input_ids, attention_mask=attention_masks,
      do_sample=True,
      max_length=85,
      top_k=120,
      top_p=0.98,
      early_stopping=True,
      num_return_sequences=50
  )
  print ("\nOriginal Question ::")
  print (sentence)
  print ("\n")
  print ("Paraphrased Questions :: ")
  final_outputs =[]
  for beam_output in beam_outputs:
      sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
      if sent.lower() != sentence.lower() and sent not in final_outputs:
          final_outputs.append(sent)

  for i, final_output in enumerate(final_outputs):
      print("{}: {}".format(i, final_output))

  return final_outputs

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


dataset = open(r'./MyDatasets/Neuroticism.json', encoding='utf8')
dataset = json.load(dataset)
X = []
y = []
z = []
counter = len(dataset['data'])
for item in dataset['data']:
    X.append(item['item'])
    y.append(item['label'])
    z.append(item['source'])

augmentation = {}
augmentation['data'] = []
for i in range(len(X)):
    sequence = re.sub('\\n', ' ', X[i])
    print(sequence)
    label = y[i]
    source = z[i]
    words = sequence.split(' ')
    sequences = t5(sequence)
    sequence = list(set([sequence] + sequences))
    print(sequence)
    for item in sequences:
        item_dict = {}
        item_dict['item'] = item
        item_dict['label'] = label
        item_dict['source'] = source
        augmentation['data'].append(item_dict)
    counter -= 1

with open(r'./MyDatasets/Neuroticism_T5.json', 'w') as json_file:
    json.dump(augmentation, json_file)




