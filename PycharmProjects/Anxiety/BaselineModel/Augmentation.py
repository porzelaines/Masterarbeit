import json
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from textaugment import EDA
from random import randint
from transformers import pipeline
import re
from absl import flags

flags.DEFINE_string('augmentation_type', 'eda', '')
FLAGS = flags.FLAGS

def get_mask_pipeline(words):
    fillmask = pipeline('fill-mask', top_k=3)
    mask_token = fillmask.tokenizer.mask_token
    if len(words) > 3:
        K = randint(1, len(words) - 1)
        masked_sentence = " ".join(words[:K] + [mask_token] + words[K + 1:])
        predictions = fillmask(masked_sentence)
        augmented_sequences = [re.sub('<s>|</s>', '', predictions[i]['sequence']) for i in range(3)]
    else:
        augmented_sequences = []
    return augmented_sequences

def get_eda(sequence):
    w = EDA()
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

def get_nlp_aug(sequence):
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
    return augmented_sequences


def get_augmentation(split):
    augmented = {}
    augmented['item'] = []
    augmented['label'] = []
    counter = len(split)
    for item in split:
        sequence = re.sub('\\n', ' ', item['item'])
        print(sequence)
        label = item['label']
        words = sequence.split(' ')
        if FLAGS.augmentation_type == 'fill_mask':
            sequences = get_mask_pipeline(words)
        if FLAGS.augmentation_type == 'eda':
            sequences = get_eda(sequence)
        if FLAGS.augmentation_type == 'nlp_aug':
            sequences = get_nlp_aug(sequence)
        sequence = list(set([sequence] + sequences))
        print(sequence)
        label = [label for i in range(len(sequence))]
        augmented['item'] += sequence
        augmented['label'] += label
        counter -= 1
        print("added ", len(sequence), " augmentations to item.", counter, "to go")
    return augmented


