#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Number of Unique Words: 29076
#Number of words in single caption: 49

import pandas as pd
import re
import json
import collections
import random
import pickle
import tensorflow as tf


def get_captions(path):
    with open(f'{path}\\annotations\\captions_train2017.json','r') as f:
        data = json.load(f)
        data = data['annotations']
    
    img_caption_pairs = []
    
    for x in data:
        img = '%012d.jpg' % x['image_id']
        img_caption_pairs.append([img, x['caption']])
        
    df = pd.DataFrame(img_caption_pairs, columns = ['image', 'caption'])
    df['image'] = df['image'].apply(lambda x: f'{path}\\train2017\\{x}')
    df.reset_index(drop= True)
    
    return df
    
    
    
def preprocess_captions(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = '[start] ' + text + ' [end]'
    
    return text



def tokenize(df):
    max_length = 50
    tokenizer = tf.keras.layers.TextVectorization(
        standardize=None,
        output_sequence_length=max_length
    )
    
    tokenizer.adapt(df['caption'])
    
    vocab = tokenizer.get_vocabulary()
    pickle.dump(vocab, open('vocab', 'wb'))
    
    word_to_idx = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocab
    )
    
    idx_to_word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocab,
        invert=True
    )
    
    return tokenizer, word_to_idx, idx_to_word



def split_image_captions(df, test_size):

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(df['image'], df['caption']):
        img_to_cap_vector[img].append(cap)

    img_keys = list(img_to_cap_vector.keys())
    
    random.shuffle(img_keys)

    slice_index = int(len(img_keys) * (1 - test_size))
    img_name_train_keys = img_keys[:slice_index]
    img_name_val_keys = img_keys[slice_index:]

    train_imgs = []
    train_captions = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        train_imgs.extend([imgt] * capt_len)
        train_captions.extend(img_to_cap_vector[imgt])

    val_imgs = []
    val_captions = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        val_imgs.extend([imgv] * capv_len)
        val_captions.extend(img_to_cap_vector[imgv])

    return train_imgs, train_captions, val_imgs, val_captions



def load_data(img_path, caption, tokenizer):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    caption = tokenizer(caption)
    return img, caption



def create_datasets(train_imgs, train_captions, val_imgs, val_captions, load_data, tokenizer, buffer_size, batch_size):

    train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_captions))
    train_dataset = train_dataset.map(lambda img, caption: load_data(img, caption, tokenizer), num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_imgs, val_captions))
    val_dataset = val_dataset.map(lambda img, caption: load_data(img, caption, tokenizer), num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size)
    
    return train_dataset, val_dataset

