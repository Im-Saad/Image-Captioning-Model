import tensorflow as tf
import dataset_setup  
import model_setup  
import pickle

vocab_size = 29076
embedding = 512
max_length = 50
batch_size = 16
buffer_size = 1000
epochs = 10
ff_dim = 2048

df = dataset_setup.get_captions('data')
df = df.sample(frac=1)  
df['caption'] = df['caption'].apply(dataset_setup.preprocess_captions)

tokenizer, word2idx, idx2word = dataset_setup.tokenize(df)

with open('vocab.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

train_imgs, train_captions, val_imgs, val_captions = dataset_setup.split_image_captions(df, test_size=0.3)

train_dataset, val_dataset = dataset_setup.create_datasets(
    train_imgs, train_captions, val_imgs, val_captions,
    dataset_setup.load_data, tokenizer, buffer_size=buffer_size, batch_size=batch_size
)

cnn_model = model_setup.CNN_Encoder()

encoder = model_setup.TransformerEncoderLayer(embed_dim=embedding, num_heads=8, ff_dim=ff_dim)
decoder = model_setup.TransformerDecoderLayer(embed_dim=embedding, units=ff_dim, num_heads=8, tokenizer=tokenizer, max_length=max_length)

image_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.3),
])

caption_model = model_setup.ImageCaptioningModel(
    cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation
)

cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")

caption_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=cross_entropy
)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = caption_model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

caption_model.save('image_captioning_model.h5')
