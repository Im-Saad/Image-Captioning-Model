import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import sys
from IPython.display import display


MAX_LENGTH = 45
EMBEDDING_DIM = 512
UNITS = 512

with open("vocab", 'rb') as f:
    vocab = pickle.load(f)

tokenizer = tf.keras.layers.TextVectorization(
    standardize=None,
    output_sequence_length=MAX_LENGTH,
    vocabulary=vocab
)

idx2word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True
)

def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet'
    )
    output = inception_v3.output
    output = tf.keras.layers.Reshape(
        (-1, output.shape[-1]))(output)

    cnn_model = tf.keras.models.Model(inception_v3.input, output)
    return cnn_model

class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, x, training):
        x = self.layer_norm_1(x)
        x = self.dense(x)
        attn_output = self.attention(
            query=x, value=x, key=x, training=training)
        return x + attn_output

class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(max_len, embed_dim)

    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        return token_embeddings + position_embeddings

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, units, num_heads):
        super().__init__()
        self.embedding = Embeddings(tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)
        self.attention_1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.attention_2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)
        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")

    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)
        attn_output_1 = self.attention_1(query=embeddings, value=embeddings, key=embeddings, attention_mask=mask, training=training)
        out_1 = self.layernorm_1(embeddings + attn_output_1)
        attn_output_2 = self.attention_2(query=out_1, value=encoder_output, key=encoder_output, training=training)
        out_2 = self.layernorm_2(out_1 + attn_output_2)
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)
        preds = self.out(ffn_out)
        return preds

class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_model, encoder, decoder):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder

def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

def generate_caption(img, caption_model):
    if isinstance(img, str):
        img = load_image_from_path(img)
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)
    y_inp = '[start]'
    
    for i in range(MAX_LENGTH-1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(tokenized, img_encoded, training=False, mask=mask)
        pred_idx = np.argmax(pred[0, i, :])
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')
        if pred_word == '[end]':
            break
        y_inp += ' ' + pred_word
    
    return y_inp.replace('[start] ', '')

def get_caption_model():
    cnn_model = CNN_Encoder()
    encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
    decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)
    caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
    caption_model.load_weights('model_weights.h5')
    return caption_model

def main(image_path):
    caption_model = get_caption_model()
    caption = generate_caption(image_path, caption_model)
    print(f"Generated Caption: {caption}")
    img = Image.open(image_path)
    img = img.resize((250, 250))
    display(img)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
    else:
        main(sys.argv[1])
