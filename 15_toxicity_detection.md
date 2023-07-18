# Text Classification with `Transformer` Models 

Return to the [castle](https://github.com/Nkluge-correa/teeny-tiny_castle).

Text classification is the process of categorizing text into predefined classes or categories based on its content. One common classification task in NLP is toxicity detection.

This is a common application of text classification in natural language processing, where the objective is to identify whether a piece of text contains language that is offensive, harmful, or threatening. To accomplish this, machine learning models are trained on large datasets of labeled toxic and non-toxic text examples and then used to predict the toxicity of new, unlabeled text.

We can define toxicity as:

> "_abusive speech targeting specific group characteristics, such as ethnic origin, religion, gender, or sexual orientation._"

For an in-depth discussion on toxicity detection as a machine learning problem, we recommend "_[Learning from the worst: Dynamically generated datasets to improve online hate detection](https://scholar.google.com/scholar_url?url=https://arxiv.org/abs/2012.15761&hl=pt-BR&sa=T&oi=gsb&ct=res&cd=0&d=7265559494033067667&ei=QUJYY6TJL4iKmgHXk5LgDQ&scisig=AAGBfm3gsyOD5eqcUPLFvWmVm8PlLcMr3g)_".

<img src="https://miro.medium.com/max/1400/1*d4k-PRw-warACDpklCh1mw.png" alt="toxic-image" width="800"/>

In this notebook, we will be using a dataset created from the [Toxic Comment Classification Challenge Dataset](https://github.com/tianqwang/Toxic-Comment-Classification-Challenge), created by the [Conversation AI](https://conversationai.github.io/) team, a research initiative founded by [Jigsaw](https://jigsaw.google.com/) and Google (both a part of Alphabet).

## ☣️ DISCLAIMER/WARNING ☣️ 

_This dataset contains text that may be considered profane, vulgar, or offensive. The vocabulary of the trained model also contains this type of language._

The original dataset contains an unequal distribution of “_hate_” and “_not hate_” samples for multi-classification. However, we created a smaller version of the original dataset (the `toxic_content_dataset.csv`, available for download in [this link](https://drive.google.com/uc?export=download&id=1ZvZtrsE1dAl7CiHt16Jstp2rhkDKnGlL)). The used dataset contains an equal amount of “_hate_” and “_not hate_” samples, summing up to $70157$ samples total.


```python
import pandas as pd

df = pd.read_csv("data/toxic_content_dataset.csv")

display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comment_text</th>
      <th>toxic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>explanation edits made username hardcore metal...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aww matches background colour seemingly stuck ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hey man really trying edit war guy constantly ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>make real suggestions improvement wondered sec...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>sir hero chance remember page</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70152</th>
      <td>lol gay never know good feels fuck woman ass</td>
      <td>0</td>
    </tr>
    <tr>
      <th>70153</th>
      <td>fuck pansy jew would whine nai brith beat pale...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>70154</th>
      <td>shalom semite get fuck kill son bitch leave wi...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>70155</th>
      <td>think gay fag</td>
      <td>0</td>
    </tr>
    <tr>
      <th>70156</th>
      <td>previous conversation fucking shit eating libe...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>70157 rows × 2 columns</p>
</div>


The original dataset had a lot of symbols and emojis that may hinder the performance of a language model trained for text classification. However, we already performed the cleaning procedure. Below you can find the funciton that we used to function to `preprocess` text data, removing unwanted characters and so on.

```python

import re
from unidecode import unidecode

def custom_standardization(input_data):
    clean_text = input_data.lower().replace("<br />", " ")
    clean_text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", ' ', clean_text)
    clean_text = re.sub(' +', ' ', clean_text)
    return unidecode(clean_text)

```

Text is one of the most widespread forms of sequence data and discrete signals (as opposed to continuous signals, like _images_ or _audio_). These sequences can be sequences of characters, syllables, or words.

Deep learning for NLP is pattern recognition applied to paragraphs, sentences, and words, just as computer vision is pattern recognition applied to videos, images, and pixels.

Like all neural networks, language models based on deep-learning architectures don’t take as input raw text, i.e., you _can not multiply a word by a weight matrix, add a bias, and apply a ReLU function at the end_. Neural networks only work with numeric tensors. Thus, we need to _vectorize_ our text data, i.e., transform the text into numeric tensors.

For a comprehensive guide on how to vectorize text data, we recommend Chapter 6: Deep learning for text and sequences, in [_Deep Learning with Python_](https://tanthiamhuat.files.wordpress.com/2018/03/deeplearningwithpython.pdf). Below we will be using the `TextVectorization` layer from the [Keras](https://keras.io/) library. 

In terms of preprocessing, you can also pass symbols you may want to filter, by using the `filters` argument.

Finally, we will split our dataset for training/validation/testing.


```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# number of words in our vocabulary
vocab_size = 20000

# sentences lengthier than 100 will be truncated
sequence_length = 100

vectorization_layer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
    )

vectorization_layer.adapt(df.comment_text)
vocabulary = vectorization_layer.get_vocabulary()


# Here we save our vocabulary for later use
# DISCLAIMER - THIS VOCABULARY CONTAINS TOXIC LANGUAGE
with open(r'models/toxic_vocabulary.txt', 'w', encoding='utf-8') as fp:
    for word in vocabulary:
        fp.write("%s\n" % word)
    fp.close()

# `train_test_split` is a grate way to slip datasets
x_train, x_test, y_train, y_test = train_test_split(
    df.comment_text, df.toxic, test_size=0.2, random_state=42)

# vectorize senteces and turn label into `floats`
x_train = vectorization_layer(x_train)
y_train = np.array(y_train).astype(float)
x_test = vectorization_layer(x_test)
y_test = np.array(y_test).astype(float)
```

[Recurrent neural networks](https://en.wikipedia.org/wiki/Recurrent_neural_networks "Recurrent neural networks"), like `LSTM` and `GRU`, and convolutional networks, like `1D Convnets`, are great options for dealing with problems involving NLP. In [other notebooks](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Explainability/NLP%20Interpreter/model_maker.ipynb) of our repository, you will see many examples of how to build these networks for tasks like sentiment analysis.

However, in this notebook, we will be using a `transformer` model, an extremely versatile and scalable architecture proposed by Vaswani et al. in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

<img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" alt="drawing" height="450"/>

A `transformer` is a  deep learning model that adopts the mechanism of [self-attention](https://en.wikipedia.org/wiki/Attention_(machine_learning)), differentially weighting the significance of each part of the input data. Like RNNs, transformers are designed to process sequential input data. However, unlike RNNs, transformers process the entire input all at once (_not sequencially_). The transformer does not have to process one word at a time. This allows for more parallelization, thus reducing training times, and also allowing the training on larger datasets.

For an extremely _comprehensive_ and _ilustrated_ explanation of what is "_attention_" or how a "_transformer works_", we recommend the work of _Jay Alammar_:

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/).
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/).

Using only the _decoder_ component of the original transformer architecture, we will implement a small transformer model in this notebook (a.k.a. a decoder-only transformer). The original transformer architecture, which consists of both an encoder and a decoder transformer block, was (originally) designed for _sequence-to-sequence_ tasks like translation.

If you are interested in learning about the transformer architecture, check our _[sequence-to-sequence](https://github.com/Nkluge-correa/teeny-tiny_castle/blob/fa17764aa8800c388d0d298b750c686757e0861e/ML%20Intro%20Course/16_seuqnece_to_sequence.ipynb)_ tutorial.

In general, text classification can be done using the encoder component. It's a very generic module that learns to transform a sequence into a more useful representation after ingesting it.

This model only has 4 attention heads with a capacity of 100 tokens and 4 transformer blocks. Our embedding layer's size is also restricted to embeddings with 128 dimensions and a vocabulary of 20,000 tokens (where the dense word vectors will be created).


```python
from keras import layers
from tensorflow import keras

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Apply a Transformer Encoder block to the input tensor.

    Arguments:
        - inputs : tensor of shape (batch_size, sequence_length, hidden_size) 
            representing the input sequence.
        - head_size : integer representing the dimensionality of the keys 
            and values in each attention head.
        - num_heads : integer representing the number of attention heads.
        - ff_dim : integer representing the number of filters in the feedforward layer.
        - dropout : float between 0 and 1 representing the dropout rate.

    Returns:
        - tensor of shape (batch_size, sequence_length, hidden_size) representing 
            the output of the Transformer Encoder block.
    """
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0):

    vocab_size = 20000
    sequence_length = 100
    embed_size = 128

    inputs = tf.keras.Input(shape=input_shape, dtype="int32")
    x = tf.keras.layers.Embedding(input_dim=vocab_size,
                              output_dim=embed_size,
                              input_length=sequence_length)(inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


model = build_model(
    (x_train.shape[1]),
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.2,
)

model.compile(loss=tf.losses.BinaryCrossentropy(),
              optimizer='adam',
              metrics=['accuracy'])

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
model.summary()

callbacks = [keras.callbacks.ModelCheckpoint("models/toxicity_model.keras",
                                                save_best_only=True),
            keras.callbacks.EarlyStopping(monitor="val_loss",
                                            patience=5,
                                            verbose=1,
                                            mode="auto",
                                            baseline=None,
                                            restore_best_weights=True)]
model.fit(x_train,
          y_train,
          validation_split = 0.2,
          epochs=20,
          batch_size=16,
          verbose=1,
          callbacks=callbacks)

test_loss_score, test_acc_score = model.evaluate(x_test, y_test)

print(f'Final Loss: {round(test_loss_score, 2)}.')
print(f'Final Performance: {round(test_acc_score * 100, 2)} %.')
```

    Version:  2.10.1
    Eager mode:  True
    GPU is available
    Model: "model"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     input_1 (InputLayer)           [(None, 100)]        0           []                               
                                                                                                      
     embedding (Embedding)          (None, 100, 128)     2560000     ['input_1[0][0]']                
                                                                                                      
     layer_normalization (LayerNorm  (None, 100, 128)    256         ['embedding[0][0]']              
     alization)                                                                                       
                                                                                                      
     multi_head_attention (MultiHea  (None, 100, 128)    527488      ['layer_normalization[0][0]',    
     dAttention)                                                      'layer_normalization[0][0]']    
                                                                                                      
     dropout (Dropout)              (None, 100, 128)     0           ['multi_head_attention[0][0]']   
                                                                                                      
     tf.__operators__.add (TFOpLamb  (None, 100, 128)    0           ['dropout[0][0]',                
     da)                                                              'embedding[0][0]']              
                                                                                                      
     layer_normalization_1 (LayerNo  (None, 100, 128)    256         ['tf.__operators__.add[0][0]']   
     rmalization)                                                                                     
                                                                                                      
     conv1d (Conv1D)                (None, 100, 4)       516         ['layer_normalization_1[0][0]']  
                                                                                                      
     dropout_1 (Dropout)            (None, 100, 4)       0           ['conv1d[0][0]']                 
                                                                                                      
     conv1d_1 (Conv1D)              (None, 100, 128)     640         ['dropout_1[0][0]']              
                                                                                                      
     tf.__operators__.add_1 (TFOpLa  (None, 100, 128)    0           ['conv1d_1[0][0]',               
     mbda)                                                            'tf.__operators__.add[0][0]']   
                                                                                                      
     layer_normalization_2 (LayerNo  (None, 100, 128)    256         ['tf.__operators__.add_1[0][0]'] 
     rmalization)                                                                                     
                                                                                                      
     multi_head_attention_1 (MultiH  (None, 100, 128)    527488      ['layer_normalization_2[0][0]',  
     eadAttention)                                                    'layer_normalization_2[0][0]']  
                                                                                                      
     dropout_2 (Dropout)            (None, 100, 128)     0           ['multi_head_attention_1[0][0]'] 
                                                                                                      
     tf.__operators__.add_2 (TFOpLa  (None, 100, 128)    0           ['dropout_2[0][0]',              
     mbda)                                                            'tf.__operators__.add_1[0][0]'] 
                                                                                                      
     layer_normalization_3 (LayerNo  (None, 100, 128)    256         ['tf.__operators__.add_2[0][0]'] 
     rmalization)                                                                                     
                                                                                                      
     conv1d_2 (Conv1D)              (None, 100, 4)       516         ['layer_normalization_3[0][0]']  
                                                                                                      
     dropout_3 (Dropout)            (None, 100, 4)       0           ['conv1d_2[0][0]']               
                                                                                                      
     conv1d_3 (Conv1D)              (None, 100, 128)     640         ['dropout_3[0][0]']              
                                                                                                      
     tf.__operators__.add_3 (TFOpLa  (None, 100, 128)    0           ['conv1d_3[0][0]',               
     mbda)                                                            'tf.__operators__.add_2[0][0]'] 
                                                                                                      
     layer_normalization_4 (LayerNo  (None, 100, 128)    256         ['tf.__operators__.add_3[0][0]'] 
     rmalization)                                                                                     
                                                                                                      
     multi_head_attention_2 (MultiH  (None, 100, 128)    527488      ['layer_normalization_4[0][0]',  
     eadAttention)                                                    'layer_normalization_4[0][0]']  
                                                                                                      
     dropout_4 (Dropout)            (None, 100, 128)     0           ['multi_head_attention_2[0][0]'] 
                                                                                                      
     tf.__operators__.add_4 (TFOpLa  (None, 100, 128)    0           ['dropout_4[0][0]',              
     mbda)                                                            'tf.__operators__.add_3[0][0]'] 
                                                                                                      
     layer_normalization_5 (LayerNo  (None, 100, 128)    256         ['tf.__operators__.add_4[0][0]'] 
     rmalization)                                                                                     
                                                                                                      
     conv1d_4 (Conv1D)              (None, 100, 4)       516         ['layer_normalization_5[0][0]']  
                                                                                                      
     dropout_5 (Dropout)            (None, 100, 4)       0           ['conv1d_4[0][0]']               
                                                                                                      
     conv1d_5 (Conv1D)              (None, 100, 128)     640         ['dropout_5[0][0]']              
                                                                                                      
     tf.__operators__.add_5 (TFOpLa  (None, 100, 128)    0           ['conv1d_5[0][0]',               
     mbda)                                                            'tf.__operators__.add_4[0][0]'] 
                                                                                                      
     layer_normalization_6 (LayerNo  (None, 100, 128)    256         ['tf.__operators__.add_5[0][0]'] 
     rmalization)                                                                                     
                                                                                                      
     multi_head_attention_3 (MultiH  (None, 100, 128)    527488      ['layer_normalization_6[0][0]',  
     eadAttention)                                                    'layer_normalization_6[0][0]']  
                                                                                                      
     dropout_6 (Dropout)            (None, 100, 128)     0           ['multi_head_attention_3[0][0]'] 
                                                                                                      
     tf.__operators__.add_6 (TFOpLa  (None, 100, 128)    0           ['dropout_6[0][0]',              
     mbda)                                                            'tf.__operators__.add_5[0][0]'] 
                                                                                                      
     layer_normalization_7 (LayerNo  (None, 100, 128)    256         ['tf.__operators__.add_6[0][0]'] 
     rmalization)                                                                                     
                                                                                                      
     conv1d_6 (Conv1D)              (None, 100, 4)       516         ['layer_normalization_7[0][0]']  
                                                                                                      
     dropout_7 (Dropout)            (None, 100, 4)       0           ['conv1d_6[0][0]']               
                                                                                                      
     conv1d_7 (Conv1D)              (None, 100, 128)     640         ['dropout_7[0][0]']              
                                                                                                      
     tf.__operators__.add_7 (TFOpLa  (None, 100, 128)    0           ['conv1d_7[0][0]',               
     mbda)                                                            'tf.__operators__.add_6[0][0]'] 
                                                                                                      
     global_average_pooling1d (Glob  (None, 100)         0           ['tf.__operators__.add_7[0][0]'] 
     alAveragePooling1D)                                                                              
                                                                                                      
     dense (Dense)                  (None, 128)          12928       ['global_average_pooling1d[0][0]'
                                                                     ]                                
                                                                                                      
     dropout_8 (Dropout)            (None, 128)          0           ['dense[0][0]']                  
                                                                                                      
     dense_1 (Dense)                (None, 1)            129         ['dropout_8[0][0]']              
                                                                                                      
    ==================================================================================================
    Total params: 4,689,681
    Trainable params: 4,689,681
    Non-trainable params: 0
    __________________________________________________________________________________________________
    Epoch 1/20
    2807/2807 [==============================] - 270s 95ms/step - loss: 0.2700 - accuracy: 0.8937 - val_loss: 0.1918 - val_accuracy: 0.9303
    Epoch 2/20
    2807/2807 [==============================] - 271s 97ms/step - loss: 0.1881 - accuracy: 0.9314 - val_loss: 0.1797 - val_accuracy: 0.9337
    Epoch 3/20
    2807/2807 [==============================] - 272s 97ms/step - loss: 0.1639 - accuracy: 0.9419 - val_loss: 0.2098 - val_accuracy: 0.9273
    Epoch 4/20
    2807/2807 [==============================] - 270s 96ms/step - loss: 0.1550 - accuracy: 0.9439 - val_loss: 0.1926 - val_accuracy: 0.9345
    Epoch 5/20
    2807/2807 [==============================] - 267s 95ms/step - loss: 0.1525 - accuracy: 0.9461 - val_loss: 0.2172 - val_accuracy: 0.9305
    Epoch 6/20
    2807/2807 [==============================] - 267s 95ms/step - loss: 0.1537 - accuracy: 0.9451 - val_loss: 0.2201 - val_accuracy: 0.9188
    Epoch 7/20
    2806/2807 [============================>.] - ETA: 0s - loss: 0.1473 - accuracy: 0.9471Restoring model weights from the end of the best epoch: 2.
    2807/2807 [==============================] - 267s 95ms/step - loss: 0.1473 - accuracy: 0.9471 - val_loss: 0.2765 - val_accuracy: 0.9310
    Epoch 7: early stopping
    439/439 [==============================] - 25s 57ms/step - loss: 0.1906 - accuracy: 0.9316
    Final Loss: 0.19.
    Final Performance: 93.16 %.
    

Bellow we can test our trained model. You can also download the trained model (+ learned vocabulary) in [this link](https://drive.google.com/uc?export=download&id=1slnxFW9cP6GwUksnCYhuqRfVAAjfiJG_). 🙃


```python
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('models/toxicity_model.keras')

with open('models/toxic_vocabulary.txt', encoding='utf-8') as fp:
    vocabulary = [line.strip() for line in fp]
    fp.close()

vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=20000,
                                        output_mode="int",
                                        output_sequence_length=100,
                                        vocabulary=vocabulary)

strings = [
    'I think you should shut up your big mouth',
    'I do not agree with you'
]

preds = model.predict(vectorization_layer(strings),verbose=0)

for i, string in enumerate(strings):
    print(f'{string}\n')
    print(f'Toxic 🤬 {round((1 - preds[i][0]) * 100, 2)}% | Not toxic 😊 {round(preds[i][0] * 100, 2)}\n')
    print("_" * 50)
```

    
    I think you should shut up your big mouth
    
    Toxic 🤬 95.73% | Not toxic 😊 4.27
    __________________________________________________
    
    I do not agree with you
    
    Toxic 🤬 0.99% | Not toxic 😊 99.01
    __________________________________________________
    

You can try to repurpose this architecture for other applications and tasks, like multi-classification (changing the function of the last layer to a `softmax` with the intended number of unique classes to your problem) instead of binary classification.

---

Return to the [castle](https://github.com/Nkluge-correa/teeny-tiny_castle).
