import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from PIL import Image
import argparse

# --- CONFIGURATION ---
IMG_H, IMG_W = 64, 512
BATCH_SIZE = 32
MAX_LABEL_LEN = 128 

def validate_input(root_path):
    if not root_path: return None
    d_path = os.path.join(root_path, 'data')
    m_file = os.path.join(root_path, 'directory', 'dataset_manifest.jsonl')
    return (m_file, d_path) if os.path.isdir(d_path) and os.path.isfile(m_file) else None

def scan_dataset(manifest_path):
    all_chars = set()
    total = 0
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                text = entry.get('text', '')
                if not text or len(text) > MAX_LABEL_LEN: continue
                for char in text:
                    all_chars.add(char)
                total += 1
            except: continue
    vocab = sorted(list(all_chars))
    # Index 0 is reserved for CTC blank
    char_map = {c: i + 1 for i, c in enumerate(vocab)}
    return char_map, total

def ctc_loss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    # Feature map width is IMG_W // 4 due to two pooling layers
    logit_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones(shape=(batch_len,), dtype="int64")
    label_length = tf.reduce_sum(tf.cast(tf.not_equal(y_true, -1), dtype="int64"), axis=1)

    y_true_clean = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_true), y_true)
    y_true_clean = tf.cast(y_true_clean, dtype="int32")
    y_pred_permuted = tf.transpose(y_pred, perm=[1, 0, 2])
    
    loss = tf.nn.ctc_loss(
        labels=y_true_clean,
        logits=y_pred_permuted,
        label_length=tf.cast(label_length, dtype="int32"),
        logit_length=tf.cast(logit_length, dtype="int32"),
        logits_time_major=True,
        blank_index=0
    )
    return tf.reduce_mean(loss)

def large_batch_generator(manifest_path, data_dir, char_map):
    while True:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            bx, by = [], []
            for line in f:
                try:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    if not text or len(text) > MAX_LABEL_LEN: continue
                    img_path = os.path.join(data_dir, f"{entry['id']}.png")
                    if not os.path.exists(img_path): continue

                    img = Image.open(img_path).convert('L')
                    img = img.resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
                    img_arr = np.array(img).astype(np.float32) / 255.0
                    img_arr = np.expand_dims(img_arr, -1)
                    img_arr = np.transpose(img_arr, (1, 0, 2)) 

                    label = [char_map[c] for c in text]
                    padded_label = label + [-1] * (MAX_LABEL_LEN - len(label))
                    bx.append(img_arr)
                    by.append(padded_label)

                    if len(bx) >= BATCH_SIZE:
                        yield np.array(bx), np.array(by)
                        bx, by = [], []
                except: continue

def build_model(vocab_size):
    total_classes = vocab_size + 1 
    input_img = layers.Input(shape=(IMG_W, IMG_H, 1), name='image')

    # CNN Block 1: Feature Extraction
    x = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # CNN Block 2: Deeper Features
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Reshape for RNN (Width becomes Time)
    new_shape = (IMG_W // 4, (IMG_H // 4) * 64)
    x = layers.Reshape(target_shape=new_shape)(x)
    
    # Dense Bridge
    x = layers.Dense(128, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # RNN Block: Sequence Learning
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Output: Logits for CTC
    logits = layers.Dense(total_classes, activation='linear', name='logits')(x)
    y_pred = layers.Softmax(name='output')(logits)

    return models.Model(inputs=input_img, outputs=y_pred)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to 'document training data' folder")
    args = parser.parse_args()

    input_path = args.input
    while not (paths := validate_input(input_path)):
        input_path = input("\nEnter path to 'document training data' folder: ")

    m_path, d_dir = paths
    NET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'neuralnet'))
    os.makedirs(NET_DIR, exist_ok=True)
    model_path = os.path.join(NET_DIR, 'ocr_model.keras')

    char_map, total_samples = scan_dataset(m_path)
    with open(os.path.join(NET_DIR, 'char_map.json'), 'w', encoding='utf-8') as f:
        json.dump(char_map, f, indent=4)

    model = build_model(len(char_map))
    
    # Increased learning rate to 0.001 to break the plateau
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=ctc_loss)

    # --- CALLBACKS ---
    stop_early = callbacks.EarlyStopping(
        monitor='loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor='loss',
        save_best_only=True,
        verbose=1
    )

    print(f"\nAlphabet: {len(char_map)} chars + 1 blank.")
    print(f"Training on {total_samples} samples with Kickstart optimizations...")

    model.fit(
        large_batch_generator(m_path, d_dir, char_map), 
        steps_per_epoch=max(1, total_samples // BATCH_SIZE), 
        epochs=200, 
        callbacks=[stop_early, checkpoint]
    )

    print(f"\nFinal model and char_map ready in: {NET_DIR}")

if __name__ == "__main__":
    main()