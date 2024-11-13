import tensorflow as tf
from model_definition import build_model

def train_model(X_train, y_train, X_test, y_test):
    model = build_model(X_train.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    return model, history
