import os
import ccxt
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from tkinter import ttk
from PIL import Image, ImageTk
from datetime import datetime


class CNN_LSTM_Model(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv1D(filters=hp.Int('conv1d_filters', min_value=32, max_value=128, step=32),
                                         kernel_size=hp.Choice('conv1d_kernel_size', values=[3, 5, 7]),
                                         activation='relu',
                                         input_shape=self.input_shape))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.LSTM(units=hp.Int('lstm_units', min_value=30, max_value=120, step=30),
                                       return_sequences=True))
        model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(tf.keras.layers.LSTM(units=hp.Int('lstm_units', min_value=30, max_value=120, step=30)))
        model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(tf.keras.layers.Dense(units=1))

        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(
                          hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')))

        return model

# Constants
exchange = ccxt.binance()
lookback_hours = 94
limit = 12000  # candles
model_directory = 'models/'
model_name_template = '{}_{}_model.h5'
scaler_name_template = '{}_{}_scaler.pkl'
interval_time = 9000  # in ms
symbol_options = ['BTC/USDT', 'ETH/USDT', 'LTC/USDT',
                  'ADA/USDT', 'XRP/USDT', 'MATIC/USDT', 'AGIX/USDT', 'BNB/USDT']
time_frame_options = ['5m', '15m', '1h', '4h', '1d']

# Default symbol and time frame
symbol = 'ETH/USDT'
time_frame = '1h'


def get_scaler(symbol, time_frame):
    scaler_name = scaler_name_template.format(
        symbol, time_frame).replace('/', '_')
    scaler_path = os.path.join(model_directory, scaler_name)

    if os.path.isfile(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        historical_data = exchange.fetch_ohlcv(symbol, time_frame, limit=limit)
        prices = np.array([row[4] for row in historical_data])
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(prices.reshape(-1, 1))
        joblib.dump(scaler, scaler_path)

    return scaler


def get_model(symbol, time_frame):
    model_name = model_name_template.format(
        symbol, time_frame).replace('/', '_')
    model_path = os.path.join(model_directory, model_name)

    if os.path.isfile(model_path):
        return tf.keras.models.load_model(model_path)

    scaler = get_scaler(symbol, time_frame)
    historical_data = exchange.fetch_ohlcv(symbol, time_frame, limit=limit)
    prices = np.array([row[4] for row in historical_data])
    scaled_prices = scaler.transform(prices.reshape(-1, 1))
    return train(scaled_prices, model_path)


def get_data(symbol, time_frame):
    scaler = get_scaler(symbol, time_frame)
    model = get_model(symbol, time_frame)
    historical_data = exchange.fetch_ohlcv(symbol, time_frame, limit=limit)
    prices = np.array([row[4] for row in historical_data])
    scaled_prices = scaler.transform(prices.reshape(-1, 1))
    return scaler, model, scaled_prices


def train(scaled_prices, model_path):
    X_train = []
    y_train = []
    for i in range(lookback_hours, len(scaled_prices)):
        X_train.append(scaled_prices[i - lookback_hours:i, 0])
        y_train.append(scaled_prices[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    input_shape = (lookback_hours, 1)
    hypermodel = CNN_LSTM_Model(input_shape)
    tuner = RandomSearch(
        hypermodel,
        objective='loss',
        max_trials=10,
        executions_per_trial=3,
        directory='my_dir',
        project_name='helloworld')

    tuner.search(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, epochs=200)

    best_model = tuner.get_best_models()[0]
    best_model.save(model_path)

    return best_model


def update_accuracy():
    global scheduled_accuracy_update_id
    X_test = []
    y_test = []
    for i in range(lookback_hours, len(scaled_prices)):
        X_test.append(scaled_prices[i - lookback_hours:i, 0])
        y_test.append(scaled_prices[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    loss = model.evaluate(X_test, y_test)
    accuracy = 1 - loss
    root.after(0, lambda: accuracy_label.configure(
        text="Model Accuracy: {:.2%}".format(accuracy)))
    scheduled_accuracy_update_id = root.after(interval_time, update_accuracy)


predicted_file = open('predicted_prices.txt', 'a')

last_bar_timestamp = None
last_recorded_prediction = None

def update_prediction():
    global prices, scheduled_prediction_update_id, last_bar_timestamp
    # Получить последний бар
    last_bar = exchange.fetch_ohlcv(symbol, time_frame, limit=2)[-1]
    # Обновить прогноз
    recent_prices = scaler.transform(prices[-limit:].reshape(-1, 1))
    X_predict = recent_prices[-lookback_hours:].reshape(1, -1)
    next_price = scaler.inverse_transform(model.predict(X_predict))[0][0]
    actual_price = last_bar[4]

    if next_price > prices[-1]:
        root.after(0, lambda: predicted_price_label.configure(
            text=f"The predicted price for the next {time_frame} is: ${next_price:,.2f} \u2191"))
    else:
        root.after(0, lambda: predicted_price_label.configure(
            text=f"The predicted price for the next {time_frame} is: ${next_price:,.2f} \u2193"))

    # Запись прогнозируемой цены в файл при открытии нового бара
    if last_bar_timestamp is None or last_bar[0] != last_bar_timestamp:
        last_bar_timestamp = last_bar[0]
        # Преобразовать timestamp в читаемую дату/время
        timestamp_str = datetime.fromtimestamp(last_bar_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
        with open('predicted_prices.txt', 'a') as predicted_file:
            predicted_file.write(f"{timestamp_str}: {symbol} predicted price for the next {time_frame}: ${next_price:,.2f}\n")

    prices = np.append(prices, actual_price)
    # Перепланировать обновление
    scheduled_prediction_update_id = root.after(interval_time, update_prediction)


scheduled_accuracy_update_id = None
scheduled_prediction_update_id = None


def update_data(new_symbol, new_time_frame):
    global symbol, time_frame, scaler, model, scaled_prices, prices
    global scheduled_accuracy_update_id, scheduled_prediction_update_id

    # Cancel any pending update_accuracy() and update_prediction() calls
    if scheduled_accuracy_update_id is not None:
        root.after_cancel(scheduled_accuracy_update_id)
    if scheduled_prediction_update_id is not None:
        root.after_cancel(scheduled_prediction_update_id)

    symbol = new_symbol
    time_frame = new_time_frame
    scaler, model, scaled_prices = get_data(symbol, time_frame)
    prices = np.array(
        [row[4] for row in exchange.fetch_ohlcv(symbol, time_frame, limit=limit)])
    predicted_price_label.configure(
        text=f"The predicted price for the next {time_frame} is: LOADING...")
    accuracy_label.configure(text="Model Accuracy: LOADING...")

    # Schedule the initial execution of the update_accuracy() and update_prediction() functions
    scheduled_accuracy_update_id = root.after(0, update_accuracy)
    scheduled_prediction_update_id = root.after(0, update_prediction)


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


create_directory('models')

root = tk.Tk()
root.title("Kovi AI Software")
root.geometry('600x400')  # Width x Height

# Load the image
bg_image = Image.open("/Users/kovi/Documents/ai-crypto-price-prediction/abstract-blockchain-technology-design-hexagonal-1625815.jpg")  # Use the path to your image

# Resize the image to fit the window, if necessary
bg_image = bg_image.resize((600, 400), Image.ANTIALIAS)

# Create a PhotoImage from the Image
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Label with the PhotoImage and pack it
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

style = ttk.Style(root)
style.configure("Black.TMenubutton", foreground="black")

scaler, model, scaled_prices = get_data(symbol, time_frame)
prices = np.array(
    [row[4] for row in exchange.fetch_ohlcv(symbol, time_frame, limit=limit)])

symbol_var = tk.StringVar()
symbol_var.set(symbol)  # set default value
symbol_menu = ttk.OptionMenu(root, symbol_var, symbol, *symbol_options,
                            command=lambda new_symbol: update_data(new_symbol, time_frame_var.get()))
symbol_menu.config(style="Black.TMenubutton")
symbol_menu.pack()

time_frame_var = tk.StringVar()
time_frame_var.set(time_frame)  # set default value
time_frame_menu = ttk.OptionMenu(root, time_frame_var, time_frame, *time_frame_options,
                                command=lambda new_time_frame: update_data(symbol_var.get(), new_time_frame))
time_frame_menu.config(style="Black.TMenubutton")
time_frame_menu.pack()

predicted_price_label = tk.Label(
    root, text=f"The predicted price for the next {time_frame} is: LOADING...", font=("Helvetica", 20), fg="black")
predicted_price_label.pack(padx=20, pady=5)

accuracy_label = tk.Label(
    root, text="Model Accuracy: LOADING...", font=("Helvetica", 16), fg="black")
accuracy_label.pack(padx=20, pady=5)

# Создаем Labels для отображения цен BTC/USDT и ETH/USDT
btc_price_label = tk.Label(root, text="BTC/USDT: LOADING...", font=("Helvetica", 16), fg="black")
btc_price_label.pack(padx=20, pady=5)

eth_price_label = tk.Label(root, text="ETH/USDT: LOADING...", font=("Helvetica", 16), fg="black")
eth_price_label.pack(padx=20, pady=5)

def update_prices():
    global scheduled_price_update_id
    btc_price = exchange.fetch_ticker('BTC/USDT')['last']
    eth_price = exchange.fetch_ticker('ETH/USDT')['last']
    root.after(0, lambda: btc_price_label.configure(text=f"BTC/USDT: ${btc_price:.2f}"))
    root.after(0, lambda: eth_price_label.configure(text=f"ETH/USDT: ${eth_price:.2f}"))
    scheduled_price_update_id = root.after(interval_time, update_prices)

# Запускаем функцию обновления цен
scheduled_price_update_id = root.after(0, update_prices)

root.after(0, update_accuracy)
root.after(0, update_prediction)

root.mainloop()