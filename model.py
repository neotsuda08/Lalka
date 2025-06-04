import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

studs = pd.read_csv(Path('C:/Users/Ð–Ð°Ðº/Desktop/Plane Price.csv'))
print(studs.head())

studs.dropna(subset = 'Price',inplace = True)

studs = studs[["Rcmnd cruise Knots", "Stall Knots dirty", "Fuel gal/lbs", "Eng out rate of climb", "Takeoff over 50ft", "Price"]]
studs = studs.dropna()

print(studs.head())

Y = studs['Price']
X = studs.drop("Price", axis=1)

from sklearn.preprocessing import StandardScaler, MinMaxScaler  

ss = StandardScaler()
x2 = ss.fit_transform(X)

x2 = pd.DataFrame(x2)
print (x2.head(3))

from sklearn.model_selection import train_test_split


X_train_full,X_test,y_train_full,y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
X_train,X_valid,y_train,y_valid = train_test_split(X_train_full,y_train_full,test_size=0.2,random_state=0)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


# 1. ÐžÐ¿Ñ€ÐµÐ´ÐµÐ»ÐµÐ½Ð¸Ðµ RÂ² Ð¼ÐµÑ‚Ñ€Ð¸ÐºÐ¸ (Ð´Ð¾Ð±Ð°Ð²Ð»ÑÐµÐ¼ accuracy-Ð°Ð½Ð°Ð»Ð¾Ð³ Ð´Ð»Ñ Ñ€ÐµÐ³Ñ€ÐµÑÑÐ¸Ð¸)
def r_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())

# 2. Ð¡Ð¾Ð·Ð´Ð°Ð½Ð¸Ðµ Ð¼Ð¾Ð´ÐµÐ»Ð¸
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

# 3. ÐšÐ¾Ð¼Ð¿Ð¸Ð»ÑÑ†Ð¸Ñ Ñ Ñ‚Ñ€ÐµÐ¼Ñ Ð¼ÐµÑ‚Ñ€Ð¸ÐºÐ°Ð¼Ð¸
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae', r_squared])  # Ð¢ÐµÐ¿ÐµÑ€ÑŒ 2 Ð¼ÐµÑ‚Ñ€Ð¸ÐºÐ¸ + loss = Ð²ÑÐµÐ³Ð¾ 3 Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸Ñ

# 4. ÐžÐ±ÑƒÑ‡ÐµÐ½Ð¸Ðµ
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
    verbose=1
)

# 5. Ð’Ð¸Ð·ÑƒÐ°Ð»Ð¸Ð·Ð°Ñ†Ð¸Ñ
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss (MSE)')
plt.legend()

# MAE
plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Train')
plt.plot(history.history['val_mae'], label='Validation')
plt.title('MAE')
plt.legend()

# RÂ²
plt.subplot(1, 3, 3)
plt.plot(history.history['r_squared'], label='Train')
plt.plot(history.history['val_r_squared'], label='Validation')
plt.title('RÂ² Score')
plt.legend()

plt.tight_layout()
plt.show()

# 6. ÐžÑ†ÐµÐ½ÐºÐ° Ð¼Ð¾Ð´ÐµÐ»Ð¸ (Ñ‚ÐµÐ¿ÐµÑ€ÑŒ Ð¿Ñ€Ð°Ð²Ð¸Ð»ÑŒÐ½Ð¾ Ñ€Ð°ÑÐ¿Ð°ÐºÐ¾Ð²Ñ‹Ð²Ð°ÐµÐ¼ 3 Ð·Ð½Ð°Ñ‡ÐµÐ½Ð¸Ñ)
test_loss, test_mae, test_r2 = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Results:")
print(f"MSE (Loss): {test_loss:.2f}")
print(f"MAE: {test_mae:.2f}")
print(f"RÂ² Score: {test_r2:.2f}")

from tensorflow.keras.models import load_model


model.save("my_model.keras")

import os

file_size = os.path.getsize("my_model.keras")
print(f"Ð Ð°Ð·Ð¼ÐµÑ€ Ñ„Ð°Ð¹Ð»Ð°: {file_size} Ð±Ð°Ð¹Ñ‚") 
