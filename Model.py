import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report

# Load your dataset
# data = pd.read_csv('your_dataset.csv')  # Uncomment and modify to load your dataset

# Check data types
print(data.dtypes)

# Inspect unique values for 'Temperature' and 'Usage'
print(data['Temperature'].unique())
print(data['Usage'].unique())

# Convert 'Temperature' to numeric, handle errors
data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')

# Convert 'Usage' to numeric if applicable, or one-hot encode if it's categorical
data['Usage'] = pd.to_numeric(data['Usage'], errors='coerce')  # If it's numeric
# If 'Usage' is categorical, use one-hot encoding:
# data = pd.get_dummies(data, columns=['Usage'], drop_first=True)

# Check for missing values
print(data.isnull().sum())

# Handle missing values (fill or drop)
data.fillna(0, inplace=True)  # Example: filling NaNs with 0

# Define features and target
X = data.drop(columns=['Failure A'])
y = data['Failure A']

# Ensure all columns in X are numeric
print(X.dtypes)  # Check data types of X

# Convert any remaining non-numeric columns to numeric or drop them
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Check for any remaining non-numeric columns
print(X.dtypes)

# Drop any remaining non-numeric columns (if necessary)
X.dropna(axis=1, inplace=True)  # Drop columns with NaN values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# If binary classification
y_train = y_train.values  # No need for to_categorical if binary
y_test = y_test.values

# Build and compile the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Change to 1 unit for binary classification
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary_crossentropy for binary classification
              metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(
    filepath="best_model_weights.keras",  # Change to .keras
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, (y_pred > 0.5).astype(int)))
