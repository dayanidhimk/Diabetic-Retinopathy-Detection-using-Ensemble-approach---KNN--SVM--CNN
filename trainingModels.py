import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from PIL import Image
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_csv("train.csv")

# Construct image paths
image_dir = "colored_images/"
df["image_path"] = image_dir + df["id_code"] + ".png"

# Data Pre-Processing
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((128, 128))  # Resize to 128x128
        img_array = np.array(img).flatten()  # Flatten the image  -  converts the 3D array with shape (128, 128, 3) into a 1D array of length 128 * 128 * 3 = 49,152
        return img_array
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Apply the preprocess_image function to each image path
df["features"] = df["image_path"].apply(preprocess_image)

# Drop rows with invalid images
df = df.dropna(subset=["features"])



# Preparing the Data
# Separate features and labels
X = np.array(df["features"].tolist())
y = df["diagnosis"].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')





# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate KNN
y_pred_knn = knn.predict(X_test)
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn)}")        #KNN Accuracy: 0.7366984993178718

# Save the KNN model
joblib.dump(knn, 'knn_model.joblib')






# Train SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict and evaluate SVM
y_pred_svm = svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")           #SVM Accuracy: 0.7244201909959073

# Save the SVM model
joblib.dump(svm, 'svm_model.joblib')








# CNN Model

def preprocess_image_for_cnn(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))  # Resize to 128x128
    return np.array(img)

# Prepare data for CNN
X_cnn = np.array([preprocess_image_for_cnn(img) for img in df["image_path"]])
X_cnn = X_cnn / 255.0  # Normalize the pixel values

# Convert labels to one-hot encoding
y_cnn = tf.keras.utils.to_categorical(df["diagnosis"].values, num_classes=5)

# Split data for CNN
X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_cnn, y_cnn, test_size=0.2, random_state=42)

# CNN Model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        BatchNormalization(),
        Dense(5, activation='softmax')  # 5 classes, softmax activation
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train the CNN model
cnn_model = create_cnn_model()
cnn_model.fit(X_train_cnn, y_train_cnn, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the CNN model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn)
print(f"CNN Accuracy: {cnn_accuracy}")                       #CNN Accuracy: 0.7394270300865173

# Save the CNN model
cnn_model.save('cnn_model.h5')
