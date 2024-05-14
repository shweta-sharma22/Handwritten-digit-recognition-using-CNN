import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

train_new_model = True

if train_new_model:
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)

    val_loss, val_acc = model.evaluate(X_test, y_test)
    print("Model loss: ", val_loss)
    print("Model Accuracy: ", val_acc)

    model.save('handwritten_digits.keras')
else:
    model = tf.keras.models.load_model('handwritten_digits.keras')

# Title and Information Page
def project_info():
    st.title("Project Overview")
    st.write("Our project \"Handwritten Digit Recognition using CNN \" leverages the power of the MNIST dataset—a curated collection of 28x28 grayscale images depicting handwritten digits from 0 to 9. Through meticulous training on this dataset, our CNN model achieves unparalleled accuracy in recognizing and classifying digits. Whether you're deciphering a scribbled 1 or a looping 9, our user-friendly interface ensures a smooth and intuitive experience, making handwritten digit recognition very simple.")

# Model Information and Evaluation Page
def model_info():
    st.title("Model Information")
    st.header("MNIST Dataset")
    st.write("The MNIST dataset comprises 60,000 training images and 10,000 test images of handwritten digits. Widely recognized in the realm of machine learning, it serves as a benchmark for training and evaluating classification algorithms, especially Convolutional Neural Networks (CNNs). Each image is precisely 28x28 pixels in size and grayscale, rendering it ideal for training sophisticated image classification models.")

    st.header("Convolutional Neural Network (CNN)")
    st.write("Convolutional Neural Networks (CNNs) represent a cornerstone in image classification endeavors. Distinguished by their convolutional layers, pooling layers, and fully connected layers, CNNs exhibit a remarkable proficiency in deciphering intricate visual patterns. These networks autonomously discern hierarchical features from input images, rendering them invaluable for tasks such as handwritten digit recognition.")

    st.header("Model Evaluation")
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()

    x_test = x_test / 255.0
    model = tf.keras.models.load_model('handwritten_digits.keras')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(x_test), axis=1)

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Classification Report
    st.subheader("Classification Report")
    cls_report = classification_report(y_test, y_pred, output_dict=True)

    df_cls_report = pd.DataFrame.from_dict(cls_report)
    df_cls_report = df_cls_report.transpose()
    st.table(df_cls_report)

    st.write(f"Test Loss: {loss}")
    st.write(f"Test Accuracy: {accuracy}")

    # Plotting epochs vs accuracy and epochs vs loss
    st.subheader("Model Training Metrics")
    history = model.history.history

    if history:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Plotting Accuracy
        ax[0].plot(history['accuracy'], label='Training Accuracy')
        ax[0].set_title('Training Accuracy over Epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()

        # Plotting Loss
        ax[1].plot(history['loss'], label='Training Loss', color='orange')
        ax[1].set_title('Training Loss over Epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].legend()

        st.pyplot(fig)
    else:
        st.write("Model history not available.")

# Preprocess Image function
def preprocess_image(image):
    grayscale_image = image.convert('L')
    resized_image = grayscale_image.resize((28, 28)) 
    inverted_image = ImageOps.invert(resized_image)
    preprocessed_image = np.array(inverted_image)
    return preprocessed_image

# Predict Digit function
def predict_digit(image, model):
    try:
        img = preprocess_image(image)
        prediction = model.predict(np.array([img]))
        predicted_digit = np.argmax(prediction)

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f"Predicted Digit: {predicted_digit}")
        return predicted_digit
    except Exception as e:
        st.write(f"Error: {e}")

# Main Project Page
def main_project():
    st.title("Handwritten Digit Recognition")
    st.write("Upload an image of a handwritten digit and the model will predict the digit.")
    model = tf.keras.models.load_model('handwritten_digits.keras')
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        resized_image = image.resize((200, 200))
        st.image(resized_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Preprocess Image'):
            preprocessed_image = preprocess_image(resized_image)
            plt.imshow(preprocessed_image, cmap='gray')
            plt.axis('off')
            plt.title('Preprocessed Image')
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

        if st.button('Predict'):
            predict_digit(resized_image, model)

# Main Streamlit App
def main():
    st.sidebar.title("Contents")
    selected_page = st.sidebar.selectbox("Go to", ["Project Overview", "Model Information", "Main Project"])

    if selected_page == "Project Overview":
        project_info()
    elif selected_page == "Model Information":
        model_info()
    elif selected_page == "Main Project":
        main_project()

if __name__ == "__main__":
    main()
