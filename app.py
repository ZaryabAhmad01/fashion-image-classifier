import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/fashion_cnn.h5')

# tf.keras.models.load_model('models/fashion_cnn.h5')

# tf.keras.models.load_model('models/fashion_cnn.keras')

model = load_model()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.set_page_config(page_title="Fashion Classifier", page_icon="👗")

st.title("👗 Fashion Image Classifier")
st.write("Upload an image of clothing (JPG, JPEG, PNG) and the AI will predict its category.")
st.warning("Note: This model is trained on 28x28 grayscale images (Fashion-MNIST). It works best on simple, centered clothing images.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image (match training preprocessing)
        img = image.convert('L') # Grayscale
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1) # Add batch and channel dims
        
        # Predict
        with st.spinner('Analyzing image...'):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
        
        # Display Results
        st.success(f"**Predicted Category:** {class_names[predicted_class]}")
        st.info(f"**Confidence:** {confidence:.2f}%")
        
        # Show confidence for all classes
        st.subheader("Confidence Scores for All Classes:")
        for i, (name, prob) in enumerate(zip(class_names, predictions[0])):
            st.write(f"{name}: {prob*100:.2f}%")
            st.progress(float(prob))
            
    except Exception as e:
        st.error(f"Error processing the image: {e}. Please try another image.")