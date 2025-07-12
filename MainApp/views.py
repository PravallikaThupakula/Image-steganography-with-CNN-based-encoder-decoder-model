from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras import layers, Model

# Create your views here.
def index(request):
    return render(request,'index.html')

def LogAction(request):
    uname=request.POST['username']
    pwd=request.POST['password']
    if uname == 'Admin' and pwd == 'Admin':
        return render(request,'AdminHome.html')
    else:
        context={'msg':'Login Failed...!!'}
        return render(request,'AdminApp/index.html',context)

def home(request):
    return render(request,'AdminHome.html')

def encode(request):
    return render(request,'Encode.html')




# Define the character set and mappings for one-hot encoding
char_set = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
char_to_index = {char: idx for idx, char in enumerate(char_set)}
index_to_char = {idx: char for idx, char in enumerate(char_set)}

# Function to convert text to one-hot encoding
def text_to_one_hot(text, char_to_index, text_length):
    one_hot_encoded = np.zeros((text_length, len(char_to_index)))  # Shape: (text_length, vocab_size)
    for i, char in enumerate(text):
        if i < text_length:
            index = char_to_index.get(char, -1)  # Get index of the character
            if index != -1:
                one_hot_encoded[i, index] = 1  # Set the corresponding index to 1
    return one_hot_encoded


# Encoder Network: Embeds the secret text into the image
def build_encoder(input_image_shape, text_length, vocab_size):
    image_input = layers.Input(shape=input_image_shape)
    text_input = layers.Input(shape=(text_length, vocab_size))

    # CNN layers to process the image
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)

    # Concatenate the text input with the image features
    x = layers.Concatenate()([x, layers.Flatten()(text_input)])

    # Fully connected layer to produce the final encoded image
    encoded_image = layers.Dense(np.prod(input_image_shape), activation='sigmoid')(x)
    encoded_image = layers.Reshape(input_image_shape)(encoded_image)

    encoder = Model(inputs=[image_input, text_input], outputs=encoded_image)
    return encoder

# Decoder Network: Extracts the secret text from the image
def build_decoder(input_image_shape, text_length, vocab_size):
    image_input = layers.Input(shape=input_image_shape)

    # CNN layers to process the encoded image and extract text features
    x = layers.Conv2D(128, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    decoded_text = layers.Dense(text_length * vocab_size, activation='softmax')(x)

    decoded_text = layers.Reshape((text_length, vocab_size))(decoded_text)

    decoder = Model(inputs=image_input, outputs=decoded_text)
    return decoder









global filename, uploaded_file_url
def EncodeAction(request):
    global filename, uploaded_file_url
    decoded_text='null'
    if request.method == 'POST' and request.FILES['file']:
        myfile = request.FILES['file']
        message = request.POST['message']
        if len(message) == 20:
            fs = FileSystemStorage()
            location = myfile.name
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            imagedisplay = cv2.imread(BASE_DIR + "/" + uploaded_file_url)

            # Load an image from your system (ensure the path is correct)
            #image_path = myfile  # Replace with the actual path to your image
            img = image.load_img(BASE_DIR + "/" + uploaded_file_url, target_size=(32, 32))  # Resize to 128x128
            img_array = image.img_to_array(img)  # Convert image to numpy array

            # Normalize the image to [0, 1]
            img_array = img_array / 255.0

            # Display the image
            plt.imshow(img_array)
            plt.title('Uploaded Image')
            plt.show()
            plt.close()


            print("Image Shape:", img_array.shape)

            # Example secret text and text length
            secret_text = message
            text_length = 20  # Maximum length of the secret text
            encoded_text = text_to_one_hot(secret_text, char_to_index, text_length)

            print("Encoded Text Shape:", encoded_text.shape)


            # Define input image shape
            input_image_shape = (32, 32, 3)

            # Build encoder and decoder
            encoder = build_encoder(input_image_shape, text_length, len(char_to_index))
            decoder = build_decoder(input_image_shape, text_length, len(char_to_index))

            # Define the full model (encoder + decoder)
            encoded_image = encoder.output
            decoded_text = decoder(encoded_image)

            full_model = Model(inputs=[encoder.input[0], encoder.input[1]], outputs=decoded_text)
            full_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            full_model.summary()


            # Prepare the image and text for training
            train_images = np.expand_dims(img_array, axis=0)  # Add batch dimension
            train_texts = np.expand_dims(encoded_text, axis=0)  # Add batch dimension

            # Train the full model
            full_model.fit([train_images, train_texts], train_texts, epochs=25, batch_size=1)

            # Save the model for future use
            full_model.save('encoder_decoder_model.h5')

            # Test the model with a new image and secret text
            test_image = np.expand_dims(img_array, axis=0)  # Use the same test image
            test_text = np.expand_dims(encoded_text, axis=0)  # Use the same secret text

            # Encode the secret text into the image
            encoded_test_image = encoder.predict([test_image, test_text])


            # Remove the batch dimension by squeezing it (assuming only one image in the batch)
            img_single = encoded_test_image[0]  # Or use img_array.squeeze() if you want to squeeze all dimensions

            # Display the image
            plt.imshow(img_single)# img_single will have shape (32, 32, 3)
            plt.title('Steganography Image')
            plt.axis('off')  # Hide axis for clarity
            plt.savefig('Stegano_image.png', bbox_inches='tight', pad_inches=0)
            plt.show()
            plt.close()


            # Decode the text from the encoded image
            decoded_text_from_image = decoder.predict(encoded_test_image)


            # Convert the decoded output (one-hot encoding) back to text
            decoded_text_from_image = np.argmax(decoded_text_from_image, axis=-1)  # Get the indices
            decoded_text = ''.join([index_to_char[idx] for idx in decoded_text_from_image[0]])

            print("Decoded Text:", decoded_text)
            context = {'data': decoded_text}
            return render(request,'Decode.html', context)
        else:
            context = {'data': "Message Length Must Be 20 Characters"}
            return render(request,'Encode.html', context)



