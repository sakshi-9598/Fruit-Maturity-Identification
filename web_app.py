import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
from tensorflow.keras.models import load_model
import os



# Define the model paths
model_paths = {
    'own_cnn_model': 'Models//New_PROPOSED.h5',
    'densenet_model': 'Models//New_DenseNet121.h5',
    'vgg16': 'Models//New_VGG16.h5',
    'inceptionV3': 'Models//New_INCEPTIONV3.h5'
}


# Function to save the uploaded image
def save_uploaded_image(uploaded_image):
    image_path = "uploaded_image.jpg"
    with Image.open(uploaded_image) as image:
        image.save(image_path)
    return image_path

# Function to predict the fruit maturity
def predict_maturity(image, fruit, model):
    fruit_mapping = {"banana": 0}  # Add mappings for other fruits if needed
    class_labels = ["Overripe", "Ripe", "Unripe"]

    fruit_index = fruit_mapping.get(fruit.lower(), -1)
    if fruit_index == -1:
        return "Invalid fruit selection"

    image_path = save_uploaded_image(image)
    img = Image.open(image_path)
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image

    # Load the selected model
    model_path = model_paths.get(model)
    if model_path is None:
        return "Invalid model selection"

    loaded_model = load_model(model_path)

    # Perform prediction
    img_array = np.expand_dims(img_array, axis=0)
    predictions = loaded_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_class_index]

    return predicted_class

#------------------------------------------------------------------------------

# Home page
def home_page():
    
    # Set the page title
    background_style = """
        <style>
        .overlay {
            background-color: rgba(255, 0, 0, 0.9);
            padding: 20px;
        }
        </style>
        """
    # Render the CSS and HTML code
    st.markdown(background_style, unsafe_allow_html=True)

    # Add an overlay div to contain the content with opacity
    st.markdown(
        """
        <div class="overlay">
            <h1>Fruit Maturity Classifier</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    #-----------------------------------------------------------------

    #Add image at complete width
    st.markdown(
        """
        <style>
        .wide-image {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the image with adjusted width
    st.image("Images//fruits2.jpg", use_column_width=True, output_format='auto')
    
   #---------------------------------------------------------------------------
    background_style = """
            <style>
            .overlay1 {
                background-color: rgba(230, 160, 160, 0.8);
                padding: 20px;
            }
            </style>
            """
    # Render the CSS and HTML code
    st.markdown(background_style, unsafe_allow_html=True)

    # Add an overlay div to contain the content with opacity
    st.markdown(
        """
        <div class="overlay1">
            <h3>Welcome to the Fruit Maturity Classifier web application!</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    #-------------------------------------------------------------------------------------------
    
    st.subheader("Select a fruit and a model to predict its maturity")

    # Fruit selection dropdown menu
    selected_fruit = st.selectbox("Select a fruit", ["Banana"])
    st.write("You selected:", selected_fruit)

    # Model selection dropdown menu
    selected_model = st.selectbox("Select a model", ["own_cnn_model", "densenet_model", "vgg16", "inceptionV3"])
    st.write("You selected:", selected_model)

    # Image uploader
    st.header("Upload Image")
    
    uploaded_image = st.file_uploader("Upload an image of the fruit", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        st.write("Classifying...")

        # Perform prediction
        prediction = predict_maturity(uploaded_image, selected_fruit, selected_model)
        st.subheader("Prediction Result:")
        st.write("Maturity:", prediction)
        
    st.subheader("Hope you are doing great!!!")
    st.subheader("All the best...")
    # st.image("Images//front-end-developer.jpg", caption="Fruit Images", use_column_width=True)


#------------ABOUT US--------------------------
# About us page
def about_us_page():
    st.title("About Us")
    st.markdown("Welcome to the About Us page of the Fruit Maturity Classifier application! We are a team of passionate developers dedicated to providing accurate fruit maturity classification using deep learning techniques.")
    
        
    st.header("Our Mission")
    st.markdown("Our mission is to provide an accurate and user-friendly fruit maturity classification system that helps consumers make informed decisions about their fruit purchases. We strive to continuously improve our algorithms and expand our database to support a wide range of fruits.")

    st.header("Contact Us")
    st.markdown("Have any questions or suggestions? We'd love to hear from you! Please feel free to reach out to us through the contact form on our Contact Us page.")

    # Display the image
    st.image("Images//front-end-developer.jpg", caption="External Resources", use_column_width=True)
    
    
    st.header("Meet Our Team")
    
    # Display team member details
    team_members = [
        {
            "name": "Sakshi Kushwaha",
            "role": "Lead Developer",
            "background": "Sakshi Kushwaha is an experienced software engineer with a focus on machine learning and computer vision and has expertise in deep learning and image classification. She holds a Ph.D. in Data Science and has published several research papers on the topic.",
            "image_path": "Images//SakshiKushwaha.jpeg"
        },
        {
            "name": "Akrati Sachan",
            "role": "Data Scientist",
            "background": "Akrati Sachan is a final year undergraduate in computer science and technology and has expertise in deep learning and image classification. .",
            "image_path": "Images//Akrati.jpeg"
        },
        # Add more team members...
    ]
    
    for team_member in team_members:
        st.subheader(team_member["name"])
        st.image(team_member["image_path"], caption=team_member["name"], width=200)
        st.write(team_member["role"])
        st.write(team_member["background"])
        st.write("---")

#----------CONTACT US PAGE---------------------------
# Contact us page
def contact_us_page():
    st.title("Contact Us")
    st.markdown("Feel free to reach out to us with any questions, feedback, or inquiries. We're here to assist you!")

    st.header("Contact Form")
    
    # Display contact form
    name = st.text_input("Your Name")
    email = st.text_input("Your email adress")
    message = st.text_area("Feel free to discuss anything", height=200)
    
    if st.button("Send Message"):
        # Process the message and send it to the desired destination
        # You can add your custom logic here for processing the contact form submission
        st.success("Thank you for your message! We will get back to you shortly.")
        
    st.header("Our Location")
    st.markdown("Visit our office located at:")
    st.markdown("123 Main Street, City, Country")

    # st.image("office_location.jpg", caption="Our Office Location", use_column_width=True)

    st.header("Connect with Us")
    st.markdown("Stay connected with us on social media for updates and announcements.")

    # Display social media icons with links
    social_media_icons = {
        "Twitter": "https://twitter.com/fruitclassifier",
        "LinkedIn": "https://www.linkedin.com/company/fruit-maturity-classifier",
        # Add more social media platforms...
    }

    for platform, link in social_media_icons.items():
        st.markdown(f"[![{platform}](social_media_icons/{platform.lower()}.png)]({link})")

#--------------------MORE ABOUT FRUITS PAGE-----------------------
# More page
def more_page():
    st.title("More About Fruits")
    st.markdown("Explore additional information and resources related to fruit maturity classification.")

    st.header("Educational Content")
    st.markdown("Learn more about fruit maturity and classification through our educational articles and materials.")

    st.subheader("Article 1: Introduction to Fruit Maturity")
    st.markdown("Maturity at harvest is the most important factor that determines storage-life and final fruit quality. Immature fruits are more subject to shrivelling and mechanical damage, and are of inferior flavour quality when ripe. Overripe fruits are likely to become soft and mealy with insipid flavour soon after harvest. Fruits picked either too early or too late in their season are more susceptible to postharvest physiological disorders than fruits picked at the proper maturity.All fruits, with a few exceptions (such as pears, avocados, and bananas), reach their best eating quality when allowed to ripen on the plant. However, some fruits are usually picked mature but unripe so that they can withstand the postharvest handling system when shipped long-distance. Most currently used maturity indices are based on a compromise between those indices that would ensure the best eating quality to the consumer and those that provide the needed flexibility in marketing.")
    
    st.image("Images//home_bottom.jpeg", caption="External Resources", use_column_width=True)

    st.subheader("Article 2: Fruit Classification Techniques")
    st.markdown("Recent advancements in computer vision have enabled wide-ranging applications in every field of life. One such application area is fresh produce classification, but the classification of fruit and vegetable has proven to be a complex problem and needs to be further developed. Fruit and vegetable classification presents significant challenges due to interclass similarities and irregular intraclass characteristics. Selection of appropriate data acquisition sensors and feature representation approach is also crucial due to the huge diversity of the field. Fruit and vegetable classification methods have been developed for quality assessment and robotic harvesting but the current state-of-the-art has been developed for limited classes and small datasets. The problem is of a multi-dimensional nature and offers significantly hyperdimensional features, which is one of the major challenges with current machine learning approaches. Substantial research has been conducted for the design and analysis of classifiers for hyperdimensional features which require significant computational power to optimise with such features. In recent years numerous machine learning techniques for example, Support Vector Machine (SVM), K-Nearest Neighbour (KNN), Decision Trees, Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) have been exploited with many different feature description methods for fruit and vegetable classification in many real-life applications. This paper presents a critical comparison of different state-of-the-art computer vision methods proposed by researchers for classifying fruit and vegetable.")

    st.image("Images//fruits.jpeg", caption="Educational Articles", use_column_width=True)

    st.header("External Resources")
    st.markdown("Discover external websites and research papers for further exploration.")

    # Display external resource links
    st.markdown("[Website 1](https://www.example.com)")
    st.markdown("[Website 2](https://www.example.com)")
    st.markdown("[Research Paper](https://www.example.com)")

    st.image("Images//more.png", caption="External Resources", use_column_width=True)
    
#--------------------------------------------------------------------------------------------------------------------------------

# Function to switch pages based on sidebar selection
def page_router():
    pages = {
        "Home": home_page,
        "About Us": about_us_page,
        "Contact Us": contact_us_page,
        "More About Fruits": more_page
    }
    page = st.sidebar.selectbox("Navigate", list(pages.keys()))
    pages[page]()

# Main function
def main():
    st.set_page_config(page_title="Fruit Maturity Classifier", layout="wide")
    st.sidebar.title("Navigation")
    page_router()

if __name__ == "__main__":
    main()
