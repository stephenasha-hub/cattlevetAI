import io
import os
import uuid

import numpy as np
#import openai
import pyttsx3
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template, jsonify

#from api_key import OPENAI_API_KEY

app = Flask(__name__)

model_path = "models/keras_Model.h5"
labels_path = "models/labels.txt"
# Load the labels
class_names = open(labels_path, "r").readlines()

# Load the trained model
model = tf.keras.models.load_model(model_path)

class_labels = class_names

#audio_dir = 'static/audio_files/'

#openai.api_key = OPENAI_API_KEY





# if not os.path.exists(audio_dir):
#     os.makedirs(audio_dir)


engine = pyttsx3.init()
welcome = "Hello, am Thunder your cattle Vet AI assistant, i can identify the disease your animal is suffering from just in a blink of an eye, "
wanatry = "Come on, choose an image and press the process image button and wait to see the magic, am waiting "
say_txt = welcome + wanatry
engine.say(say_txt)
engine.runAndWait()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'})

    try:
        # Read image using PIL
        img = Image.open(io.BytesIO(image_file.read()))

        # Preprocess the image for model input
        img = img.resize((224, 224))  # Cnverting model to 224x224
        img_array = np.asarray(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform classification
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        

        # Convert class prediction to text
        text_response = f"The image is classified as: {predicted_class}"
        fmd = """
            <h2>Foot and Mouth Disease (FMD)</h2> is a highly contagious viral disease that affects cloven-hoofed animals like cattle, pigs, sheep, and goats. The primary focus for managing FMD in cattle is containment, supportive care, and prevention. Here is a comprehensive approach to treating and managing FMD in cattle:
            "<br><br>"
            1. Isolation and Containment:<br>
            - Immediate Isolation: Infected and suspected animals should be isolated from healthy ones to prevent the spread of the virus.<br>
            - Quarantine: Implement a quarantine zone around the affected area. Restrict movement of animals, personnel, and equipment.
            <br><br>
            2. Supportive Care:<br>
            - Hydration and Nutrition: Ensure that animals have access to clean water and are provided with easily digestible and nutritious feed, as they may have difficulty eating due to mouth lesions.<br>
            - Pain Management: Administer anti-inflammatory and pain-relieving medications as prescribed by a veterinarian to alleviate discomfort.<br>
            - Wound Care: Clean and disinfect lesions on the mouth, feet, and udders. Use antiseptics to prevent secondary bacterial infections.
            <br><br>
            3. Vaccination:<br>
            - Emergency Vaccination: In areas where FMD is prevalent or during an outbreak, vaccination can help control the spread. Follow the guidelines provided by veterinary authorities regarding the type and schedule of vaccination.
            <br><br>
            4. Biosecurity Measures:<br>
            - Sanitation: Regularly disinfect equipment, vehicles, and facilities. Use approved disinfectants known to be effective against the FMD virus.<br>
            - Protective Clothing: Use dedicated clothing and footwear for handling infected animals. Ensure proper disposal or thorough cleaning and disinfection of protective gear.<br>
            - Visitor Control: Limit access to the farm and ensure that any visitors follow strict biosecurity protocols.
            <br><br>
            5. Monitoring and Reporting:<br>
            - Regular Monitoring: Closely monitor all animals for signs of the disease. Early detection can help prevent the spread of FMD.<br>
            - Reporting: Report any suspected cases to local veterinary authorities immediately. FMD is a notifiable disease, and authorities will implement control measures to manage the outbreak.
            <br><br>
            6. Eradication Measures:<br>
            - Culling: In severe cases and under veterinary and governmental guidance, culling of infected and at-risk animals may be necessary to control the outbreak.<br>
            - Disposal: Proper disposal of carcasses through incineration or burial to prevent further spread of the virus.
            <br><br>
            7. Long-Term Strategies:<br>
            - Education and Training: Train farm personnel on recognizing symptoms, biosecurity measures, and proper handling of infected animals.<br>
            - Surveillance: Implement regular surveillance and monitoring programs to detect and respond to FMD outbreaks promptly.
            <br><br>
            Additional Considerations:<br>
            - Consultation with Veterinarians: Always seek advice and treatment plans from licensed veterinarians. They can provide specific recommendations based on the severity of the outbreak and local regulations.<br>
            - Community Cooperation: Work with neighboring farms and local communities to implement regional control and prevention measures effectively.
            <br><br>
            By following these steps, you can help manage and control Foot and Mouth Disease in cattle, minimizing its impact on animal health and farm operations.
            """


        treatment_diarrhea = """
        Diarrhea in cattle can be caused by various factors including infections (bacterial, viral, or parasitic), dietary changes, stress, and environmental conditions. Proper management and treatment are crucial to prevent dehydration and further complications. Here is a comprehensive approach to treating and managing diarrhea in cattle:
        \n
        1. Isolation and Containment:
        - Immediate Isolation: Isolate affected animals to prevent the spread of infectious agents to healthy cattle.
        - Quarantine: Implement a quarantine zone for new or returning animals to the herd to monitor for signs of illness before reintroduction.
        \n
        2. Supportive Care:
        - Hydration: Ensure that affected animals have access to plenty of clean, fresh water. Oral rehydration solutions or electrolytes may be necessary to prevent dehydration.
        - Nutrition: Provide easily digestible, high-quality feed. Avoid sudden changes in diet and ensure the feed is free from contaminants.
        - Rest: Minimize stress and provide a comfortable environment for recovery.
        \n
        3. Medical Treatment:
        - Antimicrobials: Use antibiotics only if a bacterial infection is confirmed. Always follow veterinary advice and prescribed dosages.
        - Antiparasitics: Administer appropriate antiparasitic treatments if parasitic infection is suspected or confirmed.
        - Probiotics: Consider using probiotics to help restore gut flora and improve digestion.
        \n
        4. Biosecurity Measures:
        - Sanitation: Maintain strict hygiene in feeding and watering areas. Regularly clean and disinfect barns, pens, and equipment.
        - Protective Clothing: Use dedicated clothing and footwear when handling sick animals. Properly clean and disinfect gear to prevent spreading the infection.
        - Visitor Control: Limit access to the farm and ensure that any visitors follow strict biosecurity protocols.
        \n
        5. Monitoring and Reporting:
        - Regular Monitoring: Keep a close eye on all animals, particularly those showing signs of diarrhea. Monitor hydration status, appetite, and overall health.
        - Reporting: Report severe or persistent cases of diarrhea to a veterinarian. They can help determine the underlying cause and recommend appropriate treatments.
        \n
        6. Long-Term Strategies:
        - Vaccination: Follow a vaccination schedule for common infectious agents that cause diarrhea, as recommended by your veterinarian.
        - Nutrition Management: Ensure a balanced diet that meets the nutritional needs of the cattle. Avoid abrupt changes in feed and gradually introduce new diets.
        - Stress Reduction: Minimize stress factors such as overcrowding, poor ventilation, and abrupt changes in environment or management practices.
        \n
        7. Environmental Management:
        - Clean Water Supply: Ensure a reliable and clean water source to prevent waterborne diseases.
        - Pasture Management: Rotate pastures and avoid overgrazing to reduce the risk of parasitic infections.
        - Bedding: Provide clean, dry bedding to prevent environmental contamination and reduce stress on the animals.
        \n
        Additional Considerations:
        - Consultation with Veterinarians: Always seek advice and treatment plans from licensed veterinarians. They can provide specific recommendations based on the severity of the outbreak and local conditions.
        - Record Keeping: Maintain detailed records of animal health, treatments administered, and management practices. This information can be valuable for identifying patterns and preventing future outbreaks.
        \n
        By following these steps, you can help manage and control diarrhea in cattle, minimizing its impact on animal health and farm operations.
        """
          
        treatment = "" 
        if predicted_class == "diarrhea":
           treatment = treatment_diarrhea 
           
        if predicted_class == "diarrhea\n":
           treatment = treatment_diarrhea 
           
        if predicted_class == "foot and mouth":
           treatment = fmd 
        if predicted_class == "foot and mouth\n":
           treatment = treatment_diarrhea 
        
        text_response = treatment
        

        
        disease = predicted_class
        print(predicted_class)
        
        engine = pyttsx3.init()
        audio_filename = str(uuid.uuid4()) + '.mp3'
        audio_path = os.path.join(audio_dir, audio_filename)
        engine.save_to_file(text_response, audio_path)
        #engine.say(text_response)
        #engine.runAndWait()
        print(prediction)

        # Return both text and audio response
        return jsonify({'text': text_response, 'disease': disease})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
