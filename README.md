## Increadible_engineers_object_recognition
## Object Recognition project

#Objective/Problem Statement:

To Develop a multipurpose AI for Visually impaired and Blind person.
Moto:
Empowering Independence through Technology.

Target:
    Blind and visually impaired individuals<br>
    Organizations supporting visually impaired persons<br>
    Developers and researchers in assistive technologies<br><br>
Implementation Steps:<br>
1.Problem Understanding and Research<b>
    Identify the Needs: Understand the specific requirements and challenges faced by blind and visually impaired individuals. This could involve consulting with organizations and individuals who are visually impaired.<br>
    Research Existing Solutions: Analyze current assistive technologies to identify gaps and areas for improvement.<br><br>
2. Software Development<br>
    Data Collection and Annotation: Collect and annotate a large dataset of images with various objects commonly encountered by visually impaired individuals.<br>
    Model Selection: Choose an appropriate object detection model (e.g., YOLO, SSD, Faster R-CNN)(in this case we have chosen openCV).<br>
    Training the Model: Train the model on the annotated dataset, ensuring it can accurately detect and identify objects.<br>
    Optimization: Optimize the model for real-time performance on the selected hardware.<br><br>
3. Integration and Testing<br>
    System Integration: Integrate the camera, processing unit, and audio output. Ensure seamless communication between components.<br>
    User Interface: Develop a simple and intuitive user interface, possibly controlled by voice commands.<br>
    Testing: Conduct extensive testing in various environments to ensure reliability and accuracy. Gather feedback from visually impaired users for further refinement.<br><br>
4. Deployment and Maintenance<br>
    Deployment: Distribute the device to target users and provide training on its usage.<br>
    Support and Updates: Offer ongoing support and regular updates to improve functionality based on user feedback.<br>
    Community Building: Create a community of users and developers to share experiences and collaborate on improvements.<br><br>
5. Hardware Selection(yet to be implemented)<br>
    Camera: High-quality camera to capture real-time images.<br>
    Processing Unit: A powerful processor like NVIDIA Jetson Nano, Raspberry Pi, or a mobile phone.<br>
    Power Supply: Portable and long-lasting battery.<br>
    Connectivity: Options like Bluetooth, Wi-Fi for data transmission.<br>
    Audio Output: Speakers or earphones for audio feedback.<br><br>
# Additional Considerations
    Affordability: Ensure the device is cost-effective to make it accessible to a larger population.<br>
    Portability: Design the device to be lightweight and easy to carry.<br>
    User Privacy: Implement strong privacy measures to protect users' data.<br>
    Legal and Ethical Compliance: Ensure the device complies with legal standards and ethical guidelines related to assistive technologies.<br>

## Instructions to run the code<br><br>
Clone the repo (or) download zip file and extract<br>
Install all the required libraries using requirements.txt
```
pip install -r requirements.txt
```

Navigate the command prompt to the folder where codes are stored<br>
Run "launch.py"
```
python launch.py
```

