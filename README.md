# Design-and-Development-of-AI-based-Nursing-Robot
To develop a nursing robot that can detect and provide and dispense timely medicines to the patients in need.
## Abstract
Recent events have demonstrated how robotics’ involvement in healthcare and related fields is evolving, with special worries regarding the management and control of the transmission of viruses that can cause pandemics. Such robots are mostly used in hospitals and other similar settings, such as quarantines, to minimise direct human contact and to ensure cleaning, sterilisation, and assistance. As a result, the risk to the lives of medical professionals and doctors participating actively in pandemic management will be reduced. Overall, even though nurse robots have a lot of potential uses and advantages, it’s critical to keep in mind their drawbacks, such the absence of human contact, and make sure they work in tandem with human carers to give patients the best care possible. The system’s goal is to deliver medications based on patient information and established routes. Facial detection can help to identify a particular patient. The robot is designed to perform various tasks related to patient care in healthcare settings such as assisting healthcare professionals in delivering high-quality care to patients, improve patient safety, and increase efficiency in healthcare delivery. Nurse robots can help with tasks such as monitoring vital signs, administering medication, providing physical therapy, and assisting with activities of daily living. The design and creation of a nurse robot utilising a Raspberry Pi are covered in this project. HAAR cascasde and LPBH algorithms are implemented for facial detection and recognition to identify the patient. The robot dispenses medicines based on the identity of the patient. Sentiment analysis using AI, i.e a CNN emotion detection model is implemented in order to estimate the mental state of the patient. These robots enhance efficiency by automating routine tasks, leading to a 30-40% reduction in nurses’ workload. Additionally, nurse robots have the potential to generate cost savings of 15-20% in overall healthcare expenditures. An accuracy of 95.2% is obtained for Haar Cascade and 93.9% for Local Binary Pattern Histogram (LBPH) classifier while training the facial data for recognition and detection. Emotion detection had an accuracy of 63.66%.
## Objectives
* To develop an Automated self-learning Robot for medical treatment and predictive analytics. 
* To equip working of robot through wireless , Internet Of Things (IOT) and Sensor Technologies relevant to Medical Treatment 
* To implement sentiment analysis and online messaging service to improve capability of the robot
## Methodology
The methology of the robot consists of the following steps:<br />
**Dataset Collection**<br />
In the initial step of the project, a systematic approach is taken to assign a unique identification (ID) to each patient. This process involves utilizing a Raspberry Pi camera module to capture a continuous video stream, from which individual frames are extracted. To accurately detect faces within these frames, the Haar Cascade algorithm is employed, leveraging the power of Intel’s Pretrained detector coupled with Rainer’s trained model. By applying this algorithm, the system effectively identifies and localizes faces within the video frames. As a result, a comprehensive dataset is compiled, consisting of 100 images per patient, each associated with their corresponding ID. This dataset serves as a valuable resource for subsequent stages of analysis and recognition. <br />
**Model Training**<br />
The second step uses the dataset from the previous step. It employs the HAAR cascade algorithm for face detection and the algorithm for face recognition.The HAAR detector identifies faces in the images using the detect.Multiscale function. The faces and their corresponding IDs are recorded in separate arrays for each image. These arrays are then used to train the recognizer model, which is saved as a YAML file containing histograms and IDs.<br />
**Facial Recognition**<br />
In the main step, the robot, built with Raspberry Pi, motors, motor drivers, and sensors, begins its movement by comparing the system time with the specified time in the code. Using the L293D motor driver IC, the robot controls its wheels. It follows predefined paths to reach the patient and instructs them to look into the camera. The robot detects faces using the HAAR cascade algorithm and recognizes the patient with the previously trained LPBH model. If recognized, the patient is prompted to place their finger on temperature and heart rate sensors connected to the Raspberry Pi via MCP3008 Analog-to-Digital Converter (ADC). After recording the sensor values, the robot informs the patient and sends a telegram message with their name, temperature, and heart rate values. It opens the door, informs the patient to collect their medicines, and proceeds to the next patient. If the patient is not recognized, an appropriate message is played, and the robot moves on to the next patient.
## Execution
* Loading model and weights: The pre-trained CNN model and its weights are loaded into memory. It can detect the following emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.
* Face Detection: The HAAR cascade algorithm is utilized to detect faces within the image.
* Preprocessing: Converting the images into greyscale format in order to reduce complexity.
* Feed to model and predict emotions: The face images are passed to the model, which generates a probability value for each emotion class. The highest probability indicates the predicted emotion for each detected face.
