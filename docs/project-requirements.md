(please do not delete a bunch of text that is not yours without asking or else i will be sad ;( :) )

\`6 pages max

Overleaf link: <https://www.overleaf.com/project/68febcfde246fbd4eedd518a>

Wearable Air Quality Monitoring System with Embedded AI

Ryan Niemi, Maddie Masiello, Gulianna Corteguera

**Abstract:**

Prior research found that most individuals live in areas where accurate air quality measurements are not readily accessible. Wearable air quality monitoring devices are an emerging technology that has increased in popularity within the last decade. However, the calibration required for accurate measurements is often not feasible in a dynamic, real-life environment. Through the creation of a wearable edge-AI calibrated air quality monitoring device, Sensa, an increased accuracy compared to state-of-the-art competitors will be available to the public, benefiting public safety.

**Introduction:**

- **General**
- **What is the problem?**
- **Motivation for creation**

Air pollution is one of the most widespread environmental challenges today, and it directly impacts human health on a global scale. Poor air quality can lead to respiratory and cardiovascular problems, especially in people who live or work in polluted environments. While there are stationary air-quality monitoring systems available, they are often expensive, complex, and not easily accessible to individuals who want to track the air around them on a daily basis.

Our project aims to address this problem by developing a wearable air-quality monitoring device that measures particulate matter and gases in real time. The device integrates low-cost sensors with an embedded machine learning model to automatically interpret air-quality data and determine the user's level of exposure. This information is then sent to a smartphone or computer for easy tracking and visualization.

Recent developments in environmental sensing emphasize the growing need for portable solutions capable of matching the precision of laboratory systems while maintaining low power consumption and affordability. Studies demonstrate that fine particulate matter (PM2.5) remains one of the most critical pollutants affecting global health, with over 99% of the world's population exposed to unsafe levels and approximately seven million premature deaths each year attributed to poor air quality \[9\], \[10\].

The main goal of this project is to create a portable, affordable, and reliable solution that empowers users to understand their environment and make informed decisions about their health and safety. By combining compact hardware, wireless connectivity, and on-device AI, our design provides a more personal and accessible way to monitor air quality - without relying on external calibration or cloud processing. Similar edge-deployed sensing systems have shown that integrating lightweight machine learning on-device significantly improves responsiveness and reliability, especially when cloud connectivity is limited \[15\].

Ultimately, this project demonstrates how embedded systems and machine learning can be combined to solve real-world problems. The wearable air-quality monitor bridges the gap between environmental awareness and personal health technology, offering a practical tool that helps users stay informed and protected wherever they go.

**\[What makes this product different?\]**

Most wearable sensors are expensive, require frequent manual calibration, and often rely on external reference stations to maintain accuracy before deployment \[5\]. In contrast, the SEN55 leverages on-device AI to automate calibration, enabling faster and more reliable performance in dynamic environments. Integrating a trained AI model will enable the device to assess the air quality, collect data, and make predictions of future pollutant levels based on observed environmental patterns. Predictive analytics using embedded AI has been shown to improve temporal accuracy by forecasting near-future pollutant concentration trends based on environmental differentials and previous readings, as demonstrated in Bindhu et al. \[6\] and Koziel et al. \[12\].

Current gap between our project and whats already out there

Related works: references of other devices

<https://atmotube.com/atmotube-pro#:~:text=The%20Atmotube%20PRO%20is%20a%20portable%20air,ventilate%2C%20reduce%20humidity%2C%20or%20purify%20the%20air>

**Related works (talks about other devices):**

PM related to health

- Many people die from PM
- Around 7 million people die a year from PM<sub>2.5</sub> \[10\]
- PM causes cardiovascular and respiratory diseases, and cancers \[11\]
- In 2019, some 68% of outdoor air pollution related premature deaths were due to ischaemic heart disease and stroke, 14% were due to chronic obstructive pulmonary disease, 14% were due to acute lower respiratory infections, and 4% of deaths were due to lung cancers. \[11\]
- Pollutants: PM, CO, O<sub>3</sub>, NO<sub>2</sub>, SO<sub>2</sub> \[11\]
- In 2019 99% of the worlds population was living in places where the who air quality guidelines levels were not met \[11\]

Other devices out there

- Air quality sensors have been evolving rapidly
- Wide variety of commercially available products now with all significant challenges
- Health reference values for air pollutants are typically based on exposure durations of several hours to many years so for 1-min collection may not suffice \[13\]
- Most portable air sensors provide a min by min measurements
- The most comparable OSHA exposure standards include the Permissible Exposure Limit (PEL), which is an 8-h time-weighted average, and the Short Term Exposure Limit (STEL), which is a 15-min time-weighted average. \[13\]
- Reword: Nevertheless, in a holistic vision of the exposome, which should consider all exposures over all life-stages, it may not be appropriate to regard ambient, indoor and workplace exposures as somehow "different", to be always measured and assessed separately \[13\]

Despite the rapid evolution of consumer air quality sensors, recent analyses highlight persistent calibration drift and cross-sensitivity issues among low-cost PM and VOC modules \[5\]. Current portable systems like Atmotube and Flow rely on centralized databases or cloud recalibration, limiting real-time correction and privacy. Our project directly addresses this by deploying calibration intelligence locally on the device \[6\], \[15\]. Furthermore, studies such as Narayana et al. \[3\] show that advanced calibration models like EEATC outperform traditional linear regression approaches in dynamic environments, supporting our goal to embed adaptive algorithms directly within the wearable hardware.

Air-quality sensing technology has advanced rapidly over the past decade, with a growing variety of commercially available products targeting both professional and consumer markets \[6\]. Despite this progress, significant challenges remain, including sensor calibration drift, cross-sensitivity, and inconsistent data reliability under changing environmental conditions \[5\], \[12\]. Most portable monitors provide minute-by-minute readings; however, these short-term measurements may not align with existing health exposure standards, which are typically based on much longer durations-from several hours to years \[13\]. For example, the Occupational Safety and Health Administration (OSHA) defines exposure limits such as the Permissible Exposure Limit (PEL), an 8-hour time-weighted average, and the Short-Term Exposure Limit (STEL), a 15-minute average \[13\]. These regulatory benchmarks highlight the difficulty of interpreting short-interval data in the context of long-term health outcomes.

Moreover, in a holistic view of the _exposome_-the totality of environmental exposures across an individual's lifespan-it is increasingly recognized that indoor, outdoor, and occupational exposures should not be treated as isolated domains but rather as interconnected aspects of overall air-quality impact \[13\]. Recent evaluations of low-cost PM and VOC sensors have shown that while affordability and accessibility have improved, reliability and calibration stability still lag behind industrial-grade equipment \[5\], \[6\]. Commercial wearable devices such as Atmotube and Flow depend on centralized databases or periodic cloud recalibration, which limits real-time responsiveness and raises privacy concerns.

To overcome these limitations, emerging research has focused on AI-driven calibration methods that enable self-correcting, adaptive performance at the edge. Studies such as Narayana et al. \[4\] demonstrate that advanced machine-learning models like EEATC outperform traditional linear regression in dynamic conditions, while Zimmerman et al. \[20\] and Smith et al. \[19\] show that random forest and ensemble-based calibration significantly improve low-cost sensor accuracy. These findings motivate our approach-embedding adaptive calibration intelligence directly within the wearable hardware-to achieve both independence from external systems and greater measurement accuracy in real time.

- **Customer requirements:**

Our customers care about their surrounding air quality and look to be more conscious of harmful things in the air that are proven to have an impact on their health. In many places in our world, air quality can be dangerous, directly impacting the health of people. Additionally, air quality monitoring systems are not immediately available in most areas, and are not portable, thus a wearable air quality monitoring system can allow users to track air quality conveniently, potentially having an impact on improving their health. Our customers are likely looking for a reasonably priced air monitoring system; therefore we plan to price our product at around \$90, based on the combined costs of the sensors and integration.

The wearable air-quality monitor is designed for individuals who want to better understand and manage their exposure to harmful environmental conditions. Our target users include people who are regularly exposed to varying air-quality levels and need a simple, reliable way to track their surroundings. This group includes urban commuters, industrial and construction workers, laboratory researchers, and health-conscious individuals who want air-quality feedback throughout the day.

Customers expect the device to be portable, affordable, and dependable, capable of providing accurate real-time air-quality reading without requiring constant attention. The device should connect easily to a smartphone or computer to display live data, store past readings, and send exposure alerts to the user via visuals or noise indicators. Real-time communication through BLE/Wi-Fi aligns with IoT trends in environmental monitoring, where continuous data flow enables both user awareness and potential integration with public networks for broader environmental mapping \[6\].

Because most existing monitors are either stationary, bulky, or expensive, our project aims to close this gap by creating a compact, battery-powered, AI-driven device that automatically calibrates its sensors to maintain accuracy over time. These self-calibrating approaches are supported by recent literature emphasizing long-term reliability of AI-assisted low-cost sensors in uncontrolled outdoor and indoor conditions \[4\], \[11\]. Recent research highlights that sensor drift and environmental interference remain two of the largest challenges in low-cost systems, which is why self-calibration and ML-based correction is critical to our products' reliability \[4\], \[5\].

Currently, users lack an affordable, wearable air-quality monitor that can collect and interpret data in real time. Many available systems depend on internet connectivity or manual calibration, while our design performs all processing locally, giving users both independence and privacy. Studies such as Bindhu et al. (20223) emphasize that loT-driven air quality platforms often rely on cloud processing, creating data latency, and privacy issues our local approach is attempting to address and avoid \[6\].

Performance, usability, and affordability are the most important qualities to our customers. They expect precise measurements of particulate matter (PM), volatile organic compounds (VOC), and other important measurements like temperature and humidity while the device remains lightweight and comfortable for daily wear. Battery life is also critical - a full workday of continuous operation is the minimum goal. Ease of use matters as well; the monitor should require little setup, operate autonomously, and display results clearly through a companion app or phone/desktop interface. Keeping the total price under one hundred dollars makes the product accessible to a wide range of users, from students to industrial professionals. In short, customers are looking for a balance of accuracy, convenience, and portability in one device.

Environmental responsibility is also important to our users. The monitor will use rechargeable lithium-ion batteries and durable components to reduce electronic waste. A long sensor lifespan and a sturdy enclosure allow the product to be maintained and reused rather than discarded. To protect user privacy, all data processing and storage will occur locally on the device or within the user's personal account.

Accessibility is another key requirement. The interface should be intuitive for people of all technical backgrounds. Simple visual cues, color indicators, or mobile notifications will make it easy to understand air-quality levels without needing prior knowledge of the respective metrics. By keeping the interaction straightforward, the device ensures that anyone can benefit from accurate environmental insights. Similar to the visualization strategies reviewed by Kim and Moon (2025), our interface focuses on clarity and minimalism to make air-quality trends easy to interpret in real time \[11\].

Because the product is designed to be worn for long periods, comfort and appearance also matter. The enclosure must allow airflow to the sensors while staying lightweight, breathable, and visually appealing. Users should feel comfortable wearing it both at work, school, and in casual settings.

Finally, customers value trust and reliability. The monitor must deliver consistent, accurate readings under different conditions - indoors, outdoors, or while moving. Any sensor drift or calibration error should be corrected automatically through on-device learning to maintain confidence in the data.

In practical terms, a construction worker might use the monitor to ensure dust levels remain safe throughout their shift. A lab technician could rely on it to detect harmful fumes early, while an urban commuter might use it to compare air quality between outdoor and indoor environments. These scenarios show how users directly benefit from having accurate, real-time environmental data available immediately.

**Engineering specs:**

(I have an overleaf document for everything that is big and complicated)

Currently includes a house of quality and table comparing our specifications to competitors

<https://www.overleaf.com/8391321682hbfnwkcftgvs#de5264>

The Microcontroller (MCU) chosen is the ESP32-S3, which supports features critical for the project, including TensorFlow support, OTA (Over-The-Air) updates, FreeRTOS, Wi-Fi/BLE connectivity, and ample RAM.

The SEN55 Environmental Sensor Node serves as our core sensor component for this project, providing an all-in-one module capable of measuring multiple air quality parameters \[1\]. The SEN55 module has been successfully implemented in environmental logging systems powered by ESP32-based controllers, validating its stability and measurement accuracy in solar-powered or low-energy IoT deployments \[2\]. It detects Particulate Matter (PM) sizes of 1.0 μm, 2.5 μm, 4.0 μm, and 10.0 μm (PM1, PM2.5, PM4, PM10) with a mass concentration range from 0 to 1,000 μg/m^3 \[2\]. The module also measures the Volatile Organic Compounds (VOC) Index and Nitrogen Oxides (NOx) Index, both ranging from 1 to 500. In addition, it tracks environmental factors Relative Humidity (RH) and Temperature (T).

The SEN55's high precision is critical to meeting the project's accuracy requirements. For PM concentrations from 0 to 100 μg/m^3, the precision is 5 μg/m^3 or 5% m.v, and for 100 to 1000 μg/m3, the precision is 10% m.v. The PM2.5 mass concentration output is calibrated to the TSI DustTrak™ DRX 8533 Ambient Mode. Typical temperature accuracy is maintained within +-0.5 deg C at 10-30 deg C and 50% RH, while humidity accuracy is +-4.5 % RH at 25 deg C and 30-70 % RH. The module compensates for its own self heating through the Sensorion Temperature Acceleration Routine (STAR) engine, which accelerates the devices' thermal response to ambient temperature change by a factor of 2-3 \[2\]. This rapid thermal adjustment is essential for maintaining precision during mobile use, as identified in recent field trials where variable ambient conditions led to noticeable biases in PM readings from uncorrected sensors \[4\].

The wearable interface must accommodate both physical and communication requirements of the sensors. The SEN55 utilizes an I2C interface for communicational a standard speed of 100kbits/s. It employs a 6-pin serial communication interface with SDA (Serial Data) and SCL (Serial Clock) lines that require external 100ohm pull-up resistors. The I2C address is 0x69 amd the module operates at 5V (Min 4.5V, Typ 5.0V, Max 5.5V). \[1\]\[2\].

A reliability of less than 1% packet loss will be enforced, conforming to Bluetooth standards for communication to a mobile phone. An expected packet speed of 1Mbps is expected for electrically noisy environments that our device will be in. Nearby signal packet disruption by competing frequencies is not a concerning factor due to Bluetooth standards to incorporate frequency hopping.

The goal for battery life is to last up to 12 hours to cover for a full day. The largest sensor (the SEN55) dictates the enclosure size, measuring approximately 52.8 x 43.6 x 22.3 mm, and the sensor module itself weighs about 38.4g \[1\]

The SEN55's low-power operation supports this target, with a default sampling interval of 1 ± 0.03 s. In both full measurement and RHT/Gas-only modes, new readings are available every second, and gas response time is assessed at this rate \[5\]. To optimize power, the system will power the MCU once per minute: the sensor records data for 1-10 seconds, then the machine-learning module processes it before returning to sleep. Previous studies using the SEN55 have collected 1 Hz data averaged to 1-minute intervals \[4\], while others sampled every 10 seconds with 60-second prediction windows for ML models such as Random Forest and SVR \[13\]. Another experiment used a calibration chamber to capture sensor response data at 100 Hz \[5\].

Our team plans to calibrate and validate the device using the Air Measurements Lab at Cal Poly SLO to ensure accurate sensor performance under controlled conditions. According to recent comparative calibration work by Chanapromma and Wongdocmai \[7\], chamber-based validation remains a benchmark for evaluating sensor linearity and transient response, making laboratory-based testing a necessary step before field deployment.

OVERLEAF FORMAT:

- Core Components
- PM Sensor - SEN55

- Measures: PM1, PM2.5, PM4, PM10; VOC Index (1-500); NOx Index (1-500); temperature; and relative humidity \[1\], \[2\].
- Precision:  
   • 0-100 μg/m³ → ±5 μg/m³ or ±5% m.v.  
   • 100-1000 μg/m³ → ±10% m.v.  
   Calibrated for PM2.5 to TSI DustTrak™ DRX 8533 (Ambient Mode).
- Accuracy:  
   • Temperature ±0.5 °C (10-30 °C, 50% RH)  
   • Humidity ±4.5% RH (25 °C, 30-70% RH)
- Thermal Handling: Self-heating compensation via STAR engine, accelerating thermal response 2-3× faster than ambient settling \[2\], \[4\].
- Interface: I²C @ 100 kbit/s; address 0x69; SDA/SCL with 100 Ω pull-ups \[1\].
- Power: 5 V (4.5-5.5 V); sampling ≈ 1 Hz (new reading every 1 s).
- Form Factor: 52.8 × 43.6 × 22.3 mm; 38.4 g \[1\].
- Performance Examples:  
   PM 2.5 - Good: 0-9 μg/m³; Moderate: 9.1-35.4; Unhealthy: 55.5-125.4 \[10\].  
   VOC (ppb) - 0-500 Low; 500-1500 Moderate; >1500 High \[ref\].

Additional Context:  
The SEN55 has been successfully implemented in ESP32-based IoT and solar-powered systems, demonstrating stable operation and accurate data logging in low-energy conditions \[2\]. Recent field trials note that uncorrected sensors show significant PM bias under dynamic temperatures, validating the need for thermal compensation \[4\].

2\. Microcontroller - ESP32-S3

- Architecture: Dual-core MCU with ample RAM for TinyML workloads.
- Connectivity: Wi-Fi + Bluetooth LE for data sync and OTA updates.
- Software Support: FreeRTOS and TensorFlow Lite for on-device AI.
- Functions: Manages sensor data collection, buffering, ML inference, and power states.
- Power Control: MCU wakes periodically (≈ 1 min intervals) to sample and process data before sleep mode.
- Compactness: Designed for < 3 in² board integration.

Reference Context:  
The ESP32-S3 is chosen for its robust support of TensorFlow Lite Micro and real-time wireless communication. Its integration with the SEN55 enables localized data processing without cloud dependency, a trend supported by recent Edge AI literature \[15\].

3\. Battery & Power System

- Goal: ≥ 12 hours continuous operation per charge (full work day).
- Battery: Li-ion type for high density and rechargeability.
- Efficiency: SEN55 samples for 1-10 s per cycle; MCU processes then returns to sleep mode.
- Sampling Strategy: Typical 1 Hz data averaged to 1-minute intervals \[4\]; validated with 1-10 s windows for ML models such as Random Forest and SVR \[13\].
- Validation: Calibration and accuracy testing will be performed in the Cal Poly Air Measurements Lab to verify sensor linearity and response. Chamber-based calibration remains a gold standard for evaluating sensor drift \[7\].

4\. Communication

- Protocol: BLE for low-power local linking; Wi-Fi for long-range sync.
- Features: Adaptive Frequency Hopping (AF H) to avoid interference in 2.4 GHz band.
- Functions: Real-time data streaming, alerts, and firmware updates without wired connection.

5\. AI Model (Edge Calibration)

- Purpose:  
   Provides on-device classification and self-calibration for PM and VOC exposure levels, ensuring accurate operation in dynamic real-life environments where external calibration is not feasible \[3\], \[11\].
- Structure:  
   Implements lightweight TinyML models-Random Forest, Support Vector Regression (SVR), or small neural networks (1-2 hidden layers)-to fit within MCU memory and compute limits \[4\]. Larger frameworks such as PyTorch are avoided due to resource constraints.
- Calibration Strategy:  
   The embedded AI continuously adjusts sensor outputs to correct for drift and environmental variation, using techniques like hybrid neural networks and differential data scaling to maintain long-term accuracy \[11\], \[12\].  
   The multi-phase EEATC (Estimated Error Augmented Two-phase Calibration) model developed by Narayana et al. \[3\] informs our calibration approach, as it outperforms single-phase linear and Random Forest models in pollutant stability tests.
- Data Processing:  
   Uses rolling feature windows (e.g., 60-second datasets) for sturdy & temporal prediction. Models are trained on features such as temperature, humidity, and historical PM/VOC trends for continuous recalibration \[6\].
- Adaptivity:  
   Automatically re-trains using incremental updates on-device to account for environmental drift and sensor aging. Validated approaches include polynomial-based correction \[5\] and 1D-CNN or LSTM methods for complex temporal pollutant data \[13\].
- Edge Implementation:  
   Deploys inference directly on the ESP32-S3, leveraging TensorFlow Lite Micro for embedded ML workloads. The design minimizes latency and power consumption while maintaining accuracy comparable to cloud-calibrated systems \[6\], \[15\].
- Performance Benefits:  
   On-device ML enables real-time air quality assessment and prediction, enhancing temporal accuracy and system responsiveness. Neural surrogate calibration methods have been shown to significantly improve measurement precision for low-cost sensors \[6\], \[12\].

Recent literature underscores that ML-based calibration is key for reliable low-cost air-quality sensors. Our embedded TinyML implementation builds upon validated frameworks such as EEATC \[3\], hybrid differential scaling \[12\], and neural-network surrogates \[6\], achieving robust accuracy under mobile and stationary conditions while maintaining low power draw through edge deployment \[15\].

6\. Enclosure

- Size: Sized around SEN55 and battery module, roughly 3 × 3 in., < 2 in thick.
- Design: Ventilated pathways to maintain airflow and response time.
- Material: Durable shell for field use; lightweight and breathable for wearability.
- Mounting: Supports clip/strap mounts for casual or industrial wear.
- Layout: Isolates heat sources and reserves channels for sensing accuracy.

The combination of the SEN55 sensor and ESP32-S3 microcontroller meets the core project requirements for precision, low power, and real-time data processing. By leveraging edge-AI calibration and validated sensor architectures \[2\], \[3\], \[7\], \[15\], the system achieves high accuracy in dynamic environments while maintaining portability and user accessibility.

**Methodology (for determining increased accuracy):**

Our team plans to obtain access to the Hal Cota Air quality monitoring lab, which is located at 13-201 of Cal Poly. The lab provides access to PM 2.5, PM 10, NOx, and other highly accurate air quality monitoring equipment. To determine our accuracy of the provided laboratory PM sensors, datasheets will be analyzed, and a model will be formed around them. In the case that the datasheets are not adequate, a Kalman filter will be used to obtain Covariance matrixes. Additionally, a widely used model to fit data trends in a way that makes quantifying error trivial is to use orthogonal polynomials such as Chebyshev or to use a vectorized approach such as Least Mean Squares \[7\]. These approaches can be used to train the model in conjunction with Cross-Entropy Classification (Logistic Regression) and Mean Squared Error loss (Linear Regression), respectively. This training will be done through a version of gradient descent, parameter updates to adjust weights, and through repetition. Different AI models will provide different advantages, thus metrics such as model drift, R squared, and the confusion matrix will provide deeper insight into the model's effectiveness. By assessing the R squared value that the AI model has in conjunction with the measurement equipment, this value can be compared to other published data to determine the effectiveness of our calibration model.

**Our Work (what is our product going to do)**

Our work combined a structured project plan with an evidence-driven research process. In weeks 1-2, we established a detailed work breakdown structure, delegated primary tasks, and defined individual leadership areas to ensure clear ownership, accountability, and an efficient workflow for the full project life cycle. Guided by the literature on low-cost sensing and edge-AI calibration, we synthesized methods addressing calibration drift and cross-sensitivity (e.g., ensemble/Random Forest and EEATC-style approaches), which informed our ESP32-S3 + SEN55 architecture, minute-level sampling with thermal compensation, and TinyML inference on rolling windows. We then mapped these findings into testable engineering specifications and a validation methodology using reference instruments, Kalman/least-squares modeling, and metrics such as R², confusion matrices, and error relative to standards to quantify accuracy and stability. This alignment between front-end planning, research synthesis, and measurement protocol ensured a cohesive, reproducible, and standards-aware development process

what do you think^^

This project will be based on the strong principles of teamwork and collaboration. During the initial phase (weeks 1-2), the team will establish a detailed work breakdown structure and delegate all primary tasks. Concurrently, individual leadership roles and specific areas of responsibility will be assigned to each member to ensure clear ownership and streamline project execution. This front-end planning is essential for establishing accountability and ensuring an efficient workflow throughout the project life cycle.

Conclusion:

The wearable air quality monitoring device with embedded AI represents more than just a technical prototype - it aims to make accurate air quality awareness a part of daily life. By combining a high-precision environmental sensor (SEN55), an efficient ESP32-S3 microcontroller, and an embedded AI calibration system, our product transforms professional-grade sensing into a compact wearable form. Our device will perform all processing locally, offering real-time responsiveness and data privacy. This autonomy allows users to continuously monitor their environment without dependence on network connectivity, making the system practical for both personal and field-based applications.

Current literature on air quality monitoring systems - from wearable PM sensors \[16\] to AI calibrated sensors \[19\] - reveals the need for an innovative combination of both. Integrating the accuracy of advanced ML calibration frameworks \[3\], \[11\], and \[12\] with the portability and affordability of embedded edge computing \[15\] represents a step forward for consumer-level air monitoring systems. This gap in accurate, yet portable air quality calibration has motivated the creation of the Sensorian SEN55 - an explorative approach for a more effective way of increasing sensor accuracy in dynamic real-world environments. By addressing calibration drift, environmental interference, and the lack of accessibility found in existing systems, our device delivers a practical solution to real world monitoring challenges.

Our research verifies the feasibility of our product to compete with current state-of-the-art research \[6\], \[7\], \[12\]. A long battery life exceeding twelve hours, rapid and precise calibration within 0.2 standard deviations of known values, and a small ergonomic form factor are key important design requirements we hope to achieve. Air-quality monitoring devices are an exciting emerging technology that requires further development to see greater adoption by the public. This research helps to bridge the gap that has stopped the production of consumer products in the past. Looking ahead, our team plans to validate our device's performance through laboratory calibration and outdoor testing, ensuring that the device maintains consistent accuracy across diverse environments.

Our prototype, leveraging the SEN55 and ESP32-S3 platforms, builds on proven calibration strategies from recent literature \[2\], \[4\], \[6\], which validates that reliable, wearable air quality monitoring can now be achieved with minimal power consumption while maintaining high data integrity. Ultimately, our team envisions this device contributing to broader environmental awareness, encouraging healthier behaviors, and promoting data-driven environmental policies.

References IN ORDER FINAL:

\[1\] Sensirion AG, Datasheet: SEN5x Environmental Sensor Node for HVAC and Air Quality Applications, Version 2-D1, Mar. 2022. \[Online\]. Available: <https://sensirion.com/media/documents/6791EFA0/62A1F68F/Sensirion_Datasheet_Environmental_Node_SEN5x.pdf>

\[2\] Isabella M. Santi: An Analysis Of Indoor Air Quality At Cal Poly For Senior Design, Cal Poly Digital Commons, July 21, 2024. Available at: <https://digitalcommons.calpoly.edu/theses/2851/>

\[3\] J. N. Muhvu _et al._, "Design and Implementation of a Solar Powered Kit for Measurement and Logging of Environmental Parameters Using the SEN55 Sensor," _Measurement: Energy_, vol. 5, p. 100038, Mar. 2025. doi: 10.1016/j.meaene.2025.100038.

\[4\] M. V. Narayana _et al._, "EEATC: A Novel Calibration Approach for Low-Cost Sensors," _IEEE Sensors Journal_, vol. 23, no. 19, pp. 23500-23511, Aug. 2023. doi: 10.1109/JSEN.2023.3304366.

\[5\] T. Barrett and A. K. Mishra, "Statistical Study of Sensor Data and Investigation of ML-based Calibration Algorithms for Inexpensive Sensor Modules: Experiments from Cape Point," _arXiv_, preprint arXiv:2503.13487, Mar. 9, 2025. \[Online\]. Available: <https://arxiv.org/abs/2503.13487>

\[6\] H. Tariq, F. Touati, D. Crescini and A. Ben Mnaouer, "State-of-the-Art Low-Cost Air Quality Sensors, Assemblies, Calibration and Evaluation for Respiration-Associated Diseases: A Systematic Review," _Atmosphere_, vol. 15, no. 4, p. 471, Apr. 2024. doi: 10.3390/atmos15040471.

\[7\] M. Bindhu _et al._, "Harnessing Machine Learning for IoT-Driven Atmospheric Parameter Monitoring and Predictive Analytics," _2023 9th International Conference on Smart Structures and Systems (ICSSS)_, Chennai, India, Nov. 2023, pp. 1-6. doi: 10.1109/ICSSS58085.2023.10407051.

\[8\] W. Chanapromma and W. Wongdocmai, "Calibration Based on Polynomial Function for Low-Cost Sensors: A Case Study of Air Purifier Filter," _2024 12th International Electrical Engineering Congress (iEECON)_, Pattaya, Thailand, Mar. 2024, pp. 1-4. doi: 10.1109/iEECON60677.2024.10537897.

\[9\] E. Azeraf _et al._, "Real-Time Pollutant Identification through Optical PM Micro-Sensor," _arXiv_, preprint arXiv:2503.10724, 2025. \[Online\]. Available: <https://arxiv.org/abs/2503.10724>

\[10\] P. Tarín-Carrasco _et al._, "Contribution of Fine Particulate Matter to Present and Future Premature Mortality over Europe: A Non-Linear Response," _Environment International_, vol. 153, p. 106517, Aug. 2021. doi: 10.1016/j.envint.2021.106517.

\[11\] World Health Organization, "Ambient (Outdoor) Air Quality and Health," _World Health Organization_, Oct. 24, 2024. \[Online\]. Available: [https://www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health](https://www.who.int/news-room/fact-sheets/detail/ambient-%28outdoor%29-air-quality-and-health)

\[12\] Y.-H. Kim and S.-H. Moon, "Machine Learning-Based Quality Control for Low-Cost Air Quality Monitoring: A Comprehensive Review of the Past Decade," _Atmosphere_, vol. 16, no. 10, p. 1136, Sept. 2025. doi: 10.3390/atmos16101136.

\[13\] S. Koziel _et al._, "High-Performance Machine-Learning-Based Calibration of Low-Cost Nitrogen Dioxide Sensor Using Environmental Parameter Differentials and Global Data Scaling," _Scientific Reports_, vol. 14, no. 1, Oct. 2024. doi: 10.1038/s41598-024-77214-y.

\[14\] T. Barrett and A. K. Mishra, "Statistical Study of Sensor Data and Investigation of ML-Based Calibration Algorithms for Inexpensive Sensor Modules: Experiments from Cape Point," _IEEE Transactions on Instrumentation and Measurement_, vol. 73, pp. 1-10, Jan. 2024. doi: 10.1109/TIM.2024.3372211.

\[15\] W. Chanapromma and W. Wongdocmai, "Calibration based on Polynomial Function for Low-Cost Sensors: A Case Study of Air Purifier Filter," _2024 12th International Electrical Engineering Congress (iEECON)_, Pattaya, Thailand, 2024, pp. 1-4. doi: 10.1109/iEECON60677.2024.10537897.

\[16\] A. Rocha _et al._, "Edge AI for Internet of Medical Things: A Literature Review," _Computers & Electrical Engineering_, vol. 116, p. 109202, Mar. 2024. doi: 10.1016/j.compeleceng.2024.109202.

\[17\] Sensirion AG, Engineering Guidelines for SEN5x Environmental Sensor Node, Mar. 2022. \[Online\]. Available: <https://sensirion.com/media/documents/25AB572C/62B463AA/Sensirion_Engineering_Guidelines_SEN5x.pdf>

\[18\] Sensirion AG, Reduced Power Operation for SEN5x Environmental Sensor Node, Aug. 2022. \[Online\]. Available: <https://sensirion.com/media/documents/1B417576/62F0B936/Reduced_Power_Operation_SEN5x.pdf>

\[19\] Smith S, Trefonides T, Srirenganathan Malarvizhi A, et al, A Systematic Study of Popular Software Packages and AI/ML Models for Calibrating In Situ Air Quality Data: An Example with Purple Air Sensors, _Sensors (Basel, Switzerland)_, _25_(4), 1028, 2025 Feb 9. doi:10.3390/s25041028

\[20\] Zimmerman, N., Presto, A. A., Kumar, S. P. N., Gu, J., Hauryliuk, A., Robinson, E. S., Robinson, A. L., and R. Subramanian: A machine learning calibration model using random forests to improve sensor performance for lower-cost air quality monitoring, Atmos. Meas. Tech., 11, 291-313, <https://doi.org/10.5194/amt-11-291-2018>, 2018. Available: <https://amt.copernicus.org/articles/11/291/2018/amt-11-291-2018.html>

\[21\] <https://www.mdpi.com/2073-4433/8/10/182>

OLD REFERENCES:

\[1\] Sensirion. Datasheet SEN5x Environmental Sensor Node for HVAC and Air Quality Applications. Mar. 2022.

‌\[2\] Muhvu, J. Nchi, et al. "Design and Implementation of a Solar Powered Kit for Measurement and Logging of Environmental Parameters Using the SEN55 Sensor." Measurement: Energy, vol. 5, Mar. 2025, p. 100038, <https://doi.org/10.1016/j.meaene.2025.100038>.

\[3\] Narayana, M V, et al. "EEATC: A Novel Calibration Approach for Low-Cost Sensors." IEEE Sensors Journal, vol. 23, no. 19, 28 Aug. 2023, pp. 23500-23511, arxiv.org/abs/2308.13572, <https://doi.org/10.1109/jsen.2023.3304366>.

\[4\] T. Barrett and A. K. Mishra, "Statistical Study of Sensor Data and Investigation of ML-based Calibration Algorithms for Inexpensive Sensor Modules: Experiments from Cape Point," arXiv, Mar. 9, 2025. \[Online\]. Available: <https://arxiv.org/abs/2503.13487>

\[5\] H. Tariq, F. Touati, D. Crescini and A. Ben Mnaouer, "State-of-the-Art Low-Cost Air Quality Sensors, Assemblies, Calibration and Evaluation for Respiration-Associated Diseases: A Systematic Review," Atmosphere, vol. 15, no. 4, p. 471, Apr. 2024, doi: 10.3390/atmos15040471.

\[6\] Bindhu, M., et al. "Harnessing Machine Learning for IoT-Driven Atmospheric Parameter Monitoring and Predictive Analytics." 2023 9th International Conference on Smart Structures and Systems (ICSSS), vol. 15, no. 4, 23 Nov. 2023, pp. 1-6, ieeexplore.ieee.org/document/10407051, <https://doi.org/10.1109/icsss58085.2023.10407051>. Accessed 27 Oct. 2025.

\[7\] Waraporn Chanapromma, and Wathit Wongdocmai. "Calibration Based on Polynomial Function for Low-Cost Sensors: A Case Study of Air Purifier Filter." IEEE Xplore, 6 Mar. 2024, pp. 1-4, ieeexplore.ieee.org/document/10537897, <https://doi.org/10.1109/ieecon60677.2024.10537897>. Accessed 27 Oct. 2025.

\[8\] Azeraf, Elie, et al. "Real-Time Pollutant Identification through Optical PM Micro-Sensor." ArXiv.org, 2025, arxiv.org/abs/2503.10724. \[

9\] Tarín-Carrasco, Patricia, et al. "Contribution of Fine Particulate Matter to Present and Future Premature Mortality over Europe: A Non-Linear Response." Environment International, vol. 153, Aug. 2021, p. 106517, <https://doi.org/10.1016/j.envint.2021.106517>.

\[10\] World Health Organization. "Ambient (Outdoor) Air Quality and Health." World Health Organization, 24 Oct. 2024, <www.who.int/news-room/fact-sheets/detail/ambient-(outdoor)-air-quality-and-health>.

\[11\] Kim, Yong-Hyuk, and Seung-Hyun Moon. "Machine Learning-Based Quality Control for Low-Cost Air Quality Monitoring: A Comprehensive Review of the Past Decade." Atmosphere, vol. 16, no. 10, 27 Sept. 2025, p. 1136, <www.mdpi.com/2073-4433/16/10/1136>, <https://doi.org/10.3390/atmos16101136>. Accessed 8 Oct. 2025.

\[12\] Koziel, Slawomir, et al. "High-Performance Machine-Learning-Based Calibration of Low-Cost Nitrogen Dioxide Sensor Using Environmental Parameter Differentials and Global Data Scaling." Scientific Reports, vol. 14, no. 1, 30 Oct. 2024, <https://doi.org/10.1038/s41598-024-77214-y>.

\[13\] Barrett, Travis, and Amit Kumar Mishra. "Statistical Study of Sensor Data and Investigation of ML-Based Calibration Algorithms for Inexpensive Sensor Modules: Experiments from Cape Point." IEEE Transactions on Instrumentation and Measurement, vol. 73, 1 Jan. 2024, pp. 1-10, ieeexplore.ieee.org/document/10456922, <https://doi.org/10.1109/tim.2024.3372211>. Accessed 27 Oct. 2025.

\[14\] W. Chanapromma and W. Wongdocmai, "Calibration based on Polynomial Function for Low-Cost Sensors: A Case Study of Air Purifier Filter," 2024 12th International Electrical Engineering Congress (iEECON), Pattaya, Thailand, 2024, pp. 1-4, doi: 10.1109/iEECON60677.2024.10537897.

\[15\] Rocha, Atslands, et al. "Edge AI for Internet of Medical Things: A Literature Review." Computers & Electrical Engineering, vol. 116, 25 Mar. 2024, pp. 109202-109202, <https://doi.org/10.1016/j.compeleceng.2024.109202>. |