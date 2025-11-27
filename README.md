# ☕ Ceyloncoffee
### *AI-Powered Decision Support for Sri Lankan Coffee Industry*

This project is a multi component AI system designed to improve coffee farming, processing, logistics, and export decision making in Sri Lanka. Each module targets a specific real world challenge from plant diseases to price prediction and resource allocation.

---

## 📌 Project Components Overview

---

## 1️⃣ Coffee Plant Disease Detection & Yield Prediction  
**Owner:** Aloka A.M.R.V

### 🔍 Key Components
- Disease classification model  
- Yield prediction model  
- User-friendly frontend

### 🎯 Objectives
- Identify coffee plant diseases using image classification  
- Predict production yield per plant or plot  
- Scale from small-farm analysis to plantation-level forecasting  

---

## 2️⃣ Coffee Price & Demand Prediction  
**Owner:** Jayalath M.D.T.L

### 🔍 Key Components
- Price prediction model  
- Demand prediction model  
- AI advisory module for exporters  

### 🎯 Objectives
- Predict global and Sri Lankan coffee prices (short & medium term)  
- Forecast domestic and export demand  
- Use economic, weather, and market data for predictions  
- Assist farmers, exporters, and policymakers  

---

## 3️⃣ Predictive Resource Allocation (Labor & Transportation)  
**Owner:** Rumalya

### 🔍 Key Components
- Labor forecasting model  
- Transportation demand prediction model  
- Scheduling & optimization dashboard  

### 🎯 Objectives
- Predict required workers for harvesting, loading, etc.  
- Estimate vehicles needed for bean transport (daily → seasonal)  
- Reduce extra cost & waiting time  
- Connect with yield/weather data for improved accuracy  
- Provide dashboards with schedules, costs, and alerts  

---

## 4️⃣ Coffee Bean Type & Grade Identification  
**Owner:** S.S. Liyanage

### 🔍 Key Components
- Coffee bean image dataset (Sri Lankan varieties)  
- CNN-based image classification model  
- Web/mobile app for prediction  

### 🎯 Objectives
- Identify bean type (Arabica, Robusta)  
- Identify grade based on defects, size, color  
- Help farmers & exporters ensure quality  
- Support Sri Lankan beans in meeting global quality standards  
 
---

## 🧠 Technologies Used
- Python  
- TensorFlow / PyTorch  
- Machine Learning (Regression, Classification)  
- Computer Vision (CNNs)  
- Data Preprocessing  
- Web Frontend  
- Scheduling algorithms  

---

## 📈 Why This Project Matters
Sri Lanka’s coffee industry faces challenges in:
- Disease monitoring  
- Yield estimation  
- Market forecasting  
- Labor and transportation planning  
- Bean grading and quality assurance  

This system uses AI to modernize the entire supply chain and improve export competitiveness.

---

## 🚀 Future Enhancements
- IoT integration (weather, soil sensors)  
- Real-time dashboards  
- Drone-based imaging  
- Blockchain for supply chain tracking  
- Regional multi-farm integration  

---

## 📝 Contributors
| Name | Component |
|------|-----------|
| **Aloka A.M.R.V** | Disease Detection & Yield Prediction |
| **Jayalath M.D.T.L** | Price & Demand Prediction |
| **Rumalya** | Resource Allocation (Labor & Transport) |
| **S.S. Liyanage** | Bean Type & Grade Identification |

---
flowchart TD

    A[Coffee Supply Chain Intelligence System]

    %% Main Components
    A --> B1[1. Disease Detection & Yield Prediction\nOwner: Aloka]
    A --> B2[2. Price & Demand Prediction\nOwner: Jayalath]
    A --> B3[3. Labor & Transport Allocation\nOwner: Rumalya]
    A --> B4[4. Bean Type & Grade Identification\nOwner: S.S. Liyanage]

    %% Details of Component 1
    B1 --> C1A[Disease Classification Model]
    B1 --> C1B[Yield Prediction Model]
    B1 --> C1C[User-Friendly Frontend]
    B1 -.-> L1[Limitations:\n• Physical Damage\n• Fragmented Supply Chain\n• Moisture/Heat Issues\n• Inconsistent Quality]

    %% Details of Component 2
    B2 --> C2A[Price Prediction Model]
    B2 --> C2B[Demand Prediction Model]
    B2 --> C2C[AI Advisory Module]
    B2 -.-> L2[Limitations:\n• Data Availability\n• Data Quality Issues\n• Unpredictable Global Events]

    %% Details of Component 3
    B3 --> C3A[Labor Forecast Model]
    B3 --> C3B[Transport Demand Model]
    B3 --> C3C[Scheduling Dashboard]
    B3 -.-> L3[Limitations:\n• Missing Data\n• Weather Changes\n• Worker Variability\n• Vehicle Limits]

    %% Details of Component 4
    B4 --> C4A[Image Dataset of Beans]
    B4 --> C4B[CNN Classification Model]
    B4 --> C4C[Web/Mobile Interface]
    B4 -.-> L4[Limitations:\n• Regional Differences\n• Dataset Scaling Issues]

    %% Technologies
    A --> T[Technologies:\nPython, ML, CNNs, Forecasting, Dashboards]

---

<p align="center">
<img src="https://github.com/Thiwanka49/Ceyloncofee-Research/blob/main/550847209_671150316017200_6014927099350871654_n.jpg" alt="Ceylon Coffee Logo" width="300"></center>
</p>

---



