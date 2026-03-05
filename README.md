# ☕ Ceyloncoffee
### *AI POWERED COFFEE QUALITY GRADING & EXPORT ENHANCEMENT SYSTEM FOR SRI LANKA*

This project presents an integrated, intelligent decision support system designed to modernize and optimize the coffee value chain, with a specific focus on Sri Lankan coffee production and export markets. The system combines artificial intelligence, machine learning, and data driven forecasting to address challenges faced by coffee farmers, exporters, logistics managers, and policymakers. By covering the entire lifecycle of coffee production from plant health and yield estimation to price forecasting, labor planning, and transportation optimization the project aims to improve productivity, reduce losses, and increase global competitiveness.

The first major component of the project focuses on Coffee Plant Disease Detection and Production Estimation. Using image based classification models, the system can automatically identify common coffee plant diseases such as Leaf Rust and Cercospora Leaf Spot from images of leaves and beans. Early detection enables farmers to take timely preventive action, minimizing crop damage and financial loss. Alongside disease detection, a yield prediction model estimates coffee production in kilograms per plant or plot, even under disease stress. This component is designed to scale from individual plants to entire plantations and can be integrated into agricultural advisory platforms, providing actionable insights for better farm level decision making.

The second component addresses Coffee Price and Demand Prediction at both domestic and global levels. By analyzing macroeconomic indicators, weather patterns, and market trends, the system forecasts short  and medium term coffee prices and demand. This allows exporters, traders, and policymakers to plan production, storage, and sales strategies more effectively. An AI-based advisory module translates complex predictions into practical recommendations, helping stakeholders respond proactively to market volatility. Despite challenges such as data availability and unpredictable global events, this component strengthens strategic planning and risk management across the coffee industry.

The third component used to create the decisions for calculating fertilizer requirement takes into account many parameters: coffee type; plot size measured in perches; number of plants per acre; nutrient availability levels of soil, such as Nitrogen, Phosphorus, Potassium; the growth stage of the crop; and any diseases that are present in the crop, as well as their relative severity; patterns in rainfall; temperature; and anticipated yield. The data collected then has been analyzed using a supervised regression-based approach, in order to derive an appropriate estimate of the amount of fertilizer per acreage required. There are many differing variables which can affect how the soil will react to the crop’s demand and how many inputs/outputs will be generated from that reaction, so ensemble learning methods have been used to represent the wide range of interactions between the condition of the soil, the crops’ demand from nutrients, and the yield targets. Since we also wanted to account for how the disease severity may affect the crops were displayed using a normalized, 0-1 scale. In addition, we used environmental conditions (i.e. rainfall, humidity, and temperature) to adjust the severity results for any given cycle so that we could accurately determine how much pesticide to apply. During harvest periods, the calculation of anticipated amount of crop to harvest is calculated. 

The fourth component is designed to improve the quality assessment process of coffee beans using advanced artificial intelligence technologies. The system utilizes a coffee bean image dataset consisting of Sri Lankan coffee varieties and applies a Convolutional Neural Network (CNN)-based image classification model to accurately identify different bean types and quality grades. By integrating an IoT device for real-time bean identification and grading, the system can automatically analyze characteristics such as defects, size, and color of the beans. The main objective of this solution is to identify coffee bean types such as Arabica and Robusta and determine their grade based on physical attributes, helping farmers and exporters maintain consistent quality standards. The implementation uses technologies such as Python, MobileNetV3 with PyTorch, machine learning techniques including regression and classification, and computer vision algorithms to process and analyze the data efficiently. Additionally, data preprocessing and scheduling algorithms are used to optimize the processing workflow. Overall, this system supports Sri Lankan coffee producers in meeting global quality standards and improving competitiveness in the international coffee market.


---
# System Overview Diagram
<p>
<img src="https://github.com/Thiwanka49/Ceyloncoffee-Research/blob/main/Untitled%20Diagram.drawio.png" alt="Main diagram" width="1200"">
</p>

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
- Scale from small farm analysis to plantation-level forecasting  

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

## 3️⃣AI Based Fertilizer, Pest Control, and Harvest Resource Planning System
**Owner:** Rumalya

### 🔍 Key Components
- Multi parameter soil & crop data collection module 
- Disease severity normalization system  
- Harvest yield prediction and resource planning module

### 🎯 Objectives
-Calculate precise fertilizer requirements based on soil nutrients and crop demand 
- Optimize pesticide usage by adjusting disease severity with environmental conditions
- Predict harvest yield during harvesting periods for better resource allocation
- Reduce input costs while maximizing crop productivity
- Support sustainable and data-driven coffee cultivation in Sri Lanka
---

## 4️⃣ Coffee Bean Type & Grade Identification  
**Owner:** S.S. Liyanage

### 🔍 Key Components
- Coffee bean image dataset (Sri Lankan varieties)  
- CNN-based image classification model  
- IOT device for bean identification & grading  

### 🎯 Objectives
- Identify bean type (Arabica, Robusta)  
- Identify grade based on defects, size, color  
- Help farmers & exporters ensure quality  
- Support Sri Lankan beans in meeting global quality standards  
 
---

## 🧠 Technologies Used
- Python  
- MobileNetV3 / PyTorch  
- Machine Learning (Regression, Classification)  
- Computer Vision (CNNs)  
- Data Preprocessing  
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

## 👥 Group Details
| Name                | Components      	    |      Email	                |  GitHub Profile	                               | Status
| -------------       | -------------------- |  ------------------------- |  ----------------------------------------------| ----------------------
| ALOKA A.M.R.V       | IT22312662           |  IT22312662@my.sliit.lk    | [@VishvaAloka](https://github.com/VishvaAloka) |  ⭐Leader
| Jayalath M.D. T. L  | IT22328366           |  IT22328366@my.sliit.lk    | [@Thiwanka](https://github.com/Thiwanka49)     |   👨‍💻Member
| S.S Liyanage        | IT22310682           |  IT22310682@my.sliit.lk    | [@Sahan](https://github.com/Sahan003)          |   👨‍💻Member
| Mahadurage R.N      | IT22582638           |  IT22582638@my.sliit.lk    | [@Rumalya](https://github.com/RumalyaNeli)     |   👨‍💻Member


---

<p align="center">
<img src="https://github.com/Thiwanka49/Ceyloncofee-Research/blob/main/550847209_671150316017200_6014927099350871654_n.jpg" alt="Ceylon Coffee Logo" width="300"></center>
</p>

---



