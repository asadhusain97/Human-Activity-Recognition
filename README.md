# Human Activity Recognition

This repository contains the codes for a real world project I did with Purdue Datamine team from Jan to May'2022. Learn more about the project [here](https://asadhusain97.github.io/projects/humanactivity.html)

## Problem
Our client had a worker safety management system, with an arm worn device with various sensors for tracking his/her activity.
Our task in this project was to classify an event in the following four categories using 15-sec (25Hz) accelerometer data. 

- slip
- trip
- fall
- other

## What did we do?

**Step 1 - Understanding data**

We had 375 acceleration values in x,y, and z directions for every incident. 

**Step 2 - Brainstorming features**

From those 3 columns we created upwards of 20 new features which included simple statistical manipulations of the acceleration values like mean, max, variance, range, and standard deviation to more complex concepts like fourrier series and signal processing.
 
 **Step 3 - Modelling**
 
We tried multiple machine learning models and got the best results with Neural Network (MLP) and a convoluted neural network. 

**Results**

Overall we improved the model precision to 74%. 


## About the Data

The dataset is available as a compressed `.gz` file and consists of 9 variables (i.e., columns) with 986,250 rows.

**1. What is the sampling frequency?**

The sampling frequency of the wearable worn was 25 Hz - meaning that 25 samples (or readings) were collected per second from the accelerometer

**2. Where does the 15 seconds come from?**

While the wearable's accelerometer is continuously 'listening' to the motion of the wearer, it was designed to 'start' recording once 2Gs of acceleration was exceeded in either the X, Y, or Z axes. The moment of time that when this threshold is exceeded is at the mid-point. And the wearable 'stores' the previous 7.5 seconds of information and the following 7.5 seconds. Hence, the 15 seconds.

**3. What is an incident?**

An incident is when the wearable's accelerometer exceeds 2Gs of acceleration in either the X, Y, or Z axes. It is at this moment in time when the wearable will record the previous 7.5 seconds and the following 7.5 seconds. This 15 second data-stream is referred to as an incident and classified as either a slip, trip, fall, or other.


|    Variable   | Data Type |                                                                                        Description                                                                                        |
|:-------------:|:---------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    hash_id    |   string  | anonymized id that could indicate the wearer of the device                                                                                                                                |
|  incident_id  |   double  | ID number of an incident. Every time the wearable sends data about a potential motion, this instance is known as an incident.                                                             |
|     motion    |   string  | type of motion that wearer experienced (verified by the wearer)                                                                                                                           |
| sample_number |   double  | sample number of the recorded data from accelerometer. Each incident consists of 375 samples                                                                                              |
|  milliseconds |   double  | derived column that describes the 'time' elapsed since the last data point. Each sample is recorded at 40 ms. The sampling frequency is 25 Hz (i.e., 25 samples are collected per second) |
|    seconds    |   double  | derived column that describes which 'second' this data belongs to. Each incident consists of 15 seconds of data.                                                                          |
|       x       |   double  | the acceleration (in G's) of the wearable in the X-axis                                                                                                                                   |
|       y       |   double  | the acceleration (in G's) of the wearable in the Y-axis                                                                                                                                   |
|       z       |   double  | the acceleration (in G's) of the wearable in the Z-axis                                                                                                                                   |
