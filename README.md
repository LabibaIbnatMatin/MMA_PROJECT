# MMA_PROJECT
SmartSort Bin is a modular waste-sorting system using Raspberry Pi and ESP32. It automates lid control, monitors fill levels, and provides real-time feedback via OLED and audio. With UART-based communication and sensor-driven actuation, it’s designed for scalable, smart disposal.
# SmartSort Bin ♻️

SmartSort Bin is an intelligent, sensor-driven waste sorting system designed to automate bin lid control, monitor fill levels, and provide real-time feedback via audio and display. Built using Raspberry Pi and ESP32-S, it integrates servos, ultrasonic sensors, OLED display, DFPlayer Mini, and camera module for a complete IoT solution.

## 🚀 Features
- Automatic lid opening via servo motors
- Fill level detection using HC-SR04 ultrasonic sensors
- Audio feedback using DFPlayer Mini
- Visual status via OLED display
- UART-based communication between Raspberry Pi and ESP32
- Modular, scalable design for future waste classification
## 📁 Folder Overview
- `RaspberryPi/`: Python scripts for display, audio, camera, and UART
- `ESP32/`: Arduino code for servo, sensor, and command handling
- `Schematics/`: Wiring diagrams, pin maps, and Fritzing files
- `Docs/`: System overview, task breakdown, and troubleshooting notes

## 🛠️ Technologies
- Raspberry Pi 4
- ESP32-S
- DFPlayer Mini
- HC-SR04 Ultrasonic Sensors
- OLED Display (I²C)
- Servo Motors
- Fritzing, EasyEDA, Python, Arduino
