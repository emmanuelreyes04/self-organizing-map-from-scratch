# Self-Organizing Map (SOM) from Scratch in Python

A **Self-Organizing Map (SOM)** neural network implemented **from scratch in Python**, without using high-level machine learning frameworks.  
This project focuses on unsupervised learning, competitive learning, and topology preservation.

## 📌 Project Overview

A Self-Organizing Map (SOM) is an unsupervised neural network that projects high-dimensional data into a lower-dimensional (usually 2D) grid while preserving the topological relationships of the data.

This implementation was developed to:
- Understand the internal mechanics of SOMs
- Experiment with different grid sizes
- Visualize how neurons self-organize during training

## 🧠 Key Concepts Implemented

- Euclidean distance for similarity measurement
- Best Matching Unit (BMU) selection
- Neighborhood function (Gaussian)
- Learning rate decay
- Neighborhood radius decay
- Competitive and cooperative learning

## ⚙️ Implementation Details

- Language: **Python**
- Libraries used:
  - `math`, `random` (core logic)
  - `matplotlib` (visualization)

> No machine learning libraries were used for the SOM model itself.

## 🧪 Experiments

The SOM was trained using a small synthetic dataset and evaluated with different grid configurations:

### 🔹 SOM 2x2
- Faster convergence
- Lower resolution mapping

### 🔹 SOM 4x4
- Better topology preservation
- Higher granularity representation

Training parameters such as learning rate, neighborhood radius, and number of epochs were adjusted for each configuration.

## 📊 Visualization

The project includes visualizations of:
- Initial random weights
- Final trained weight maps
- Color-coded representations of neuron weights

These visualizations help illustrate how the SOM organizes itself during training.

## 📁 Project Structure

.
├── som.py              # Main SOM implementation
├── README.md           # Project documentation


## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/self-organizing-map-from-scratch
2. Install dependencies:
pip install matplotlib

3. Run the script:
python SOM_Project.py


## 👤 Author

Emmanuel  
Computer Engineering student (7th semester)  
Interests: Machine Learning, Frontend Development, Intelligent Systems

