# Matura Project

This project implements a Convolutional Neural Network (CNN) and Multilayer Perceptron (MLP) for handwritten digit recognition using the MNIST and EMNIST dataset. 

## Abstract

This Matura project investigates neural networks on a low-level. The goal is to program models from the ground up that function and perform equivalently to those implemented with deep learning libraries. We develop two architectures—a multilayer perceptron (MLP) and a convolutional neural network (CNN)—and apply them to the MNIST and EMNIST Balanced handwriting datasets. All components, including forward propagation, backward propagation, gradient computation, parameter updates, and convolutional operations, are theoretically defined, analysed and implemented in Python without the use of deep learning libraries. 
On MNIST, the MLP achieves an accuracy of 89.4% with SGD, and 98.3% with Adam. On EMNIST Balanced, the CNN achieves 86.8% with Adam. These results–both within 1–2% of state-of-the-art–demonstrate that fully manual implementations can reach competitive performance relative to standard baselines. The thesis offers an accessible, rigorous and concise reference for readers seeking to understand neural networks and convolutional networks at a fundamental level.

## Project Structure

ProjectFolder/

├─  main.py 

├─  Models/ (downloaded trained model files go here)

└─  EMNIST Kaggle/ (downloaded dataset files go here)

## Requirements

This project requires Python 3.9 or newer.

Required Python packages:
numpy, scipy, pandas, tensorflow, pillow (PIL), opencv-python, tkinter (included by default on Windows), matplotlib (optional for visualization).

Install everything using:

pip install numpy scipy pandas tensorflow pillow opencv-python matplotlib tkinter

## Download Required Files

This project requires external data and model files that are not included in the repository because of file size limitations.

Download the following folders and place them in the same directory as main.py.

- Models
- EMNIST Kaggle

Download Location:

- https://mega.nz/folder/QA4AXLhD#JaZJ-gY0hEqTbDR10exnAA

Ensure that the folders are unzipped after download.

## How to Run

Run the main script from the project directory with:

python main.py

## License

MIT License.

## Credits

Created by Eric Lüscher as part of my 2025 Matura project.
