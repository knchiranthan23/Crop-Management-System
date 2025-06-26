import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import Counter
import traceback  # Replacing cgitb for debugging
import sys

# Header information for better context in debug messages
header = ['State_Name', 'District_Name', 'Season', 'Crop']

# Question class
class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val == self.value

    def match2(self, example):
        return example.lower() in ['true', '1']

    def __repr__(self):
        return f"Is {header[self.column]} == {str(self.value)}?"

# Function to count occurrences of classes
def class_counts(data):
    counts = {}
    for row in data:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    return counts

# Leaf class represents a leaf node
class Leaf:
    def __init__(self, data):
        self.predictions = class_counts(data)

# Decision Node class represents a question node
class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

# Function to print the decision tree
def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + "--> True:")
    print_tree(node.true_branch, spacing + " ")

    print(spacing + "--> False:")
    print_tree(node.false_branch, spacing + " ")

# Function to print class probabilities
def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {lbl: f"{int(counts[lbl] / total * 100)}%" for lbl in counts.keys()}
    return probs

# Function to classify a row based on the tree
def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

# Main script execution
if __name__ == "__main__":
    try:
        # Load the decision tree model
        model_path = r'C:\Users\chira\OneDrive\Desktop\Crop Managment\crop-management-system\ML\crop_prediction\filetest2.pkl'
        dt_model_final = joblib.load(model_path)


        # Parse input arguments
        if len(sys.argv) < 4:
            raise ValueError("Usage: python script.py <state> <district> <season>")
        state = sys.argv[1]
        district = sys.argv[2]
        season = sys.argv[3]

        # Prepare testing data
        testing_data = [[state, district, season]]

        # Predict for testing data
        for row in testing_data:
            predict_dict = print_leaf(classify(row, dt_model_final)).copy()

        # Output the predictions
        for key, value in predict_dict.items():
            print(key)
            print("  ,  ")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
