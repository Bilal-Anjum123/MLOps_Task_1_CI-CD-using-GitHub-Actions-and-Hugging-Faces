# File: app.py

import gradio as gr
import pickle 

# Load the saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the prediction function
def predict(median_income):
    income = [[float(median_income)]]  # Convert input to 2D array
    prediction = model.predict(income)[0]  # Get the predicted value
    return f"Predicted Median House Value: ${prediction * 100000:.2f}"

# Create a Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="California Housing Price Predictor",
    description="Enter the median income to predict the median house value."
)

if __name__ == "__main__":
    interface.launch()
