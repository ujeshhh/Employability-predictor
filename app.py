import gradio as gr
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load or train models
model_files = ["scaler.pkl", "logistic_regression.pkl", "perceptron.pkl"]

# Check if all model files exist
if all(os.path.exists(file) for file in model_files):
    try:
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        
        with open("logistic_regression.pkl", "rb") as model_file:
            logistic_regression = pickle.load(model_file)
        
        with open("perceptron.pkl", "rb") as model_file:
            perceptron = pickle.load(model_file)

    except Exception as e:
        print(f"Error loading models: {e}")
        exit()

else:
    print("Training models from scratch...")

    # Load dataset
    df = pd.read_excel("Student-Employability-Datasets.xlsx", sheet_name="Data")

    # Prepare dataset
    X = df.iloc[:, 1:-1].values  # Select all feature columns
    y = (df["CLASS"] == "Employable").astype(int)  # Convert to binary labels

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    logistic_regression = LogisticRegression(random_state=42)
    logistic_regression.fit(X_train_scaled, y_train)

    perceptron = Perceptron(random_state=42)
    perceptron.fit(X_train_scaled, y_train)

    # Save trained models
    with open("scaler.pkl", "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with open("logistic_regression.pkl", "wb") as model_file:
        pickle.dump(logistic_regression, model_file)
    with open("perceptron.pkl", "wb") as model_file:
        pickle.dump(perceptron, model_file)

    print("Training complete. Models saved.")

# Employability prediction function
def predict_employability(name, ga, mos, pc, ma, sc, api, cs, model_choice):
    try:
        if not name:
            name = "The candidate"

        input_data = np.array([[ga, mos, pc, ma, sc, api, cs]])

        # Ensure scaler is correctly loaded
        if scaler is None:
            return "Error: Scaler not loaded properly."

        # Transform input
        input_scaled = scaler.transform(input_data)

        # Select model and make prediction
        if model_choice == "Logistic Regression":
            prediction = logistic_regression.predict(input_scaled)
        elif model_choice == "Perceptron":
            prediction = perceptron.predict(input_scaled)
        else:
            return "Error: Invalid model selection."

        # Return result
        if prediction[0] == 1:
            return f"{name} is Employable 😊"
        else:
            return f"{name} is Less Employable - Work Hard! 💪"

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# 🚀 Employability Evaluation System")
    
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Name")
            ga = gr.Slider(1, 5, step=1, label="General Appearance")
            mos = gr.Slider(1, 5, step=1, label="Manner of Speaking")
            pc = gr.Slider(1, 5, step=1, label="Physical Condition")
            ma = gr.Slider(1, 5, step=1, label="Mental Alertness")
            sc = gr.Slider(1, 5, step=1, label="Self Confidence")
            api = gr.Slider(1, 5, step=1, label="Ability to Present Ideas")
            cs = gr.Slider(1, 5, step=1, label="Communication Skills")
            model_choice = gr.Radio(["Logistic Regression", "Perceptron"], label="Select Model")

            predict_btn = gr.Button("Get Evaluation")

        with gr.Column():
            result_output = gr.Textbox(label="Employability Prediction")

    # Button Click Event
    predict_btn.click(
        fn=predict_employability,
        inputs=[name, ga, mos, pc, ma, sc, api, cs, model_choice],
        outputs=[result_output]
    )

# Launch the app
app.launch(server_port=7860, debug=True, share=True)