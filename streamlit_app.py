import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load models
def load_models():
    with open('best_logistic_regression_model.pkl', 'rb') as f:
        lr_model = pickle.load(f)
    with open('best_decision_tree_model.pkl', 'rb') as f:
        dt_model = pickle.load(f)
    with open('best_random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('best_neural_network_model.pkl', 'rb') as f:
        mlp_model = pickle.load(f)
    return lr_model, dt_model, rf_model, mlp_model

# Scale input data
def scale_input_data(input_data, scaler):
    return scaler.transform([input_data])

# Model evaluation function
def evaluate_input(input_data, model_choice):
    if model_choice == 'Logistic Regression':
        model = loaded_lr
    elif model_choice == 'Decision Tree':
        model = loaded_dt
    elif model_choice == 'Random Forest':
        model = loaded_rf
    elif model_choice == 'Neural Network':
        model = loaded_mlp

    prediction = model.predict(input_data)
    return prediction

# Define main function
def main():
    # Title and description
    st.title("Diabetes Prediction System")
    st.write("Enter your health indicators and select a model to predict the likelihood of diabetes.")

    # Load the models
    lr_model, dt_model, rf_model, mlp_model = load_models()

    # Load a sample dataset to get scaler (assume similar feature distribution)
    file_path = 'diabetes_binary_health_indicators_BRFSS2015.csv'
    data = pd.read_csv(file_path)
    feature_columns = data.drop('Diabetes_binary', axis=1).columns

    # Select the most relevant 15 features (example selection, adjust as needed)
    selected_features = [
        'HighBP', 'HighChol', 'BMI', 'Smoker', 'PhysActivity', 
        'Fruits', 'Veggies', 'HvyAlcoholConsump', 'GenHlth', 
        'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education'
    ]

    # Prepare the scaler
    scaler = StandardScaler().fit(data[selected_features])

    # User input section
    st.sidebar.header("User Input Parameters")
    input_data = []
    for feature in selected_features:
        value = st.sidebar.slider(f"{feature}", min_value=0.0, max_value=100.0, step=0.1)
        input_data.append(value)

    # Convert input data to numpy array and scale
    input_data = np.array(input_data).reshape(1, -1)
    scaled_input_data = scale_input_data(input_data, scaler)

    # Model selection
    st.sidebar.header("Choose Model")
    model_choice = st.sidebar.selectbox(
        'Model',
        ('Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network')
    )

    # Model accuracies (example values, replace with actual model evaluation)
    model_accuracies = {
        "Logistic Regression": 0.85,
        "Decision Tree": 0.80,
        "Random Forest": 0.90,
        "Neural Network": 0.88
    }
    st.sidebar.text(f"Accuracy of {model_choice}: {model_accuracies[model_choice]*100:.2f}%")

    # Predict button
    if st.sidebar.button('Predict'):
        # Get the prediction result
        prediction = evaluate_input(scaled_input_data, model_choice)

        # Display the result
        if prediction == 1:
            st.write("The model predicts that you have a higher likelihood of getting diabetes.")
        else:
            st.write("The model predicts that you have a lower likelihood of getting diabetes.")

# Run the main function
if __name__ == "__main__":
    main()
