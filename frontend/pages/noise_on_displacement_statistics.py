import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from utils import simulate_with_noise_on_params
# Markdown hinzufügen
st.markdown("# Simulation with Noise on Displacement Statistics")
st.markdown("""
This page simulates a system with noise on the displacement (Y = y + gauss-noise) and displays the results.
XGBoost is used to train from randomized parameters (labels) and corresponding displacement solutions (features).
Then, tne trained XGBoost model is taken as a prediction model to be considered on the displacement added with Gaussian noise simlated before, to calculate the parameters for each sample.
This will be statistically presented, and the mean is taken to be compared to the true parameter displacement curves.
""")


def simulate_with_noise(m, c, k, f_0, w_F, t_span, z_0, z_1, t_eval, noise_level=0.1, N=10):
    # Parameter an Backend senden
    params = {
        "m": m,
        "c": c,
        "k": k,
        "f_0": f_0,
        "w_F": w_F,
        "t_span": t_span,
        "z_0": z_0,
        "z_1": z_1,
        "t_eval": t_eval
    }
    response = requests.post("http://localhost:8000/simulate", json=params)
    task_id = response.json()["task_id"]
    
    # Polling für Ergebnisse
    status = "processing"
    while status == "processing":
        result_response = requests.get(f"http://localhost:8000/result/{task_id}")
        result = result_response.json()
        status = result["status"]
        if status == "processing":
            time.sleep(1)
    
    # Ergebnisse anzeigen
    if status == "completed":
        result = result["result"]
        t = result["t"]
        z = np.array(result["z"])
        
        # Rauschen hinzufügen
        samples = []
        noise = np.random.normal(0, noise_level, size=(N, z.shape[0]))  # Rauschpegel anpassen
        for i in range(N):
            noisy_z = z + noise[i]
            samples.append(noisy_z)
        
        return t, samples
    else:
        raise Exception(result["message"])

# Input fields for parameters
m = st.number_input("Mass of the system (m)", value=1.0)
c = st.number_input("Damping parameter (c)", value=0.5)
k = st.number_input("Stiffness of the spring (k)", value=2.0)
f_0 = st.number_input("Magnitude of the forcing function (f_0)", value=1.0)
w_F = st.number_input("Frequency of the forcing function (w_F)", value=1.0)
start_time = st.number_input("Start Time", value=0.0)
end_time = st.number_input("End Time", value=10.0)
t_span = [start_time, end_time]
num_steps = st.number_input("Number of Time Steps", value=100, min_value=1)
t_eval = [start_time + i * (end_time - start_time) / (num_steps - 1) for i in range(num_steps)]
z_0 = st.number_input("Initial Position (z_0)", value=1.0)
z_1 = st.number_input("Initial Velocity (z_1)", value=1.0)
noise_level = st.number_input("Noise Level Variance", value=2.0, max_value=3.0, min_value=0.001)
N = st.number_input("Number of Samples (N)", value=10, min_value=1)
M = st.number_input("Artifical Trainset size (M)", value=100, min_value=1)

# Collect parameters
params = {
    'm': m,
    'c': c,
    'k': k,
    'f_0': f_0,
    'w_F': w_F,
    'z_0': z_0,
    'z_1': z_1,
    't_span': t_span,
    't_eval': t_eval
}

# Simulate and display results
import matplotlib.pyplot as plt

# Simulate and display results
if st.button("Run Simulation"):
    t, samples = simulate_with_noise(**params, noise_level=noise_level, N=N)
    
    # Erstelle einen DataFrame für alle Samples
    all_samples_df = pd.DataFrame(samples).T
    all_samples_df.columns = [f"Sample {i+1}" for i in range(N)]
    all_samples_df["t"] = t
    
    # Berechne den Mittelwert
    mean_sample = all_samples_df.drop(columns=["t"]).mean(axis=1)
    
    # Plotten
    fig, ax = plt.subplots()
    
    # Alle Samples in Grau plotten
    for column in all_samples_df.drop(columns=["t"]).columns:
        ax.plot(all_samples_df["t"], all_samples_df[column], color='grey', alpha=0.5)
    
    # Mittelwert in Rot plotten
    ax.plot(all_samples_df["t"], mean_sample, color='red', label='Mean')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('z')
    ax.legend()
    
    st.pyplot(fig)
    
    # Infer and display parameter reconstruction
    with st.spinner("Running Reconstruction..."):
        # Collect parameters
        t, samples, noisy_params_list = simulate_with_noise_on_params(**params, noise_level=noise_level, N=M)
        
        # Convert samples and noisy_params_list to DataFrames
        samples_df = pd.DataFrame(samples)
        params_df = pd.DataFrame(noisy_params_list)
        
        # Show statistics of the data set
        with st.expander("Show Statistics of the Simulated Train Data Set"):
            st.write("Statistics of the simulated train data set:")
            st.write(samples_df.describe())
            st.write("Statistics of the simulated train data set:")
            st.write(params_df.describe())
        
        # Train XGBoost model
        model = XGBRegressor()
        model.fit(samples_df, params_df)
        
        # Display the model's parameters
        st.write("Model trained successfully!")
        
        # Predict parameters for each sample in all_samples_df
        predicted_params = []
        for column in all_samples_df.drop(columns=["t"]).columns:
            sample = all_samples_df[column].values.reshape(1, -1)
            predicted_param = model.predict(sample)
            predicted_params.append(predicted_param[0])
        
        # Convert predicted parameters to DataFrame
        predicted_params_df = pd.DataFrame(predicted_params, columns=params_df.columns)
        
        # Show boxplots of the predicted parameters
        st.write("Boxplots of the predicted parameters:")
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Create boxplots for each parameter
        predicted_params_df.boxplot(ax=ax)
        
        # Add stars for the user-set parameters
        user_params = [params[col] for col in predicted_params_df.columns]
        for i, param in enumerate(user_params):
            ax.plot(i + 1, param, 'r*', markersize=15, label='User-set parameter' if i == 0 else "")
        
        ax.set_title("Boxplots of Predicted Parameters with User-set Parameters")
        ax.legend()
        
        st.pyplot(fig)
        
        # Plot the solution curves
        st.write("Plotting solution curves...")
        
        # Simulate with user parameters
        response = requests.post("http://localhost:8000/simulate", json=params)
        task_id = response.json()["task_id"]
        
        # Poll for results
        status = "processing"
        while status == "processing":
            result_response = requests.get(f"http://localhost:8000/result/{task_id}")
            result = result_response.json()
            status = result["status"]
            if status == "processing":
                st.write("Processing...")
                time.sleep(1)
        
        # Display results
        if status == "completed":
            result = result["result"]
            st.write("Simulation with user parameters completed!")
            user_solution = pd.DataFrame({
                "t": result["t"],
                "z": result["z"]
            })
        
        # Simulate with mean of reconstructed parameters
        mean_reconstructed_params = predicted_params_df.mean().to_dict()
        mean_reconstructed_params["t_span"] = t_span    
        mean_reconstructed_params["t_eval"] = t_eval
        
        response = requests.post("http://localhost:8000/simulate", json=mean_reconstructed_params)
        task_id = response.json()["task_id"]
        
        # Poll for results
        status = "processing"
        while status == "processing":
            result_response = requests.get(f"http://localhost:8000/result/{task_id}")
            result = result_response.json()
            status = result["status"]
            if status == "processing":
                st.write("Processing...")
                time.sleep(1)
        
        # Display results
        if status == "completed":
            result = result["result"]
            st.write("Simulation with mean reconstructed parameters completed!")
            mean_solution = pd.DataFrame({
                "t": result["t"],
                "z": result["z"]
            })
        
        # Plot all sample curves in grey
        fig, ax = plt.subplots()
        for column in all_samples_df.drop(columns=["t"]).columns:
            ax.plot(all_samples_df["t"], all_samples_df[column], color='grey', alpha=0.5)
        
        # Plot user solution in red
        ax.plot(user_solution["t"], user_solution["z"], color='red', label='User Solution')
        
        # Plot mean reconstructed solution in blue
        ax.plot(mean_solution["t"], mean_solution["z"], color='blue', label='Mean Reconstructed Solution')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('z')
        ax.legend()
        
        st.pyplot(fig)
        