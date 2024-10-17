import streamlit as st
import numpy as np
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt


st.markdown("# Simulation with Noise on Displacement")
st.markdown("""
```python
m (dzdt)^2 + c dzdt + k*z = f_0 * cos(w_F t)
dzdt(0) = z_1
z(0) = z_0

simulation = z+noise
```
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
z_0 = st.number_input("Initial Position (z_0)", value=0.0)
z_1 = st.number_input("Initial Velocity (z_1)", value=0.0)
noise_level = st.number_input("Noise Level Variance", value=10.0, min_value=0.001)
N = st.number_input("Number of Samples (N)", value=10, min_value=1)

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