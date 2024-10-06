# streamlit_app.py
import os
import streamlit as st
import requests
import time
import subprocess
import pandas as pd

# Start FastAPI service if not running
def start_fastapi_service():
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            raise requests.ConnectionError
    except requests.ConnectionError:
        subprocess.Popen(["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"])

start_fastapi_service()

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Spring-Mass-Damper System Simulation", "Noise on Displacement", "Noise on parameters"])

if page == "Spring-Mass-Damper System Simulation":
    # Streamlit app layout
    st.title("Spring-Mass-Damper System Simulation")

    # Input fields for parameters
    # Input fields for parameters
    m = st.number_input("Mass of the system (m)", value=1.0)
    c = st.number_input("Damping parameter (c)", value=0.5)
    k = st.number_input("Stiffness of the spring (k)", value=2.0)
    f_0 = st.number_input("Magnitude of the forcing function (f_0)", value=1.0)
    w_F = st.number_input("Frequency of the forcing function (w_F)", value=1.0)

    # Input fields for time parameters
    start_time = st.number_input("Start Time", value=0.0)
    end_time = st.number_input("End Time", value=10.0)
    t_span = [start_time, end_time]

    # Input fields for initial conditions
    z_0 = st.number_input("Initial Position (z_0)", value=0.0)
    z_1 = st.number_input("Initial Velocity (z_1)", value=0.0)

    # Input fields for evaluation times
    num_steps = st.number_input("Number of Time Steps", value=100, min_value=1)
    # Generate t_eval based on uniform distribution using t_span
    t_eval = [start_time + i * (end_time - start_time) / (num_steps - 1) for i in range(num_steps)]


    # Convert input strings to lists


    # Button to start simulation
    if st.button("Simulate"):
        # Send parameters to backend
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
            st.write("Simulation Completed!")
            # Create a DataFrame for better plotting
            data = pd.DataFrame({
                "t": result["t"],
                "z": result["z"]
            })
            # Plot z against time
            st.line_chart(data.set_index("t")["z"])
        else:
            st.write(result["message"])

elif page == "Noise on Displacement":
    # Load the experiment page
    st.title("Noise on Displacement Page")
    path_experiments = os.path.join(os.path.dirname(__file__), "pages/noise_on_displacement.py")
    exec(open(path_experiments).read())
elif page == "Noise on parameters":
    # Load the experiment page
    st.title("Noise on Parameters Page")
    path_experiments = os.path.join(os.path.dirname(__file__), "pages/noise_on_parameter.py")
    exec(open(path_experiments).read())