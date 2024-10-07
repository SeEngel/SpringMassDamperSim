


import time
import numpy as np
import requests


def simulate_with_noise_on_params(m, c, k, f_0, w_F, t_span, z_0, z_1, t_eval, noise_level=0.1, N=10):
    samples = []
    noisy_params_list = []
    for _ in range(N):
        para_list = ["m", "c", "k", "f_0", "w_F", "z_0", "z_1"]   
        noisy_params = {
            "m": np.random.normal(m, noise_level * 0.1 * abs(m)),
            "c": np.random.normal(c, noise_level * 0.1 * abs(c)),
            "k": np.random.normal(k, noise_level * 0.1 * abs(k)),
            "f_0": np.random.normal(f_0, noise_level * 0.1 * abs(f_0)),
            "w_F": np.random.normal(w_F, noise_level * 0.1 * abs(w_F)),
            "t_span": t_span,
            "z_0": np.random.normal(w_F, noise_level * 0.1 * abs(z_0)),
            "z_1": np.random.normal(w_F, noise_level * 0.1 * abs(z_1)),
            "t_eval": t_eval
        }
        new_noisy_params = {k: v for k, v in noisy_params.items() if k in para_list}
        noisy_params_list.append(new_noisy_params)
        response = requests.post("http://localhost:8000/simulate", json=noisy_params)
        task_id = response.json()["task_id"]
        
        status = "processing"
        while status == "processing":
            result_response = requests.get(f"http://localhost:8000/result/{task_id}")
            result = result_response.json()
            status = result["status"]
            if status == "processing":
                time.sleep(1)
        
        if status == "completed":
            result = result["result"]
            t = result["t"]
            z = np.array(result["z"])
            samples.append(z)
        else:
            raise Exception(result["message"])
    
    return t, samples, noisy_params_list
