import numpy as np
from scipy.integrate import solve_ivp
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from pydantic import Field
import uuid
import asyncio

app = FastAPI()
tasks: Dict[str, Dict] = {}
COMPLETED_TASKS = 0
REQUESTED_TASKS = 0

class Parameters(BaseModel):
    m: float = Field(..., example=1.0, description="Mass of the system")
    c: float = Field(..., example=0.5, description="Damping parameter")
    k: float = Field(..., example=2.0, description="Stiffness of the spring")
    f_0: float = Field(..., example=1.0, description="Magnitude of the forcing function")
    w_F: float = Field(..., example=1.0, description="Frequency of the forcing function")
    z_0: float = Field(..., example=0.0, description="Initial position")
    z_1: float = Field(..., example=1.0, description="Initial velocity")
    t_span: List[float] = Field(..., example=[0, 10], description="Time span for the simulation")
    t_eval: List[float] = Field(..., example=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], description="Time points at which to store the computed solution")

def simulate_spring_mass_damper(params: Parameters):
    def system(t, z):
        return [z[1], (params.f_0 * np.cos(params.w_F * t) - params.c * z[1] - params.k * z[0]) / params.m]

    sol = solve_ivp(system, params.t_span, [params.z_0, params.z_1], t_eval=params.t_eval)
    return {"t": sol.t.tolist(), "z": sol.y[0].tolist(), "dzdt": sol.y[1].tolist()}

@app.post("/simulate", summary="Simulate Spring-Mass-Damper System", description="Simulates the behavior of a spring-mass-damper system given the parameters.\n\nReturns:\n- task_id: Hash value to retrieve the result")
async def simulate(params: Parameters):
    global REQUESTED_TASKS  # Add this line
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": None}
    REQUESTED_TASKS += 1
    
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, simulate_spring_mass_damper, params)
    
    async def task_done_callback(fut):
        result = await fut
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result
        
    
    asyncio.ensure_future(task_done_callback(future))
    return {"task_id": task_id}

@app.get("/result/{task_id}", summary="Get Simulation Result", description="Retrieve the result of the simulation using the task_id")
async def get_result(task_id: str):
    global COMPLETED_TASKS  # Add this line
    if task_id not in tasks:
        return {"status": "error", "message": "Task ID does not exist or has already been retrieved"}
    
    task = tasks[task_id]
    if task["status"] == "processing":
        return {"status": "processing", "message": "Still calculating solution"}
    else:
        result = task["result"]
        del tasks[task_id]
        COMPLETED_TASKS += 1
        return {"status": "completed", "result": result}

@app.get("/health", summary="Health Check", description="Check the health status of the application")
async def health_check():
    global REQUESTED_TASKS  # Add this line
    global COMPLETED_TASKS
    return {"status": "healthy", 
            "completed_tasks": COMPLETED_TASKS,
            "requested_tasks": REQUESTED_TASKS}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)