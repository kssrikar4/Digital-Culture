import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import multiprocessing
from src.simulation_core import SimulationCore, ReactorState

HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 32

def run_simulation_episode(seed):
    np.random.seed(seed)
    sim = SimulationCore(dt=0.1, device=torch.device('cpu'))
    
    local_memory = []
    sim.state = ReactorState()
    sim.state.biomass = np.random.uniform(0.1, 0.5)
    sim.state.glucose = np.random.uniform(10.0, 30.0)
    
    steps = 50 
    for _ in range(steps):
        current_vals = [
            sim.state.biomass, sim.state.glucose, sim.state.byproduct,
            sim.state.do, sim.state.ph, sim.state.temperature
        ]
        
        action = {
            'rpm': np.random.uniform(100, 800),
            'airflow': np.random.uniform(0.5, 3.0),
            'feed_rate': np.random.uniform(0.0, 0.2),
            'temp_setpoint': np.random.uniform(25, 37),
            'ph_setpoint': np.random.uniform(6.5, 7.5)
        }
        
        sim.state.rpm = action['rpm']
        sim.state.airflow = action['airflow']
        sim.state.feed_rate = action['feed_rate']
        sim.state.temp_setpoint = action['temp_setpoint']
        sim.state.ph_setpoint = action['ph_setpoint']
        
        sim.step()
        
        next_vals = [
            sim.state.biomass, sim.state.glucose, sim.state.byproduct,
            sim.state.do, sim.state.ph, sim.state.temperature
        ]
        
        input_vec = current_vals + list(action.values())
        local_memory.append((input_vec, next_vals))
        
    return local_memory

class BioreactorSurrogate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BioreactorSurrogate, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class AIAgent:
    def __init__(self):
        self.input_dim = 11
        self.output_dim = 6 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BioreactorSurrogate(self.input_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        self.memory = []
        self.is_trained = False
        
        self.in_scale = torch.tensor([
            10.0, 50.0, 10.0, 10.0, 14.0, 50.0,
            1000.0, 5.0, 1.0, 50.0, 14.0
        ], device=self.device)
        self.out_scale = torch.tensor([
            10.0, 50.0, 10.0, 10.0, 14.0, 50.0
        ], device=self.device)

    def collect_data(self, episodes=50):
        seeds = [np.random.randint(0, 10000) for _ in range(episodes)]
        num_workers = min(os.cpu_count() or 1, episodes)
        if num_workers > 8: num_workers = 8
        
        try:
            with multiprocessing.Pool(processes=num_workers) as pool:
                results = pool.map(run_simulation_episode, seeds)
            for res in results:
                self.memory.extend(res)
        except:
            for s in seeds:
                self.memory.extend(run_simulation_episode(s))

    def train(self, epochs=20):
        if not self.memory:
            return
            
        self.model.train()
        inputs = []
        targets = []
        
        for inp, targ in self.memory:
            inputs.append(inp)
            targets.append(targ)
            
        inputs_t = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        targets_t = torch.tensor(np.array(targets), dtype=torch.float32).to(self.device)
        
        inputs_t = inputs_t / self.in_scale
        targets_t = targets_t / self.out_scale
        
        dataset = torch.utils.data.TensorDataset(inputs_t, targets_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        for _ in range(epochs):
            for x, y in loader:
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                
        self.is_trained = True

    def suggest_optimization(self, current_state: ReactorState, mode='maximize', target_biomass=0.0):
        if mode == 'maintain':
            error = target_biomass - current_state.biomass
            rpm_cmd = 400.0
            airflow_cmd = 2.0
            feed_cmd = 0.0
            threshold = target_biomass - 0.5
            
            if current_state.biomass < threshold:
                if current_state.glucose < 3.0:
                    feed_cmd = min(0.05, (threshold - current_state.biomass) * 0.02)
            
            if current_state.biomass >= threshold:
                feed_cmd = 0.0 
            
            if current_state.glucose > 5.0:
                 feed_cmd = 0.0
            
            if current_state.do < 3.0:
                rpm_cmd = 600.0
                airflow_cmd = 3.0
            elif current_state.do > 6.0:
                rpm_cmd = 200.0
                airflow_cmd = 1.0
                
            return {
                'rpm': rpm_cmd,
                'airflow': airflow_cmd,
                'feed_rate': feed_cmd
            }

        if not self.is_trained:
             return {'rpm': 500.0, 'airflow': 2.0, 'feed_rate': 0.05}

        feed_cmd = 0.05
        if current_state.glucose > 5.0: feed_cmd = 0.0
        rpm_cmd = 500.0
        if current_state.do < 4.0: rpm_cmd = 800.0
        
        return {
            'rpm': rpm_cmd,
            'airflow': 2.0,
            'feed_rate': feed_cmd
        }