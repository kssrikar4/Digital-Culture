import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

MU_MAX_GLUCOSE = 0.5
MU_MAX_BYPRODUCT = 0.1
KS_GLUCOSE = 0.1
KS_BYPRODUCT = 0.5
KO_OXYGEN = 0.2
KI_GLUCOSE = 2.0
Y_X_S = 0.5
Y_X_P = 0.4
Y_P_S = 0.2
MAINTENANCE = 0.08

DO_SATURATION = 8.2
GRID_SIZE = 5
DIFFUSION_COEFF = 1e-4

@dataclass
class ReactorState:
    time: float = 0.0
    volume: float = 5.0
    biomass: float = 0.1
    glucose: float = 20.0
    byproduct: float = 0.0
    do: float = 8.0
    temperature: float = 37.0
    ph: float = 7.0
    
    rpm: float = 200.0
    airflow: float = 1.0
    feed_rate: float = 0.0
    feed_conc: float = 500.0
    temp_setpoint: float = 37.0
    ph_setpoint: float = 7.0

class SimulationCore:
    def __init__(self, dt=0.01, device=None):
        self.dt = dt
        self.state = ReactorState()
        
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.spatial_grid = torch.zeros((1, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE), device=self.device)
        self.spatial_grid[0, 0, 2, 2, 2] = 2.0

        self.laplacian_kernel = torch.zeros((1, 1, 3, 3, 3), device=self.device)
        self.laplacian_kernel[0, 0, 1, 1, 1] = -6.0
        self.laplacian_kernel[0, 0, 0, 1, 1] = 1.0
        self.laplacian_kernel[0, 0, 2, 1, 1] = 1.0
        self.laplacian_kernel[0, 0, 1, 0, 1] = 1.0
        self.laplacian_kernel[0, 0, 1, 2, 1] = 1.0
        self.laplacian_kernel[0, 0, 1, 1, 0] = 1.0
        self.laplacian_kernel[0, 0, 1, 1, 2] = 1.0

    def get_kla(self, rpm, airflow):
        n_rps = max(float(rpm), 10.0) / 60.0 
        p_v = 1000.0 * ((n_rps / (500/60)) ** 3.0) 
        v_s = (max(float(airflow), 0.01) / 60.0) * 0.1 
        kla_seconds = 0.026 * (p_v ** 0.4) * (v_s ** 0.5)
        return kla_seconds * 3600.0

    def _derivatives(self, t, y, controls):
        vol, X, S, P, DO, pH, T = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
        rpm, airflow, feed_rate, feed_conc, temp_sp, ph_sp = controls
        
        temp_factor = torch.exp(-0.5 * ((T - 37.0) / 10.0)**2)
        ph_factor = torch.exp(-1.0 * (pH - 7.0)**2)
        
        mu_g = MU_MAX_GLUCOSE * (S / (KS_GLUCOSE + S + 1e-6)) * (DO / (KO_OXYGEN + DO + 1e-6))
        mu_g = mu_g * temp_factor * ph_factor
        
        inhibition = KI_GLUCOSE / (KI_GLUCOSE + S + 1e-6)
        mu_p = MU_MAX_BYPRODUCT * (P / (KS_BYPRODUCT + P + 1e-6)) * (DO / (KO_OXYGEN + DO + 1e-6)) * inhibition
        mu_p = mu_p * temp_factor * ph_factor
        
        total_mu = mu_g + mu_p
        
        dilution = feed_rate / vol
        
        dVol = feed_rate
        dX = (total_mu - 0.01) * X - (dilution * X)
        dS = -(mu_g / Y_X_S) * X - (MAINTENANCE * X) + dilution * (feed_conc - S)
        
        dP_prod = Y_P_S * mu_g * X * (S / (S + 2.0 + 1e-6))
        dP_cons = (mu_p / Y_X_P) * X
        dP = dP_prod - dP_cons - (dilution * P)
        
        kla = self.get_kla(rpm, airflow)
        our = (total_mu / 1.0) * X
        dDO = kla * (DO_SATURATION - DO) - our - (dilution * DO)
        
        acid_prod = 0.05 * total_mu * X
        ph_ctrl = 0.0
        if pH < ph_sp:
            ph_ctrl = 0.1 * (ph_sp - pH)
        dpH = -acid_prod + ph_ctrl
        
        heat_gen = 0.1 * total_mu * X
        heat_loss = 0.5 * (T - 25.0)
        temp_ctrl = 1.0 * (temp_sp - T)
        dT = heat_gen - heat_loss + temp_ctrl
        
        return torch.stack([dVol, dX, dS, dP, dDO, dpH, dT])

    def step(self):
        s = self.state
        y_curr = torch.tensor([
            s.volume, s.biomass, s.glucose, s.byproduct, s.do, s.ph, s.temperature
        ], device=self.device, dtype=torch.float32)
        
        c_tensor = torch.tensor([
            s.rpm, s.airflow, s.feed_rate, s.feed_conc, s.temp_setpoint, s.ph_setpoint
        ], device=self.device, dtype=torch.float32)
        
        controls = [c_tensor[i] for i in range(6)]
        
        k1 = self._derivatives(s.time, y_curr, controls)
        k2 = self._derivatives(s.time + 0.5*self.dt, y_curr + 0.5*self.dt*k1, controls)
        k3 = self._derivatives(s.time + 0.5*self.dt, y_curr + 0.5*self.dt*k2, controls)
        k4 = self._derivatives(s.time + self.dt, y_curr + self.dt*k3, controls)
        
        y_next = y_curr + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        y_next = torch.maximum(y_next, torch.tensor(0.0, device=self.device))
        
        mixing_rate = (s.rpm / 500.0) * 5.0
        
        grid_padded = F.pad(self.spatial_grid, (1,1, 1,1, 1,1), mode='replicate')
        laplacian = F.conv3d(grid_padded, self.laplacian_kernel)
        
        change = (DIFFUSION_COEFF * laplacian) - (mixing_rate * self.spatial_grid)
        self.spatial_grid += change * self.dt
        
        if s.feed_rate > 0:
            self.spatial_grid[0, 0, 2, 2, 2] += 0.5 * s.feed_rate * self.dt

        s.time += self.dt
        vals = y_next.cpu().numpy()
        s.volume, s.biomass, s.glucose, s.byproduct, s.do, s.ph, s.temperature = vals

    def get_spatial_field(self):
        return (self.spatial_grid[0, 0] + self.state.glucose).cpu().numpy()
