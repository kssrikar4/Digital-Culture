# Digital Culture

A virtual fed-batch stirred tank bioreactor. This application integrates physics-based biological modelling, neural network-based control, and real-time 3D visualization to simulate fermentation processes.

## Overview

This project simulates the growth of biomass (`Escherichia coli (E. coli)`) in a controlled environment. It models complex interactions including:
- **Biological Kinetics:** Monod growth models with substrate inhibition and oxygen limitation.
- **Hydrodynamics:** Van't Riet correlations for oxygen mass transfer ($k_L a$) based on agitation power input ($P/V$) and superficial gas velocity.
- **Transport Phenomena:** Simplified 3D spatial diffusion of substrate concentration gradients.

## Features

*   **Real-time Simulation:** Adjustable simulation speed using a precise time-accumulator engine.
*   **AI Control Agent:** A PyTorch-based agent capable of optimizing process parameters (RPM, Airflow, Feed) to maximize biomass yield.
*   **Interactive Dashboard:** Full manual control overrides and real-time telemetry for Dissolved Oxygen (DO), Glucose, and Biomass concentrations.

## Project Structure

```
├── app.py                  # Application entry point and UI logic
├── requirements.txt        # Python dependencies
└── src/
    ├── simulation_core.py  # Physics engine (ODEs, Mass Transfer, Kinetics)
    ├── ai_agent.py         # PyTorch Neural Network and Heuristic Controllers
    └── visualizer.py       # Three.js/WebGL integration logic
```

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/kssrikar4/Digital-Culture.git
    cd Digital-Culture
    ```

2.  **Install dependencies:**
    ```bash
    # create a virtual environment
    python3 -m venv py
    source py/bin/activate  # On Windows, use `.\py\Scripts\activate`

    # Install dependencies
    pip install -r requirements.txt
    ```

## Usage

Run the application:

```bash
streamlit run app.py
```

### Control Panel Guide
*   **Simulation Speed:** Scales time from 0.05h/s (1x) to 1.0h/s (20x).
*   **RPM:** Controls impeller speed. Higher RPM increases Oxygen transfer ($k_L a$) but increases power consumption.
*   **Airflow:** Controls sparging rate. Essential for maintaining Dissolved Oxygen.
*   **Feed Rate:** Controls the glucose feed for fed-batch operation.

### AI Modes
*   **Maximize Biomass:** The agent aggressively optimizes parameters to produce maximum growth.
*   **Target Setpoint:** The agent modulates feeding to maintain a specific biomass concentration.

## Technical Details

### Biological Model
The simulation uses a system of coupled Ordinary Differential Equations (ODEs):
*   $\frac{dX}{dt} = (\mu - k_d)X - D X$
*   $\frac{dS}{dt} = -\frac{\mu}{Y_{xs}}X + D(S_f - S)$
*   $\frac{dDO}{dt} = k_L a(DO^* - DO) - OUR - D \cdot DO$

Where $\mu$ follows multiplicative Monod kinetics dependent on Glucose ($S$) and Oxygen ($DO$).

### Hydrodynamics
Oxygen Mass Transfer Coefficient ($k_L a$) is calculated dynamically:
$$ k_L a \propto (P/V)^{0.4} (v_s)^{0.5} $$
Where $P/V \propto N^3$ (Impeller Speed).

## License
[BSD 3-Clause License](LICENSE) - Feel free to use and modify