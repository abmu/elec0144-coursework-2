# ELEC0144 - Machine Learning for Robotics - Assignment 2

This repository contains the code and resources for Assignment 2 of the ELEC0144 module (Year 2025/2026).

## Project Structure

*   `code/`: Contains all Python scripts and source code.
    *   `nn/`: Custom Neural Network library implementation (MLP, Optimisers).
    *   `q/`: Q-Learning implementation.
    *   `utils/`: Utility functions for data loading and plotting.
    *   `task-3-fruits/`: Dataset for the transfer learning task.
    *   `task-1*.py`: Regression tasks (1a, 1b, 1c, 1d).
    *   `task-2*.py`: Classification tasks (2b, 2c).
    *   `task-3.py`: Transfer Learning task.
    *   `task-4.py`: Q-Learning task.
*   `SPECIFICATION/`: Assignment guidelines and data.

## Prerequisites

Ensure you have Python 3.10 or later installed.

### Dependencies

Install the required Python packages using pip:

```bash
pip install numpy matplotlib torch torchvision
```

**Critical Note for Task 3:**
The dataset for Task 3 contains images in the **AVIF** format. To correctly process these images, you must have a modern version of the Pillow library installed (version 12.0.0 or later).

```bash
pip install --upgrade Pillow
```

## Running the Code

**Important:** It is recommended to run all scripts from the `code/` directory to ensure that relative file paths (e.g., for `task-2-iris.txt` and `task-4-grid.txt`) work correctly.

First, navigate to the code directory:

```bash
cd code
```

### Task 1: Regression (MLP)

*   **Task 1a (Main):** Trains a simple 1-3-1 MLP on polynomial data using SGD.
    ```bash
    python task-1.py
    ```
*   **Task 1b:** Experiments with different learning rates.
    ```bash
    python task-1b.py
    ```
*   **Task 1c:** Compares different optimisers (SGD, Momentum, Adam).
    ```bash
    python task-1c.py
    ```
*   **Task 1d:** Explores different network architectures and activation functions (Tanh, ReLU, Sigmoid).
    ```bash
    python task-1d.py
    ```

### Task 2: Classification

*   **Task 2 (Main):** Trains the MLP to classify Iris flowers (using `task-2-iris.txt`).
    ```bash
    python task-2.py
    ```
*   **Task 2c:** Compares SGD with Momentum vs. Adam for classification.
    ```bash
    python task2c.py
    ```

### Task 3: Transfer Learning

Fine-tunes pre-trained CNN models (AlexNet and GoogLeNet) using PyTorch on a custom fruit dataset located in `task-3-fruits/` (relative to the `code/` folder).

```bash
python task-3.py
```

### Task 4: Tabular Q-Learning

Implements and trains a Tabular Q-Learning agent to navigate a grid world environment defined in `task-4-grid.txt`.

```bash
python task-4.py
```

## Configuration & Hyperparameters

To explore different hyperparameters (as required by the assignment):
*   Open the relevant `.py` file.
*   Locate the variables at the top of the script or the `configs` lists.
*   Modify values such as `lr` (learning rate), `iterations`, `layers` (architecture), or `optimiser`.

## Outputs

*   **Console:** Displays training progress, loss values, and accuracy metrics.
*   **Plots:** 
    *   Most scripts save plots to the `out/` directory (inside `code/`) with timestamped filenames.
    *   `task2c.py` saves specific comparison plots to the `task2c_plots/` directory (inside `code/`).