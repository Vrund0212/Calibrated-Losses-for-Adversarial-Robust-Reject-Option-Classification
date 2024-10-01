# Calibrated Losses for Adversatial Robust Reject Option Classification

## Description
This repository contains the implementation of algorithms to learn linear reject option classifiers using the Double Sigmoid loss and Double Ramp loss when the data is adversarially perturbed ($\ell_{2}$).

## Installation
1. Clone the repository: `https://github.com/Vrund0212/Calibrated-Losses-for-Adversarial-Robust-Reject-Option-Classification.git`
2. Navigate into the directory: `cd Calibrated-Losses-for-Adversarial-Robust-Reject-Option-Classification`
3. Install dependencies: `pip install -r requirements.yml`

## Usage
```bash
# Example command to run the project
python -Wi main.py --model dsl --beta 0.5 --train_gamma 0.01 --test_gammas 0 0.01 0.1 --cost 0.2 --mu 1.2 --tqdm
