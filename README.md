# Calibrated Losses for Adversatial Robust Reject Option Classification

## Description
This repository contains the implementation of algorithms to learn linear reject option classifiers using the Double Sigmoid loss and Double Ramp loss when the data is $\ell_{2}$-norm adversarially perturbed.

## Installation
1. Clone the repository: `https://github.com/Vrund0212/Calibrated-Losses-for-Adversarial-Robust-Reject-Option-Classification.git`
2. Navigate into the directory: `cd Calibrated-Losses-for-Adversarial-Robust-Reject-Option-Classification`
3. Install dependencies: `pip install -r requirements.yml`

## Explanation 
1. There are two models that can be trained, one using the Double Sigmoid loss (--dsl) and another using the Double Ramp loss (--drl).
2. $\beta$ (--beta) denotes the shift parameter and $\gamma$ (--gamma) is the radius of the perturbation ball.
3. Train $\gamma$ (--train_gamma) is the perturbation used during training phase whereas Test $\gamma$ (--test_gammas) are the perturbations at test time.
4. $d$ (--cost) is the cost of rejection, which is assumed to be constant for all examples.
5. $\mu$ (--mu) is the slope.

## Usage
```bash
# Example command to run for model trained using Double Sigmoid loss
python -Wi main.py --model dsl --beta 0.5 --train_gamma 0.01 --test_gammas 0 0.01 0.1 --cost 0.2 --mu 1.2 --tqdm
