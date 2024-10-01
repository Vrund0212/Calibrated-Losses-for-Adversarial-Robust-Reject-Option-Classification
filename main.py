from copy import deepcopy
from itertools import filterfalse
from time import sleep
import argparse
import json
import os
import random

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import losses
import models


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        x should be (num_samples, num_features), y should be (num_samples)
        """

        super().__init__()
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index, :], self.y[index]


def get_data(
    dataset: str,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """
    Returns ((train_x, train_y), (val_x, val_y), (test_x, test_y))
    """
    data = torch.tensor(
        pd.read_csv(f"data/{dataset}.csv", header=None, index_col=None).to_numpy(),
        dtype=torch.float,
    )
    perm = np.random.permutation(data.shape[0])
    data = data[perm]
    train_x, train_y = (
        data[: int(data.shape[0] * 0.7), :-1],
        data[: int(data.shape[0] * 0.7), -1].long(),
    )
    val_x, val_y = (
        data[int(data.shape[0] * 0.7) : int(data.shape[0] * 0.8), :-1],
        data[int(data.shape[0] * 0.7) : int(data.shape[0] * 0.8), -1].long(),
    )
    test_x, test_y = (
        data[int(data.shape[0] * 0.8) :, :-1],
        data[int(data.shape[0] * 0.8) :, -1].long(),
    )
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def gen_adv_data(
    x: torch.Tensor,
    y: torch.Tensor,
    model: models.RISAN | models.KP1,
    loss_fn: (
        losses.ShiftedDoubleRamp
        | losses.ShiftedDoubleSigmoid
        | losses.SurrogateGeneralizedCrossEntropy
    ),
    gamma: float,
    args,
):
    if gamma == 0:
        return x.clone()

    model.cpu()
    x_new = x.clone()

    for i in tqdm(
        range(x.shape[0]),
        desc="Generating adverserial data",
        leave=False,
        disable=not args.tqdm,
    ):
        curr_point = x[i].clone()
        center = x[i].clone()
        prev_point = torch.full_like(curr_point, torch.inf)

        while torch.linalg.norm(prev_point - curr_point) > 1e-3:
            prev_point = curr_point.detach().clone()
            parameter = nn.Parameter(curr_point)
            SGD = torch.optim.SGD([parameter], lr=1e-2, maximize=True)  # type: ignore
            SGD.zero_grad()
            if isinstance(model, models.RISAN):
                f, rho = model(parameter)
                loss = loss_fn(f, rho, y[i])
                loss.backward()
            else:
                out = model(parameter).view(1, -1)
                loss = loss_fn(out, y[i].view(-1))
                loss.backward()
            SGD.step()

            curr_point = parameter.clone()
            scaling_factor = gamma / max(gamma, torch.linalg.norm(curr_point - center))
            curr_point = center + scaling_factor * (curr_point - center)
        x_new[i] = curr_point.detach().clone()
    model.to(args.device)
    return x_new


def test_model(
    cost: float, model: models.RISAN | models.KP1, dataset: CustomDataset, args
) -> dict[str, torch.Tensor]:
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2048,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
    )
    preds = []
    ys = []

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            x, y = batch
            x = x.to(args.device)
            ys += y.flatten().tolist()
            preds += model.infer(x).flatten().cpu().tolist()

    pred = torch.tensor(preds)
    y = torch.tensor(ys)
    true_positive = ((pred == 1) * (y == 1)).sum()
    false_negative = ((pred == -1) * (y == 1)).sum()
    positive_rejected = ((pred == 0) * (y == 1)).sum()
    false_positive = ((pred == 1) * (y == -1)).sum()
    true_negative = ((pred == -1) * (y == -1)).sum()
    negative_rejected = ((pred == 0) * (y == -1)).sum()

    all_samples = (
        true_positive
        + false_negative
        + positive_rejected
        + false_positive
        + true_negative
        + negative_rejected
    )
    non_rejected_samples = (
        true_negative + true_positive + false_positive + false_negative
    )

    l0d1 = (
        (false_negative + false_positive)
        + cost * (positive_rejected + negative_rejected)
    ) / all_samples
    rejection_rate = (positive_rejected + negative_rejected) / all_samples
    if non_rejected_samples != 0:
        accuracy = (true_positive + true_negative) / non_rejected_samples
    else:
        accuracy = torch.ones(1)

    return {
        "tp": true_positive,
        "fn": false_negative,
        "pr": positive_rejected,
        "fp": false_positive,
        "tn": true_negative,
        "nr": negative_rejected,
        "rej_rate": rejection_rate,
        "acc": accuracy,
        "l0d1": l0d1,
    }


def train_model(
    train_dataset: CustomDataset,
    val_dataset: CustomDataset,
    loss_fn: (
        losses.ShiftedDoubleRamp
        | losses.ShiftedDoubleSigmoid
        | losses.SurrogateGeneralizedCrossEntropy
    ),
    args,
):
    if args.model == "cwr":
        model = models.KP1().to(args.device)
    else:
        model = models.RISAN().to(args.device)
    with torch.no_grad():
        model.eval()
        model(train_dataset[0][0].to(args.device).view(1, -1))
        model.train()
    model_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, fused=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    last_improvement = 0
    best_model = deepcopy(model)
    best_loss = float("inf")

    with torch.no_grad():
        if not isinstance(loss_fn, losses.SurrogateGeneralizedCrossEntropy) and False:
            w = model.f.weight.detach()
            b = model.f.bias.detach()
            wb = torch.concatenate([w.flatten(), b], dim=0)
            norm = torch.linalg.norm(wb)
            model.f.weight = nn.Parameter(model.f.weight / norm)
            model.f.bias = nn.Parameter(model.f.bias / norm)

    for epoch_idx in tqdm(
        range(args.num_epochs), desc="Epochs", leave=False, disable=not args.tqdm
    ):
        model.train()
        for batch in tqdm(
            train_dataloader,
            desc="Training network",
            leave=False,
            disable=not args.tqdm,
        ):
            x, y = batch
            x = x.to(args.device)
            y = y.to(args.device).flatten()

            if isinstance(model, models.RISAN):
                f, rho = model(x)
                f = f.flatten()
                rho = rho.repeat(f.shape[0])
                loss = loss_fn(f, rho, y)
            else:
                out = model(x)
                loss = loss_fn(out, y)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

            with torch.no_grad():
                if not isinstance(loss_fn, losses.SurrogateGeneralizedCrossEntropy) and False:
                    w = model.f.weight.detach()
                    b = model.f.bias.detach()
                    wb = torch.concatenate([w.flatten(), b], dim=0)
                    norm = torch.linalg.norm(wb)
                    model.f.weight = nn.Parameter(model.f.weight / norm)
                    model.f.bias = nn.Parameter(model.f.bias / norm)

        model.eval()
        with torch.no_grad():
            val_loss = torch.zeros(1, device=args.device, requires_grad=False)
            num_batches = 0
            for batch in tqdm(
                val_dataloader,
                desc="Validating network",
                leave=False,
                disable=not args.tqdm,
            ):
                num_batches += 1
                x, y = batch
                x = x.to(args.device)
                y = y.to(args.device).flatten()

                if isinstance(model, models.RISAN):
                    f, rho = model(x)
                    f = f.flatten()
                    rho = rho.repeat(f.shape[0])
                    loss = loss_fn(f, rho, y)
                else:
                    out = model(x)
                    loss = loss_fn(out, y)
                val_loss += loss.detach().clone()
            val_loss /= num_batches
            if val_loss.item() < 0.99 * best_loss:
                best_loss = val_loss.item()
                best_model = deepcopy(model)
                last_improvement = 0
            else:
                last_improvement += 1
                if last_improvement == 10:
                    break
    return best_model


def run(run_num: int, args) -> tuple[
    dict[str, torch.Tensor],
    dict[float, dict[str, torch.Tensor]],
    dict[float, dict[str, torch.Tensor]],
]:
    """
    t[0] -> Clean model results
    t[1] -> Non robust results
    t[2] -> robust results
    """
    ...
    ((train_x, train_y), (val_x, val_y), (test_x, test_y)) = get_data(args.dataset)
    if args.model == "dsl":
        loss_fn = losses.ShiftedDoubleSigmoid(args.cost, args.mu, 0)
        shifted_loss_fn = losses.ShiftedDoubleSigmoid(args.cost, args.mu, args.beta)
    elif args.model == "drl":
        loss_fn = losses.ShiftedDoubleRamp(args.cost, args.mu, 0)
        shifted_loss_fn = losses.ShiftedDoubleRamp(args.cost, args.mu, args.beta)
    else:
        loss_fn = losses.SurrogateGeneralizedCrossEntropy(args.cost, args.mu)
        shifted_loss_fn = None

    train_dataset = CustomDataset(train_x, train_y)
    val_dataset = CustomDataset(val_x, val_y)
    model = train_model(train_dataset, val_dataset, loss_fn, args)
    test_dataset = CustomDataset(test_x, test_y)
    clean_results = test_model(args.cost, model, test_dataset, args)

    adv_train_x = gen_adv_data(train_x, train_y, model, loss_fn, args.train_gamma, args)
    adv_val_x = gen_adv_data(val_x, val_y, model, loss_fn, args.train_gamma, args)
    adv_train_dataset = CustomDataset(adv_train_x, train_y)
    adv_val_dataset = CustomDataset(adv_val_x, val_y)
    base_model = train_model(adv_train_dataset, adv_val_dataset, loss_fn, args)
    if shifted_loss_fn is not None:
        robust_model = train_model(
            adv_train_dataset, adv_val_dataset, shifted_loss_fn, args
        )
        robust_model.beta = args.beta
    else:
        robust_model = None

    non_robust_results = {}
    robust_results = {}
    for gamma in args.test_gammas:
        adv_test_x = gen_adv_data(test_x, test_y, model, loss_fn, gamma, args)
        test_dataset = CustomDataset(adv_test_x, test_y)
        non_robust_results[gamma] = test_model(
            args.cost, base_model, test_dataset, args
        )
        if robust_model is not None:
            robust_results[gamma] = test_model(
                args.cost, robust_model, test_dataset, args
            )

    return clean_results, non_robust_results, robust_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["synthetic", "bc", "iris", "park"],
        required=True,
    )
    parser.add_argument(
        "--model", type=str, choices=["dsl", "drl", "cwr"], required=True
    )
    parser.add_argument("--cost", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--train_gamma", type=float, required=True)
    parser.add_argument("--test_gammas", type=float, nargs="*", required=True)
    parser.add_argument("--tqdm", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    args.num_epochs = 100000
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("results", exist_ok=True)
    if args.model == "cwr":
        args.output_file_clean = f"results/{args.dataset}_{args.model}_{args.cost:f}"
        args.output_file_non_robust = (
            f"results/{args.dataset}_{args.model}_{args.cost:f}_{args.train_gamma:f}"
        )
    else:
        args.output_file_clean = (
            f"results/{args.dataset}_{args.model}_{args.cost:f}_{args.mu:f}"
        )
        args.output_file_non_robust = f"results/{args.dataset}_{args.model}_{args.cost:f}_{args.mu:f}_{args.train_gamma:f}"
        args.output_file_robust = f"results/{args.dataset}_{args.model}_{args.cost:f}_{args.mu:f}_{args.train_gamma:f}_{args.beta:}"
    print(args)

    results: list[
        tuple[
            dict[str, torch.Tensor],
            dict[float, dict[str, torch.Tensor]],
            dict[float, dict[str, torch.Tensor]],
        ]
    ] = []
    for i in tqdm(range(5), desc="Runs", leave=False, disable=not args.tqdm):
        results.append(run(i, args))
    clean_results = {}
    non_robust_results = {}
    robust_results = {}
    for metric in results[0][0].keys():
        temp_tensor = torch.tensor([_[0][metric] for _ in results], dtype=torch.float)
        clean_results[metric] = [
            torch.mean(temp_tensor).item(),
            torch.std(temp_tensor).item(),
        ]

    for gamma in results[0][1].keys():
        non_robust_results[gamma] = {}
        for metric in results[0][1][gamma].keys():
            temp_tensor = torch.tensor(
                [_[1][gamma][metric] for _ in results], dtype=torch.float
            )
            non_robust_results[gamma][metric] = [
                torch.mean(temp_tensor).item(),
                torch.std(temp_tensor).item(),
            ]

    if args.model != "cwr":
        for gamma in results[0][2].keys():
            robust_results[gamma] = {}
            for metric in results[0][2][gamma].keys():
                temp_tensor = torch.tensor(
                    [_[2][gamma][metric] for _ in results], dtype=torch.float
                )
                robust_results[gamma][metric] = [
                    torch.mean(temp_tensor).item(),
                    torch.std(temp_tensor).item(),
                ]

    with open(f"{args.output_file_clean}_clean.json", "w") as f:
        json.dump(clean_results, f, indent=4)
    with open(f"{args.output_file_non_robust}_non_robust.json", "w") as f:
        json.dump(non_robust_results, f, indent=4)
    if args.model != "cwr":
        with open(f"{args.output_file_robust}_robust.json", "w") as f:
            json.dump(robust_results, f, indent=4)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)
    main()
