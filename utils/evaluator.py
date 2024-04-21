from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict
import torch
from collections import defaultdict

def get_loss_value(loss_type):
    if loss_type == "mse":
        return torch.nn.MSELoss(reduction="sum")
    elif loss_type == "cross_entropy":
        return torch.nn.CrossEntropyLoss(reduction="sum")
    elif loss_type == "focal":
        return FocalLoss()
    else:
        raise ValueError(f"Loss type {loss_type} not supported")


def eval_to_wandb(eval_dict: Dict[str, Dict[str, float]], is_train: bool):
    """Logs the evaluation dictionary to wandb
    :param eval_dict: dictionary of evaluation values
    :param is_train: bool, whether the evaluation is for training or testing"""
    suffix = "train" if is_train else "test"
    log_dict = {}
    for key, val in eval_dict.items():
        log_dict[f"{key}_{suffix}"] = val
    return log_dict


def eval_to_print(eval_dict: Dict[str, Dict[str, float]], is_train: bool):
    """Prints the evaluation dictionary
    :param eval_dict: dictionary of evaluation values
    :param is_train: bool, whether the evaluation is for training or testing
    :param epoch: int, current epoch"""
    print_msg = ""
    suffix = "train" if is_train else "test"
    for key, val in eval_dict.items():
        if key.startswith("Loss") or key.startswith("Accuracy/accuracy"):
            print_msg += f" | {key}_{suffix} {val:.5f}"
    return print_msg


class Evaluator(ABC):
    def __init__(self, loss_fn, **kwargs):
        super(Evaluator, self).__init__()
        self.loss_fn = loss_fn
        self.n = 0
        self._init(**kwargs)

    @abstractmethod
    def _init(self, **kwargs):
        """Initializes the evaluation dictionary""" 
        pass

    def reset(self):
        for key in self.eval_dict:
            self.eval_dict[key] = 0.0
        self.n = 0

    def evaluate(self, y_pred, y_true) -> Dict[str, Dict[str, float]]:
        self._evaluate(y_pred, y_true)
        self.n += y_pred.shape[0]

    @abstractmethod
    def _evaluate(self, y_pred, y_true):
        """Computes the loss
        :return: dictionary Dict[key: val]
        eg. {"Loss/loss": 0.0}}
        """
        raise NotImplementedError

    def get_evaluation(self, reset=True) -> Dict[str, Dict[str, float]]:
        """Flushes the evaluation dictionary and returns the average of the values
        :return: dictionary of evaluation values
        """
        if self.n == 0:
            return {}
        avg_dict = {}
        for key, value in self.eval_dict.items():
            avg_dict[key] = value / self.n
        return avg_dict


class DummyEvaluator(Evaluator):
    def _init(self, **kwargs):
        """Initializes the evaluation dictionary"""
        self.eval_dict = {}

    def _evaluate(self, y_pred, y_true) -> Dict[str, Dict[str, float]]:
        """Does nothing"""
        pass


class LossEvaluator(Evaluator):
    def _init(self, **kwargs):
        self.eval_dict = {"Loss/loss": 0.0}

    def _evaluate(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)
        self.eval_dict["Loss/loss"] += loss.item()

class LossAccuracyEvaluator(Evaluator):
    def _init(self, **kwargs):
        self.eval_dict = {"Loss/loss": 0.0,
                          "Accuracy/accuracy": 0.0}

    def _evaluate(self, y_pred, y_true):
        ''' y_pred:  (B, N, C) '''
        B, N, C = y_pred.shape
        loss = self.loss_fn(y_pred.reshape(B*N, C), y_true.reshape(B*N, C))
        self.eval_dict["Loss/loss"] += loss.item() / N
        # accuracy
        _, predicted = torch.max(y_pred.data, dim=-1)
        _, y_true_class = torch.max(y_true.data, dim=-1)
        # mean over points, but sum over batches (.evaluate takes care of mean later)
        correct = (predicted == y_true_class).to(dtype=torch.float32).mean(dim=-1).sum().item()
        self.eval_dict["Accuracy/accuracy"] += correct

class LossAccuracyByClassEvaluator(Evaluator):
    def _init(self, **kwargs):
        self.eval_dict = defaultdict(lambda: 0.0)

    def _evaluate(self, y_pred, y_true):
        ''' y_pred:  (B, N, C) '''
        B, N, C = y_pred.shape
        loss = self.loss_fn(y_pred.reshape(B*N, C), y_true.reshape(B*N, C))
        self.eval_dict["Loss/loss"] += loss.item() / N
        # accuracy
        _, predicted = torch.max(y_pred.data, dim=-1)
        _, y_true_class = torch.max(y_true.data, dim=-1)
        # mean over points, but sum over batches (.evaluate takes care of mean later)
        correct = (predicted == y_true_class).to(dtype=torch.float32).sum().item()
        self.eval_dict["Accuracy/accuracy"] += correct
        self.eval_dict["Accuracy/accuracy_total_points"] += B*N
        # accuracy by class
        for c in range(C):
            correct = ((predicted == y_true_class) & (y_true_class == c)).to(dtype=torch.float32).sum().item()
            self.eval_dict[f"Accuracy/class_{c}"] += correct
            self.eval_dict[f"Accuracy/class_{c}_total_points"] += (y_true_class == c).sum().item()
    
    def get_evaluation(self, reset=True) -> Dict[str, Dict[str, float]]:
        """Flushes the evaluation dictionary and returns the average of the values
        :return: dictionary of evaluation values
        """
        if self.n == 0:
            return {}
        avg_dict = {}
        for key, value in self.eval_dict.items():
            if key.endswith("total_points"):
                continue
            elif key.startswith("Accuracy"):
                avg_dict[key] = value / self.eval_dict[key + "_total_points"] if self.eval_dict[key + "_total_points"] > 0 else -1
            else: 
                avg_dict[key] = value / self.n
        return avg_dict
    

# to be used for MIOU and Loss Calculations
class MIOU_Evaluator(Evaluator):

    def _init(self, **kwargs):
        """Initializes the evaluation dictionary""" 
        self.eval_dict = defaultdict(lambda: 0.0)
        self.total_points = 0

    def reset(self):
        # key will be in form of class_i_intersection or class_i_union for all i \in [N] where N is number of classes
        for key in self.eval_dict:
            self.eval_dict[key] = 0.0
        self.total_points = 0

    def _evaluate(self, y_pred, y_true):
        """Computes the loss
        :return: dictionary Dict[key: val]
        eg. {"Loss/loss": 0.0}}
        """
        # breakpoint()
        B, N, C = y_pred.shape
        self.total_points += B*N
        # loss calculation, I am doing it slightly differently than before
        loss = self.loss_fn(y_pred.reshape(B*N, C), y_true.reshape(B*N,C))
        self.eval_dict["Loss/loss"] += loss.item()
        # MIOU calculation
        _, predicted = torch.max(y_pred.data, dim=-1)
        _, y_true_class = torch.max(y_true.data, dim=-1)
        self.num_classes = y_pred.shape[-1]
        for c in range(self.num_classes):
            intersection = ((predicted == y_true_class) & (y_true_class == c)).to(dtype=torch.float32).sum().item()
            union = ((predicted == c) | (y_true_class == c)).to(dtype=torch.float32).sum().item()
            self.eval_dict[f"class_{c}_intersection"] += intersection
            self.eval_dict[f"class_{c}_union"] += union

    def get_evaluation(self, reset=True) -> Dict[str, Dict[str, float]]:
        """Flushes the evaluation dictionary and returns the average of the values
        :return: dictionary of evaluation values
        """
        if self.total_points == 0:
            return {}
        avg_dict = {}
        miou = 0.0
        valid_classes = 0
        for c in range(self.num_classes):
            intersection = self.eval_dict[f"class_{c}_intersection"]
            union = self.eval_dict[f"class_{c}_union"]
            miou += intersection / union if union > 0 else 0
            valid_classes += 1 if union > 0 else 0
        avg_dict["MIOU/miou"] = miou / valid_classes
        avg_dict["Loss/loss"] = self.eval_dict["Loss/loss"] / self.total_points
        return avg_dict
    
EVALUATOR_REGISTRY = {
    "LossEvaluator": LossEvaluator,
    "LossAccuracyEvaluator": LossAccuracyEvaluator,
    "LossAccuracyByClassEvaluator": LossAccuracyByClassEvaluator,
    "MIOU_Evaluator": MIOU_Evaluator
}
    
class MultipleEvaluator:
    def __init__(self, loss_fn, evaluators):
        self.evaluators = [EVALUATOR_REGISTRY[evaluator](loss_fn) for evaluator in evaluators]
        for evaluator in self.evaluators:
            evaluator._init()
    
    def reset(self):
        for evaluator in self.evaluators:
            evaluator.reset()
    
    def get_evaluation(self, reset=True) -> Dict[str, Dict[str, float]]:
        eval_dict = {}
        for evaluator in self.evaluators:
            eval_dict.update(evaluator.get_evaluation(reset))
        return eval_dict
    
    def evaluate(self, y_pred, y_true) -> Dict[str, Dict[str, float]]:
        for evaluator in self.evaluators:
            evaluator.evaluate(y_pred, y_true)




