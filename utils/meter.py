import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config

args = config.get_args()


class Meter:
    def __init__(self):
        self.correct = 0
        self.total = 0
        self.losses = []
        self.precision = 0
        self.recall = 0
        self.f1 = 0

    def update(self, pt, gt, loss):
        """
        pt: n x 1, integer
        gt: n x 1, integer
        loss: float scale
        """

        self.total += len(gt)
        self.correct += (pt == gt).sum().item()
        self.losses.append(loss)

    def get(self):
        """
        return:
            averaged loss
            total correctness
            precision
            recall
            f1
            total samples
        """
        avg_loss = sum(self.losses) / len(self.losses)
        return {
            "loss": avg_loss,
            "correct": self.correct,
            "total": self.total,
        }


class EvaluationMetrics:
    """
    Evaluate based on accuracy, precision, recall and f1_score
    Evaluate based on fully connected 2
    """

    def __init__(self):
        self.l2norm = torch.tensor([])
        self.similarity = torch.tensor([])
        self.sape = torch.tensor([])
        self.fc2_similarity = torch.tensor([])
        self.model_similarity = torch.tensor([])

    def update(self, base, pt, base_model, pt_model):
        self.l2norm = torch.cat([self.l2norm, (base - pt).square().sum(dim=1)])
        self.similarity = torch.cat(
            [self.similarity, torch.nn.CosineSimilarity(dim=1)(base, pt)]
        )
        self.sape = torch.cat(
            [
                self.sape,
                (base - pt).abs().sum(dim=1)
                / (base.abs().sum(dim=1) + pt.abs().sum(dim=1)),
            ]
        )

        # # similarity between 2 models
        # local_updates = [base_model, pt_model]
        #
        # update_flats = []
        # for update in local_updates:
        #     update_f = [var.flatten() for key, var in update.items()]
        #     update_f = torch.cat(update_f)
        #     update_flats.append(update_f)
        #
        # for i in range(len(update_flats)):
        #     list1 = []
        #     for j in range(len(update_flats)):
        #         if i != j:
        #             vector_a = np.array(update_flats[i].view(-1, 1).cpu())
        #             vector_b = np.array(update_flats[j].view(-1, 1).cpu())
        #
        #             a = vector_a[vector_a < 1.0]
        #             b = vector_b[vector_a < 1.0]
        #
        #             num = np.dot(a, b)
        #             denom = np.linalg.norm(a) * np.linalg.norm(b)
        #             cos = num / denom
        #
        #             # list1.append(cos)
        #             self.model_similarity = torch.cat(
        #                 [self.model_similarity, torch.tensor(cos)]
        #             )
        #
        #

        # base_flat_weight = torch.flatten(base_model.fc2.weight)
        # pt_flat_weight = torch.flatten(pt_model.fc2.weight)

        base_flat_weight = base_model.fc2.weight
        pt_flat_weight = pt_model.fc2.weight

        # print("quick debug 1:", base.shape)
        # print("quick debug 2:", base_model.fc2.weight.shape)

        self.fc2_similarity = torch.cat(
            [
                self.fc2_similarity,
                torch.nn.CosineSimilarity(dim=1)(
                    base_flat_weight.cpu(), pt_flat_weight.cpu()
                ),
            ]
        )

    def get(self):
        return [
            self.l2norm.mean().item(),
            self.similarity.mean().item(),
            self.sape.mean().item(),
            self.fc2_similarity.mean().item(),
            self.model_similarity,
        ]
