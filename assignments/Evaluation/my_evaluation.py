# Name: Raga Lagudua Ganesan
# Email: rl1158@g.rit.edu
# I didn't use the hint file

import numpy as np
import pandas as pd
from collections import Counter

class my_evaluation:
    # Binary class or multi-class classification evaluation
    # Each data point can only belong to one class

    def __init__(self, predictions, actuals, pred_proba=None):
        # inputs:
        # predictions: list of predicted classes
        # actuals: list of ground truth
        # pred_proba: pd.DataFrame of prediction probability of belonging to each class
        self.predictions = np.array(predictions)
        self.actuals = np.array(actuals)
        self.pred_proba = pred_proba
        if type(self.pred_proba) == pd.DataFrame:
            self.classes_ = list(self.pred_proba.keys())
        else:
            self.classes_ = list(set(list(self.predictions) + list(self.actuals)))
        self.confusion_matrix = None

    def confusion(self):
        # compute confusion matrix for each class in self.classes_
        # self.confusion = {self.classes_[i]: {"TP":tp, "TN": tn, "FP": fp, "FN": fn}}
        # no return variables
        # write your own code below
        self.confusion_matrix = {}
        for label in self.classes_:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for i in range(len(self.predictions)):
                if self.predictions[i] == label and self.actuals[i] == label:
                    tp += 1
                elif self.predictions[i] != label and self.actuals[i] != label:
                    tn += 1
                elif self.predictions[i] == label and self.actuals[i] != label:
                    fp += 1
                elif self.predictions[i] != label and self.actuals[i] == label:
                    fn += 1
            self.confusion_matrix[label] = {"TP":tp, "TN":tn, "FP":fp, "FN":fn}


    def precision(self, target=None, average = "macro"):
        # compute precision
        # target: target class (str). If not None, then return precision of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average precision
        # output: prec = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()
        precision_values = []
        if target is not None:
            if target in self.confusion_matrix:
                # Calculate precision for the target class
                tp = self.confusion_matrix[target]["TP"]
                fp = self.confusion_matrix[target]["FP"]
                # Avoiding division by zero
                if tp + fp == 0:
                    prec = 0.0
                else:
                    prec = float(tp) / (tp + fp)
                return prec
            else:
                return None
        for label in self.confusion_matrix:
            tp = self.confusion_matrix[label]["TP"]
            fp = self.confusion_matrix[label]["FP"]
            # Avoiding division by zero
            if tp + fp == 0:
                prec = 0.0
            else:
                prec = float(tp) / (tp + fp)
            precision_values.append(prec)
            # Calculate average precision based on the "average" parameter
        if average == "macro":
            prec = sum(precision_values) / len(precision_values)
        elif average == "micro":
            tp_sum = sum(self.confusion_matrix[label]["TP"] for label in self.confusion_matrix)
            fp_sum = sum(self.confusion_matrix[label]["FP"] for label in self.confusion_matrix)
            # Avoiding division by zero
            if tp_sum + fp_sum == 0:
                prec = 0.0
            else:
                prec = float(tp_sum) / (tp_sum + fp_sum)
        elif average == "weighted":
            total_samples = len(self.actuals)
            class_weights = {label: float(sum(self.actuals == label)) / total_samples for label in
                             self.confusion_matrix}

            weighted_prec_sum = sum(class_weights[label] * precision_values[i] for i, label in enumerate(self.confusion_matrix))
            prec = weighted_prec_sum
        return prec

    def recall(self, target=None, average = "macro"):
        # compute recall
        # target: target class (str). If not None, then return recall of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average recall
        # output: recall = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix==None:
            self.confusion()
        recall_values = []
        # Check if we are calculating recall for a specific target class
        if target is not None:
            if target in self.confusion_matrix:
                # Calculate recall for the target class
                tp = self.confusion_matrix[target]["TP"]
                fn = self.confusion_matrix[target]["FN"]
                # Avoiding division by zero
                if tp + fn == 0:
                    rec = 0.0
                else:
                    rec = float(tp) / (tp + fn)
                return rec
            else:
                return None
        # Calculate recall for all classes and then compute the average
        for label in self.confusion_matrix:
            tp = self.confusion_matrix[label]["TP"]
            fn = self.confusion_matrix[label]["FN"]
            # Avoiding division by zero
            if tp + fn == 0:
                rec = 0.0
            else:
                rec = float(tp) / (tp + fn)

            recall_values.append(rec)
        # Calculate average recall based on the "average" parameter
        if average == "macro":
            rec = sum(recall_values) / len(recall_values)
        elif average == "micro":
            tp_sum = sum(self.confusion_matrix[label]["TP"] for label in self.confusion_matrix)
            fn_sum = sum(self.confusion_matrix[label]["FN"] for label in self.confusion_matrix)
            # Avoiding division by zero
            if tp_sum + fn_sum == 0:
                rec = 0.0
            else:
                rec = float(tp_sum) / (tp_sum + fn_sum)
        elif average == "weighted":
            total_samples = len(self.actuals)
            class_weights = {label: float(sum(self.actuals == label)) / total_samples for label in
                             self.confusion_matrix}

            weighted_rec_sum = sum(
                class_weights[label] * recall_values[i] for i, label in enumerate(self.confusion_matrix))
            rec = weighted_rec_sum

        return rec

    def f1(self, target=None, average = "macro"):
        # compute f1
        # target: target class (str). If not None, then return f1 of target class
        # average: {"macro", "micro", "weighted"}. If target==None, return average f1
        # output: f1 = float
        # note: be careful for divided by 0
        # write your own code below
        if self.confusion_matrix is None:
            self.confusion()
            # Initialize variables to store F1-score values
        f1_values = []
        # Check if we are calculating F1-score for a specific target class
        if target is not None:
            if target in self.confusion_matrix:
                # Calculate precision and recall for the target class
                tp = self.confusion_matrix[target]["TP"]
                fp = self.confusion_matrix[target]["FP"]
                fn = self.confusion_matrix[target]["FN"]
                # Avoid division by zero when calculating precision and recall
                if tp + fp == 0 or tp + fn == 0:
                    f1 = 0.0
                else:
                    precision = self.precision(target=target, average=average)
                    recall = self.recall(target = target, average = average)
                    f1 = 2 * (precision * recall) / (precision + recall)
                    return f1
            else:
                return None
        # Calculate F1-score for all classes and then compute the average
        for label in self.confusion_matrix:
            tp = self.confusion_matrix[label]["TP"]
            fp = self.confusion_matrix[label]["FP"]
            fn = self.confusion_matrix[label]["FN"]
            # Avoiding division by zero when calculating precision and recall
            if tp + fp == 0 or tp + fn == 0:
                f1 = 0.0
            else:
                precision = float(tp) / (tp + fp)
                recall = float(tp) / (tp + fn)
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
            f1_values.append(f1)
        # Calculate average F1-score based on the "average" parameter
        if average == "macro":
            f1 = sum(f1_values) / len(f1_values)
        elif average == "micro":
            tp_sum = sum(self.confusion_matrix[label]["TP"] for label in self.confusion_matrix)
            fp_sum = sum(self.confusion_matrix[label]["FP"] for label in self.confusion_matrix)
            fn_sum = sum(self.confusion_matrix[label]["FN"] for label in self.confusion_matrix)
            # Avoiding division by zero when calculating precision and recall
            if tp_sum + fp_sum == 0 or tp_sum + fn_sum == 0:
                f1 = 0.0
            else:
                precision_micro = float(tp_sum) / (tp_sum + fp_sum)
                recall_micro = float(tp_sum) / (tp_sum + fn_sum)
                # Calculate micro-average F1-score
                if precision_micro + recall_micro == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro)
        elif average == "weighted":
            total_samples = len(self.actuals)
            class_weights = {label: float(sum(self.actuals == label)) / total_samples for label in
                             self.confusion_matrix}

            weighted_f1_sum = sum(class_weights[label] * f1_values[i] for i, label in enumerate(self.confusion_matrix))
            f1 = weighted_f1_sum
        return f1


    def auc(self, target):
        # compute AUC of ROC curve for the target class
        # return auc = float
        if type(self.pred_proba) == type(None):
            return None
        if target in self.classes_:
            # write your own code below
            target_probabilities = self.pred_proba[target]
            sorted_data = np.argsort(target_probabilities)[::-1]
            auc_target = 0.0
            fpr = [0.0]
            tpr = [0.0]
            tp = 0
            fp = 0
            fn = Counter(self.actuals)[target]
            tn = len(self.actuals) - fn

            for i in sorted_data:
                if self.actuals[i] == target:
                    tp += 1
                    fn -= 1
                else:
                    fp += 1
                    tn -= 1
                tpr.append(tp / (tp+fn))
                fpr.append(fp / (fp+tn))
                auc_target += (fpr[-1] - fpr[-2]) * (tpr[-1] + tpr[-2]) / 2
            return auc_target
        else:
            raise Exception("Unknown target class.")




