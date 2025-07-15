from ct_config import debug, epochs, batch_size, number_of_ct_patients

from ct_dataset import CTDataset
from UNet_Model.segmentation_evaluator import SegmentationEvaluator
import numpy as np

class UNETEvaluator:
    def __init__(self, dataset_train, model_runner):
        self.dataset = dataset_train
        self.model_runner = model_runner

        X_train = dataset_train.X_train_masked
        Y_train = dataset_train.X_train
        X_val = dataset_train.X_val_masked
        Y_val = dataset_train.X_val
        self.X_test = dataset_train.X_test_masked
        self.Y_test = dataset_train.X_test

        print("total patients for testing:", len(dataset_train.test_idx))
        patient_ids_test = [dataset_train.patient_ids[i] for i in dataset_train.test_idx]
        print("patient_ids_test: ", patient_ids_test)

        max_patients = 30
        selected_ids = patient_ids_test[:max_patients]
        print("selected_ids: ", selected_ids)
        print("X_test.shape: ", self.X_test.shape)
        selected_X = []
        selected_X.append(self.X_test)
        self.slice_patient_ids = [
            pid for pid, vol in zip(selected_ids, selected_X) for _ in range(vol.shape[0])
        ]

        print("Len slice_patient_ids: ", len(self.slice_patient_ids))
        print("slice_patient_ids: ", self.slice_patient_ids)

    def evaluate(self):
        evaluator = SegmentationEvaluator(self.model_runner.pipeline)

        # Run full eval
        mean_dice, mean_iou, dice_patients, iou_patients = evaluator.evaluate(
            self.X_test, self.Y_test, patient_ids=self.slice_patient_ids
        )

        Y_pred = self.model_runner.pipeline.predict(self.X_test)
        Y_pred_binary = (Y_pred > 0.5).astype(np.float32)

        dice_scores = [evaluator.compute_dice(self.Y_test[i], Y_pred_binary[i])
                       for i in range(len(self.Y_test))]

        evaluator.visualize(self.X_test, self.Y_test, Y_pred_binary, dice_scores, num_examples=8)