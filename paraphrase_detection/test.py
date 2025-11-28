import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score


class test():
    def __init__(
            self,
            model,
            test_loader,
            criterion,
            device,
    ):
        self.model=model
        self.criterion=criterion
        self.device=device
        self.test_loader=test_loader
        self.model.to(self.device)
        self.fixed=model.fixed

    def run_testing_loop(self):
        self.model.eval()
        with torch.no_grad():
            test_batch_losses=[]
            all_test_preds=[]
            all_test_labels=[]
            for test_X_batch, test_y_batch in self.test_loader:
                if self.fixed:
                    test_X_batch = np.array(test_X_batch).T
                    test_x0 = test_X_batch[:,0].tolist()
                    test_x1 = test_X_batch[:,1].tolist()
                else:
                    test_x0=test_X_batch[0]
                    test_x0={k:v.to(self.device) for k,v in test_x0.items()}
                    test_x1=test_X_batch[1]
                    test_x1={k:v.to(self.device) for k,v in test_x1.items()}
                test_y_batch.to(self.device)
                
                logits=self.model(test_x0,test_x1)
                test_y_batch=test_y_batch.view(-1)
                
                loss=self.criterion(logits, test_y_batch)
                test_batch_losses.append(loss.item())
                preds=torch.argmax(logits,dim=1)
                all_test_preds.append(preds)
                all_test_labels.append(test_y_batch)

            all_test_preds=torch.cat(all_test_preds)
            all_test_labels=torch.cat(all_test_labels)

            test_accuracy=accuracy_score(all_test_labels,all_test_preds)
            test_f1=f1_score(all_test_labels,all_test_preds)
            avg_test_loss=np.mean(test_batch_losses)

        return test_accuracy,test_f1,avg_test_loss


