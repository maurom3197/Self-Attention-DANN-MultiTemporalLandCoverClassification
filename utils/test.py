from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import pandas as pd
import os
import sklearn.metrics
from utils.metrics import metrics

def test_epoch(model, criterion, dataloader, device):
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        field_ids_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true, field_id = batch
                embds, logits = model.forward(x.to(device))
                loss = criterion(logits, y_true.to(device))
                iterator.set_description(f"test loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(logits.argmax(-1))
                y_score_list.append(logits.exp())
                field_ids_list.append(field_id)
        return torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list), torch.cat(y_score_list), torch.cat(field_ids_list)
    
    
def test_on_all_zones(zones, model, criterion, dataloader, device):
    for i in range(zones):
        traindataloader = dataloaders[i][0]
        train_zone = dataloaders[i][1]
        print('Source zone:', train_zone)

        for j in range(zones):
            if j!= i:
                # DEFINE MODE DIR AND NAME
                # if alpha scheduled save model as:
                path = 'models/vio_trasformer_dann_s'+str(i+1)+'_t'+ str(j+1)+'_maxalpha02_gamma'+str(gamma)
                # if constant alpha save model as:
                #path = 'models/vio_trasformer_dann_s'+str(i+1)+'_t'+ str(j+1)+'_alpha'+str(alpha_c)
                model_dir = Path(path)
                dann_model.load_state_dict(torch.load(model_dir))

                # SET TARGET DATA
                testdataloader = dataloaders[j][0]
                test_zone = dataloaders[j][1]
                print('Target zone:', test_zone)

                test_loss, y_true, y_pred, *_ = test_epoch(dann_model, criterion, testdataloader, device)

                scores = metrics(y_true.cpu(), y_pred.cpu())
                scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
                test_loss = test_loss.cpu().detach().numpy()[0]
                print(scores)