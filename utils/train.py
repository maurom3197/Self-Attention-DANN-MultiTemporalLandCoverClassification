from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
import torch
import pandas as pd
import os
import sklearn.metrics
from pathlib import Path
from utils.metrics import metrics
from utils.DANN_Transformer_model import ViTransformerExtractor, ViTransformerDANN, ViTransformer

d_model=64
n_head=2
n_layers=3
d_inner=128

def train_epoch(model, optimizer, criterion, dataloader, device, running_loss):
    model.train()
    losses = list()
    y_true_list = list()
    y_pred_list = list()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            x, y_true, _ = batch

            emb, logits, logprobabilities = model.forward(x.to(device))
            loss = criterion(logits, y_true.to(device))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            y_true_list.append(y_true)
            y_pred_list.append(logits.argmax(-1))
            iterator.set_description(f"train loss={loss:.2f}")
            losses.append(loss)
            
            
    return running_loss, torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list)

def dann_train_epoch(dann_model, optimizer, criterion, traindataloader, testdataloader, device, 
                running_loss1, running_loss2, running_loss3, alpha):
    
    dann_model.train()
    losses = list()
    y_true_list = list()
    y_pred_list = list()
    
    dataloader_iterator = iter(testdataloader)
    
    with tqdm(enumerate(traindataloader), total=len(traindataloader), leave=True) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            x, y_true, _ = batch
            x = x.to(device)
            y_true = y_true.to(device)
            
            #1 - train Gy on source domain data
            embds, logits = dann_model.forward(x)
            Gy_loss = criterion(logits, y_true)
            running_loss1 += Gy_loss.item()
            Gy_loss.backward()
            
            y_true_list.append(y_true)
            y_pred_list.append(logits.argmax(-1))
            iterator.set_description(f"train loss={Gy_loss:.2f}")
            losses.append(Gy_loss)
            
            #2 - train Gd on source domain data, the label is zero for all samples
            #Forward pass to the Gd discriminator
            Gd_outputs_source = dann_model(x, alpha)
            domain_labels0 = torch.zeros(y_true.size(0), dtype=torch.int64).to(device)
            Gd_loss_s = criterion(Gd_outputs_source, domain_labels0)
            running_loss2 += Gd_loss_s.item()
            Gd_loss_s.backward()  # backward pass: computes gradients

            #3 - train discriminator Gd on target domain data, the label is 1 for all samples
            #iterate batch from target domain dataloader
            try:
                batch = next(dataloader_iterator)
                x, y_true, _ = batch
          
            except StopIteration:
                dataloader_iterator = iter(testdataloader)
                batch = next(dataloader_iterator)
                x, y_true, _ = batch

            x = x.to(device)
            y_true = y_true.to(device)
            #Forward pass to Gd with target domain data
            Gd_outputs_target = dann_model(x, alpha)
            domain_labels1 = torch.ones(y_true.size(0), dtype=torch.int64).to(device)
            Gd_loss_t = criterion(Gd_outputs_target, domain_labels1)
            running_loss3 += Gd_loss_t.item()
            Gd_loss_t.backward()  # backward pass: computes gradients

            optimizer.step() # update weights based on accumulated gradients

    return running_loss1, running_loss2, running_loss3, torch.stack(losses), torch.cat(y_true_list), torch.cat(y_pred_list)

def train_all_models(zones, epochs, dataloaders, device, alphas, meta, save_freq):
    num_classes = meta["num_classes"]
    ndims = meta["ndims"]
    sequencelength = meta["sequencelength"]
    
    # Start training Loop
    for i in range(zones):
        traindataloader = dataloaders[i][0]
        train_zone = dataloaders[i][1]
        print('Training zone:', train_zone)

        for j in range(zones):
            if j!= i:
                # DEFINE MODE DIR AND NAME
                # if alpha scheduled save model as:
                path = 'models/vio_trasformer_dann_s'+str(i+1)+'_t'+ str(j+1)
                # if constant alpha save model as:
                #path = 'models/vio_trasformer_dann_s'+str(i+1)+'_t'+ str(j+1)+'alpha_c'
                model_dir = Path(path)

                # DEFINE MODEL
                feature_ex = ViTransformerExtractor(input_dim=ndims, n_head = n_head, n_layers = n_layers,               activation="relu",).to(device)
                dann_model = ViTransformerDANN(feature_ex, input_dim=ndims, num_classes=num_classes,
                                                n_layers = n_layers, 
                                                n_domain=2,
                                                activation="relu",).to(device)

                dann_model.modelname += f"_learning-rate={learning_rate}_weight-decay={weight_decay}"
                print(f"Initialized {dann_model.modelname}")

                # SET LOSS
                criterion = torch.nn.CrossEntropyLoss(reduction="mean")

                # SET OPTIMIZER AND LR SCHEDULER
                optimizer = Adam(dann_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                lambda2 = lambda epoch: 0.99 ** epoch
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)

                # SET TARGET DATA
                dataloader = dataloaders[j][0]
                test_zone = dataloaders[j][1]
                print('Target zone:', test_zone)

                for epoch in range(epochs):
                    alpha = alphas[epoch]

                    running_loss1 = 0.0
                    running_loss2 = 0.0
                    running_loss3 = 0.0

                    running_loss1, running_loss2, running_loss3, train_loss, y_true, y_pred = dann_train_epoch(dann_model,
                                optimizer, criterion,
                                traindataloader,testdataloader, device, 
                                running_loss1, running_loss2, running_loss3, alpha)
                    
                    loss_step = running_loss1 / len(traindataloader)
                    loss_Gd_source = running_loss2 / len(traindataloader)
                    loss_Gd_target = running_loss3 / len(testdataloader)

                    train_scores = metrics(y_true.cpu(), y_pred.cpu())

                    # Record loss and accuracy into the writer for training
                    #writer.add_scalar('Train Gy/Loss', loss_step, epoch)
                    train_accuracy = train_scores["accuracy"]
                    #writer.add_scalar('Train Gy/Accuracy', train_accuracy, epoch)
                    #writer.add_scalar('Train Gd source/Loss', loss_Gd_source, epoch)
                    #writer.add_scalar('Train Gd source/Loss', loss_Gd_target, epoch)
                    #writer.flush()

                    scheduler.step()

                    train_loss = train_loss.cpu().detach().numpy()[0]
                    # print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)
                    print(f"epoch {epoch}: trainloss {train_loss:.2f} ")

                    if epoch % save_freq == 0:
                        torch.save(dann_model.state_dict(), model_dir)
                torch.save(dann_model.state_dict(), model_dir)
            
def train_single_model(epochs, dann_model, optimizer, scheduler, criterion, traindataloader, testdataloader, device, alphas, model_f, save_freq):
    for epoch in range(epochs):
        alpha = alphas[epoch]

        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0

        running_loss1, running_loss2, running_loss3, train_loss, y_true, y_pred = dann_train_epoch(dann_model, optimizer,                             criterion,
                    traindataloader,testdataloader, device, 
                    running_loss1, running_loss2, running_loss3, alpha)
        loss_step = running_loss1 / len(traindataloader)
        loss_Gd_source = running_loss2 / len(traindataloader)
        loss_Gd_target = running_loss3 / len(testdataloader)

        train_scores = metrics(y_true.cpu(), y_pred.cpu())

        # Record loss and accuracy into the writer for training
        #writer.add_scalar('Train Gy/Loss', loss_step, epoch)
        train_accuracy = train_scores["accuracy"]
        #writer.add_scalar('Train Gy/Accuracy', train_accuracy, epoch)
        #writer.add_scalar('Train Gd source/Loss', loss_Gd_source, epoch)
        #writer.add_scalar('Train Gd source/Loss', loss_Gd_target, epoch)
        #writer.flush()

        scheduler.step()

        train_loss = train_loss.cpu().detach().numpy()[0]
        # print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)
        print(f"epoch {epoch}: trainloss {train_loss:.2f} ")

        if epoch % save_freq == 0:
            torch.save(dann_model.state_dict(), model_f)
    torch.save(dann_model.state_dict(), model_f)
    
def train_transformer(epochs, model, optimizer, scheduler, criterion, traindataloader, testdataloader, device, save_freq):
    for epoch in range(epochs):
        running_loss = 0.0
        running_loss, train_loss, y_true, y_pred  = train_epoch(model, optimizer, criterion, traindataloader, device, running_loss)
        losses[epoch] = running_loss / len(traindataloader)
        train_scores = metrics(y_true.cpu(), y_pred.cpu())
        # Record loss and accuracy into the writer for training
        #writer.add_scalar('Train/Loss', losses[epoch], epoch)
        #train_accuracy = train_scores["accuracy"]
        #writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
        #writer.flush()

    #    test_loss, y_true, y_pred, *_ = test_epoch(model, criterion, validationdataloader, device)
    #    testloss = torch.sum(test_loss)/len(validationdataloader)

        scheduler.step()

    #    scores = metrics(y_true.cpu(), y_pred.cpu())
    #    test_accuracy = scores["accuracy"]

        # Record loss and accuracy into the writer for valdation
    #    writer.add_scalar('Validation/Loss', testloss, epoch)
    #    writer.add_scalar('Validation/Accuracy', test_accuracy, epoch)
    #    writer.flush()

    #    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])
    #    test_loss = test_loss.cpu().detach().numpy()[0]
        train_loss = train_loss.cpu().detach().numpy()[0]
    #    print(f"epoch {epoch}: trainloss {train_loss:.2f}, testloss {test_loss:.2f} " + scores_msg)
        print(f"epoch {epoch}: trainloss {train_loss:.2f}")

    #    scores["epoch"] = epoch
    #    scores["trainloss"] = train_loss
    #    scores["testloss"] = test_loss
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), model_dir)
    torch.save(model.state_dict(), model_dir)

def train_transformer_all(zones, epochs, dataloaders, device, meta, save_freq):
    num_classes = meta["num_classes"]
    ndims = meta["ndims"]
    sequencelength = meta["sequencelength"]
    
    for i in range(zones):
        traindataloader = dataloaders[i][0]
        train_zone = dataloaders[i][1]
        print('Training zone:', train_zone)

        path = 'models/torch_transformer/vio_trasformer_train'+str(i+1)
        model_dir = Path(path)

        model = ViTransformer(input_dim=ndims, num_classes=num_classes, time_dim=sequencelength, 
                              n_head=n_head, 
                              n_layers = n_layers, 
                              activation="relu",).to(device)

        model.modelname += f"_learning-rate={learning_rate}_weight-decay={weight_decay}"
        print(f"Initialized {model.modelname}")

        losses = np.empty(epochs)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")

        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        lambda2 = lambda epoch: 0.99 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)

        train_transformer(epochs, model, optimizer, criterion, traindataloader, device, save_freq)