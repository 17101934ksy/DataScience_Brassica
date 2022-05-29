from modular import *
from scheduler import *
from validation import *

def train(model, optimizer, train_loader, vali_loader, scheduler, device, name):
    model.to(device)

    # Loss Function
    criterion = nn.L1Loss().to(device)
    best_nmae = 0.3
    train_nmae_list = []
    vali_nmae_list = []
    vali_best_nmae_list = []

    torch_save_dict = {}

    early_stopping = EarlyStopping(patience=50, verbose=True)

    for epoch in range(1,CFG["EPOCHS"]+1):
    #for epoch in range(1,2):  
        model.train()
        train_loss = []
        train_true = []
        for img, meta, label in tqdm(iter(train_loader)):
            img, meta, label = img.float().to(device), meta.float().to(device), label.float().to(device)

            
            optimizer.zero_grad()

            # Data -> Model -> Output
            logit = model(img, meta)
            #logit = model(img)
            
            # for idx, i in enumerate(logit):
            #   if i < 0:
            #     logit[idx] = 0

            # Calc loss
            loss = criterion(logit.squeeze(1), label)
            #print(loss)

            # backpropagation
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_true.extend(label.tolist())

        if scheduler is not None:
            scheduler.step()

        train_true_tensor = torch.abs(torch.tensor(train_true))
        train_nmae = np.mean(train_loss) / torch.mean(train_true_tensor)
            
        # Evaluation Validation set
        vali_nmae = validation(model, vali_loader, criterion, device)
        
        print(f'Epoch [{epoch}] Train NMAE : [{train_nmae:.5f}] Validation MAE : [{vali_nmae:.5f}]\n')
        
        train_nmae_list.append(train_nmae)
        vali_nmae_list.append(vali_nmae)

        # Model Saved
        if best_nmae > vali_nmae:
          if train_nmae < 0.4:
            best_nmae = vali_nmae
            #torch.save(model.state_dict(), './saved/best_model.pth')
            torch.save(model.state_dict(), './best_nmae/' + name + '_best_model.pth')
            print('Model Saved.')
        vali_best_nmae_list.append(best_nmae)

        if vali_nmae < 0.2:
          key = 'train'+str(train_nmae)+'_'+'vali'+str(vali_nmae)
          torch_save_dict[key] = model.state_dict()
          print('torch_save_dict save')

        # early_stopping(vali_nmae, model)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    return  train_nmae_list, vali_nmae_list, vali_best_nmae_list, torch_save_dict
