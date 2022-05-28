from  moduler import *

def validation(model, vali_loader, criterion, device):
    model.eval() # Evaluation
    vali_loss = []
    vali_true = []
    with torch.no_grad():
        for img, meta, label in tqdm(iter(vali_loader)):
            img, meta, label = img.float().to(device), meta.float().to(device), label.float().to(device)

            logit = model(img, meta)
            #logit = model(img)
            
            # for idx, i in enumerate(logit):
            #   if i < 0:
            #     logit[idx] = 0

            loss = criterion(logit.squeeze(1), label)
            #print(loss)
            
            vali_loss.append(loss.item())
            vali_true.extend(label.tolist())

    vali_true_tensor = torch.abs(torch.tensor(vali_true))
    vali_nmae_loss = np.mean(vali_loss) / torch.mean(vali_true_tensor)
    return vali_nmae_loss