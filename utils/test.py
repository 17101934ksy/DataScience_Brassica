from  moduler import *

def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img, meta in tqdm(iter(test_loader)):
            img, meta = img.float().to(device), meta.float().to(device)

            pred_logit = model(img, meta)
            pred_logit = pred_logit.squeeze(1).detach().cpu()

            for idx, i in enumerate(pred_logit):
              if i < 0:
                pred_logit[idx] = 0

            model_pred.extend(pred_logit.tolist())
    return model_pred