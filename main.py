from moduler import *
from setup import *
from loader import *
from model.cnnregressor import *
from utils.train import *
from utils.validation import *
from utils.test import *
from utils.preprocessing import *


if __name__ == '__main__':

    # Resnet 256, EfficientNet 224
    train_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.RandomHorizontalFlip(0.4),
                        transforms.RandomVerticalFlip(0.4),
                        transforms.RandomRotation(0.3),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ])

    test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        #transforms.CenterCrop(224),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ])

    # Train : Validation = 0.8 : 0.2 Split
    train_len = int(len(all_img_path)*0.80)

    train_img_ = train_img_mask_[:train_len]
    train_meta_ = meta_df_t[:train_len, -1][..., np.newaxis]
    train_label = all_label[:train_len]

    vali_img_ = train_img_mask_[train_len:]
    vali_meta_ = meta_df_t[train_len:, -1][..., np.newaxis]
    vali_label = all_label[train_len:]

    test_img_ = test_img_mask_[:]
    test_meta_ = meta_df_te[:, -1][..., np.newaxis]
        # Get Dataloader

    train_dataset = CustomDataset(train_img_, train_meta_, train_label, train_mode=True, transforms=train_transform)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    vali_dataset = CustomDataset(vali_img_, vali_meta_, vali_label, train_mode=True, transforms=test_transform)
    vali_loader = DataLoader(vali_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    cnn = CNNRegressor().to(device)

    optimizer = torch.optim.Adam(params = cnn.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.000005)

    name = 'cnn_try_CosineAnnealingLR_batch_128_early_50_resize_224_torch_2'

    train_nmae_list, vali_nmae_list, vali_best_nmae_list, torch_save_dict = train(cnn, optimizer, train_loader, vali_loader, scheduler, device, name)
    test_dataset = CustomDataset(test_img_, test_meta_, None, train_mode=False, transforms=test_transform)
    test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


    checkpoint = torch.load('./best_nmae/'+name+'_best_model.pth')
    model = cnn.to(device)
    model.load_state_dict(checkpoint)

    # Inference
    preds = predict(model, test_loader, device)