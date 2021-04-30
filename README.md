# p3-ims-obd-multihead_ensemble
p3-ims-obd-multihead_ensemble created by GitHub Classroom

```


np_load_old=np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    augmix_data=np.load('aug.npy')
    
```
- train.py 에서 위에 코드 실행 



```

 ##-> train.py 에서 augmix_data를 생성 -> 인자로 전달 
def train_valid(epoch, model, train_dl, valid_dl, criterion, optimizer, logger, device, scheduler, args, augmix_data, use_augmix): ## -> augmix_data 를 인자로 받고
    best_score = 0
    now_dl = None
    for phase in ['train', 'valid']:
        run_mIoU, run_loss = [], []
        if phase == 'train':
            model.train()
            now_dl = train_dl
        else:
            model.eval()
            now_dl = valid_dl
        logger.info(f'\n{phase} on Epoch {epoch+1}')
        with torch.set_grad_enabled(phase == 'train'):
            with tqdm(now_dl, total=len(now_dl), unit='batch') as now_bar:
                for batch, sample in enumerate(now_bar):
                    now_bar.set_description(f'{phase} Epoch {epoch}')
                    optimizer.zero_grad()
                    # 3개의 이미지에 더해주자
                    images, masks = sample['image'], sample['mask']
                    #################################################### -> 이부분에 추가 
                    if phase == 'train' and use_augmix:
                        images, masks = augmix_search(augmix_data.item(), images, masks)
                    images, masks = images.to(device), masks.to(device).long()
                    ####################################################
                    preds = model(images)
                    loss = criterion(preds, masks)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    preds = torch.argmax(
                        preds.squeeze(), dim=1).detach().cpu().numpy()
                    mIoU = label_accuracy_score(
                        masks.detach().cpu().numpy(), preds, n_class=12)[2]
                    run_mIoU.append(mIoU)
                    run_loss.append(loss.item())

                    if (batch+1) % (int(len(now_dl)//10)) == 0:
                        logger.info(
                            f'{phase} Epoch {epoch+1} ==> Batch [{str(batch+1).zfill(len(str(len(now_dl))))}/{len(now_dl)}] |  Loss: {np.mean(run_loss):.5f}  |  mIoU: {np.mean(run_mIoU):.5f}')
                    now_bar.set_postfix(run_loss=np.mean(run_loss),
                                        run_mIoU=np.mean(run_mIoU))
                if phase == 'train':
                    scheduler.step(np.mean(run_loss))
                if phase == 'valid' and best_score < np.mean(run_mIoU):
                    best_score = np.mean(run_mIoU)
                    save_model(model, args.version)
                    print('best_model_saved')
```
- utils.py 에서 train 함수에서 augmix_data를 인자도 새로 추가해서 
```
 #################################################### -> 이부분에 추가 
                    if phase == 'train' and use_augmix:
                        images, masks = augmix_search(augmix_data.item(), images, masks)
                    images, masks = images.to(device), masks.to(device).long()
                    ####################################################
```
-  추가 
