from dataset_loader import WupCdLoader
import torch
import segmentation_models_pytorch as smp
from focal_loss import FocalLoss
from torchmetrics import Precision, Recall
from tqdm import tqdm
import argparse


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_loop(dataloader, model, loss_func, optimizer, mask_name):
    size = len(dataloader)
    print("Training on %d batches..." % size)

    pbar = tqdm(dataloader)
    for data in pbar:
        tdom = data['/nDSM/B'] - data['/nDSM/A']

        src_pred = model(tdom.cuda())
        loss = loss_func(src_pred, data['/MASKS/' + mask_name].float().cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("loss: %f" % loss)

def val_loop(dataloader, model, mask_name):
    size = len(dataloader)

    print("Validation on %d batches..." % size)

    Prec = Precision(1, multiclass=False).cuda()
    Reca = Recall(1, multiclass=False).cuda()

    for data in tqdm(dataloader):
        tdom = data['/nDSM/B'] - data['/nDSM/A']
        pred = model(tdom.cuda())
        pred_thresh = (pred > 0).int().cuda()
        target = data['/MASKS/' + mask_name].cuda()

        Prec.update(pred_thresh, target)
        Reca.update(pred_thresh, target)

    p = Prec.compute().item()
    r = Reca.compute().item()

    iou = p*r/(p+r-p*r)
    f1 = (2*p*r)/(p+r)

    return iou, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WUP_CD train script.')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--validation_interval_epochs', default=5, type=int)
    parser.add_argument('--model_save_interval_epochs', default=5, type=int)
    parser.add_argument('--save_best_model', action='store_true')
    parser.add_argument('--best_model_path', default="models/best.pt", type=str)
    parser.add_argument('--latest_model_path', default="models/latest.pt", type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--mask_type', default="CONSTRUCTION_DEMOLITION", type=str)
    parser.add_argument('--model_name', default="DeepLabV3Plus", type=str)
    parser.add_argument('--encoder_name', default="resnet34", type=str)
    parser.add_argument('--encoder_weights', default="imagenet", type=str)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--multi_gpu', action="store_true")


    args = parser.parse_args()

    loader = WupCdLoader()
    train_ds, test_ds, sampler = loader.get_train_test_datasets(args.dataset_path)

    train_gen = torch.utils.data.DataLoader(train_ds, num_workers=args.num_workers, batch_size=args.batch_size, sampler=sampler)
    val_gen = torch.utils.data.DataLoader(test_ds, num_workers=args.num_workers, batch_size=args.batch_size)

    ew = args.encoder_weights
    if ew == "none":
        ew = None

    if args.model_name == "DeepLabV3Plus":
        model = smp.DeepLabV3Plus(encoder_name=args.encoder_name, encoder_weights=ew, in_channels=1).cuda()
    elif args.model_name == "UNet":
        model = smp.UNet(encoder_name=args.encoder_name, encoder_weights=ew, in_channels=1).cuda()
    else:
        raise ValueError("Unsupported Model!")

    if args.multi_gpu:
        model = torch.nn.DataParallel(model)

    print("%d Model Parameters" % count_parameters(model))

    loss = FocalLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

    old_best_f1 = 0
    for epo in range(args.train_epochs):
        print("Epoch ", epo + 1)
        model.train()
        train_loop(train_gen, model, loss, optimizer, args.mask_type)
        if (epo + 1) % args.validation_interval_epochs == 0:
            with torch.no_grad():
                model.eval()
                iou, f1 = val_loop(val_gen, model, args.mask_type)
                print("IoU: %f, F1: %f" % (iou, f1))
            if args.save_best_model and f1 > old_best_f1:
                print("New best F1, saving model...")
                old_best_f1 = f1
                torch.save({
                        'epoch': epo,
                        'f1': f1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, args.best_model_path)
        if (epo + 1) % args.model_save_interval_epochs == 0:
            torch.save({
                        'epoch': epo,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, args.latest_model_path)