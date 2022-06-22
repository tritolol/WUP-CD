from torchmetrics import Precision, Recall
from tqdm import tqdm
from dataset_loader import WupCdLoader
import argparse
import torch
import segmentation_models_pytorch as smp


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
    parser = argparse.ArgumentParser(description='WUP_CD test script.')
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--mask_type', default="CONSTRUCTION_DEMOLITION", type=str)
    parser.add_argument('--model_name', default="DeepLabV3Plus", type=str)
    parser.add_argument('--encoder_name', default="resnet34", type=str)
    parser.add_argument('--encoder_weights', default="imagenet", type=str)
    parser.add_argument('--model_path', default="models/best.pt", type=str)


    args = parser.parse_args()

    loader = WupCdLoader()
    _, test_ds, _ = loader.get_train_test_datasets(args.dataset_path)

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

    checkpoint = torch.load(args.model_path)
    fixed_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        k = k.replace("module.", "")
        fixed_state_dict[k] = v
    model.load_state_dict(fixed_state_dict)

    with torch.no_grad():
        model.eval()
        iou, f1 = val_loop(val_gen, model, args.mask_type)
        print("IoU: %f, F1: %f" % (iou, f1))