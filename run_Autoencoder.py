import pickle

import hydra
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from tqdm import tqdm


from tsa import AutoEncForecast, train, evaluate
from tsa.utils import load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def run_encoder(train_iter,test_iter, model):
    attentions, latent = [], []
    model.eval()
    for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), desc="Batch 1"):
        with torch.no_grad():
            feature, y_hist, target = batch
            att, lat = model.encode(feature.to(device))
            attentions.append(att.cpu())
            latent.append(lat.cpu())
    for j, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Batch 2"):
        with torch.no_grad():
            feature, y_hist, target = batch
            att, lat = model.encode(feature.to(device))
            attentions.append(att.cpu())
            latent.append(lat.cpu())
    latent = np.array(latent).reshape(-1, 1)
    with open('Data/latent.pkl', "wb") as f:
        pickle.dump(latent, f)

def run_decoder(train_iter,test_iter, model):
    with open('Data/new_latenl.pkl', "rb") as f:
        latent = pickle.load(f)

    batch_count = 0
    outputs, targets = [], []
    model.eval()
    with torch.no_grad():
        for i, batch_train in tqdm(enumerate(train_iter), total=len(train_iter), desc="Batch 1"):
            _, y_hist, target = batch_train
            lat = latent[batch_count*64:(batch_count+1)*64]
            batch_count+=1
            out = model.decode(torch.Tensor(lat).to(device), y_hist.to(device))
            outputs.append(out.cpu())
            targets.append(target.cpu())

        for j, batch_test in tqdm(enumerate(test_iter), total=len(test_iter), desc="Batch 2"):
            _, y_hist, target = batch_test
            lat = latent[batch_count * 64:(batch_count + 1) * 64]
            batch_count += 1
            out = model.decode(torch.Tensor(lat).to(device), y_hist.to(device))
            outputs.append(out.cpu())
            targets.append(target.cpu())
    outputs = np.concatenate([arr.numpy().reshape(-1, 12) for arr in outputs], axis=0)
    targets = np.concatenate([arr.numpy().reshape(-1, 12) for arr in targets], axis=0)
    with open('Data/decoded_data.pkl', "wb") as f:
        pickle.dump(outputs, f)
    with open('Data/target.pkl', "wb") as f:
        pickle.dump(targets, f)

@hydra.main(config_path="tsa", config_name="config", version_base="1.2")
def run(cfg):
    ts = instantiate(cfg.data)
    train_iter, test_iter, nb_features = ts.get_loaders()

    model = AutoEncForecast(cfg.training, input_size=nb_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    if cfg.general.do_eval and cfg.general.get("ckpt", False):
        model, _, loss, epoch = load_checkpoint(cfg.general.ckpt, model, optimizer, device)
        evaluate(test_iter, loss, model, cfg, ts)
    elif cfg.general.do_train:
        train(train_iter, test_iter, model, criterion, optimizer, cfg, ts)
    elif cfg.general.get_latent and cfg.general.get("ckpt", True):
        model, _, loss, epoch = load_checkpoint(cfg.general.ckpt, model, optimizer, device)
        run_encoder(train_iter, test_iter, model)
    elif cfg.general.decode_latent:
        model, _, loss, epoch = load_checkpoint(cfg.general.ckpt, model, optimizer, device)
        run_decoder(train_iter, test_iter, model)
    else:
        raise ValueError("No action specified")

if __name__ == "__main__":
    run()
