import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import numpy as np
from tqdm import tqdm

import config
from data_utils.vocab import Vocab
from model.bilstm import BiLSTM
from data_utils.dataset import SentimentDataset
from metric_utils.metrics import Metrics
from metric_utils.tracker import Tracker

import os
import pickle

metrics = Metrics()

def run_epoch(net, loader, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}

    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_accuracy'.format(prefix), tracker_class(**tracker_params))
    pre_tracker = tracker.track('{}_precision'.format(prefix), tracker_class(**tracker_params))
    rec_tracker = tracker.track('{}_recall'.format(prefix), tracker_class(**tracker_params))
    f1_tracker = tracker.track('{}_F1'.format(prefix), tracker_class(**tracker_params))

    criterion = nn.CrossEntropyLoss().cuda()
    pbar = tqdm(loader, desc=f"Epoch {epoch} - {prefix}")
    for x, y in pbar:
        x = x.cuda()
        y = y.cuda()

        y_pred = net(x)

        if train:
            optimizer.zero_grad()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            loss_tracker.append(loss.item())
            pbar.set_postfix({"loss": loss_tracker.mean.value})
        else:
            loss = np.array(0)
            scores = metrics.get_scores(y_pred.argmax(dim=-1).cpu(), y.cpu())

            acc_tracker.append(scores["accuracy"])
            pre_tracker.append(scores["precision"])
            rec_tracker.append(scores["recall"])
            f1_tracker.append(scores["F1"])
            pbar.set_postfix({"accuracy": acc_tracker.mean.value, "precision": pre_tracker.mean.value, 
                                "recall": rec_tracker.mean.value, "F1": f1_tracker.mean.value})

    if not train:
        return {
            "accuracy": acc_tracker.mean.value,
            "precision": pre_tracker.mean.value,
            "recall": rec_tracker.mean.value,
            "F1": f1_tracker.mean.value
        }
    else:
        return loss_tracker.mean.value


def main():

    cudnn.benchmark = True

    vocab = Vocab([config.train_path, config.val_path, config.test_path], 
                            specials=config.specials, vectors=config.word_embedding, 
                            tokenize_level=config.tokenize_level)
                            
    metrics.vocab = vocab
    
    train_dataset = SentimentDataset(config.train_path, vocab)
    train_loader = train_dataset.get_loader()
    
    val_dataset = SentimentDataset(config.val_path, vocab)
    val_loader = val_dataset.get_loader()
    
    test_dataset = SentimentDataset(config.test_path, vocab)
    test_loader = test_dataset.get_loader()

    net = nn.DataParallel(BiLSTM(vocab, config.embedding_dim, config.hidden_size)).cuda()
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    tracker = Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    max_f1 = 0 # for saving the best model
    f1_test = 0
    for e in range(config.epochs):
        run_epoch(net, train_loader, optimizer, tracker, train=True, prefix='Training', epoch=e)
        val_returned = run_epoch(net, val_loader, optimizer, tracker, train=False, prefix='Validation', epoch=e)
        test_returned = run_epoch(net, test_loader, optimizer, tracker, train=False, prefix='Evaluation', epoch=e)

        results = {
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'eval': {
                'accuracy': val_returned["accuracy"],
                "precision": val_returned["precision"],
                "recall": val_returned["recall"],
                "f1-val": val_returned["F1"],
                "f1-test": test_returned["F1"]

            }
        }
    
        torch.save(results, os.path.join(config.model_checkpoint, "model_last.pth"))
        if val_returned["F1"] > max_f1:
            max_f1 = val_returned["F1"]
            f1_test = test_returned["F1"]
            torch.save(results, os.path.join(config.model_checkpoint, "model_best.pth"))

        print("+"*13)

    print(f"Training finished. Best F1 score: {max_f1}. F1 score on test set: {f1_test}")
    print("="*31)

if __name__ == '__main__':
    main()
