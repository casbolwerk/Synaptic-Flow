import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

from Utils.metrics import compare_model_dicts

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def eval_preds(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    preds = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            _, top_pred = output.topk(1, dim=1)
            preds += top_pred
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return accuracy1, preds

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)

def pretrain_analysis_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    print('\r[pretrain loop] Now starting pretraining...')
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        print(f'\r[pretrain loop] Epoch {epoch} of pretraining')
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)

        # instead of evaluating, implement an instability analysis and run it
        print(f'\r[pretrain loop] Epoch {epoch} instability analysis')
        instability_analysis(model, loss, optimizer, train_loader, test_loader, device, epoch, verbose)
        # row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        # rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)

def instability_analysis(model, loss, optimizer, train_loader, test_loader, device, epoch, verbose):
    print('\r[instability analysis] Now starting analysis...')
    # launch two training sessions (ensure that they have different optimizers)
    optimizer_1 = optimizer
    optimizer_2 = optimizer

    # get model weights dictionary
    original_dict = model.state_dict()

    insta_model_1 = copy.deepcopy(model)
    insta_model_2 = copy.deepcopy(model)

    #maybe run training sessions for more than one epoch
    # for epoch in tqdm(range(epochs/10)):
    print('\r[instability analysis] Training temp model 1')
    train_loss_model_1 = train(insta_model_1, loss, optimizer_1, train_loader, device, epoch, verbose)
    print('\r[instability analysis] Training temp model 2')
    train_loss_model_2 = train(insta_model_2, loss, optimizer_2, train_loader, device, epoch, verbose)

    assert compare_model_dicts(original_dict, model.state_dict()), '\r[instability analysis] MODEL COPY NOT DEEPCOPY'

    # # compare the two through instability analysis
    # method = cosine_similarity
    # analysis = method(insta_model_1, insta_model_2)
    #
    # with open(f'epoch_{epoch}-method_{method.__name__}-instability_{analysis}.txt', 'w') as f:
    #     print(f'\r[instability analysis] Writing instability analysis results to epoch_{epoch}-instability_{analysis}.txt')
    #     f.write('Instability analysis method: '+method.__name__+'\nInstability analysis: '+analysis)

    print('l2 distance')
    method = classification_differences
    analysis = method(insta_model_1, insta_model_2, test_loader, loss)

    with open(os.path.join('Output', f'epoch_{epoch}-method_{method.__name__}-instability_{analysis}.txt'), 'w') as f:
        print(f'\r[instability analysis] Writing instability analysis results to epoch_{epoch}-instability_{analysis}.txt')
        f.write('Instability analysis method: '+method.__name__+'\nInstability analysis: '+str(analysis))

    return

from torch.nn import functional as F

def cosine_similarity(model1, model2):
    # compute once in pytorch
    # x = torch.randn(32, 100, 25)
    # compares similarity of two models (??)
    for layer1, layer2 in zip(model1.parameters(), model2.parameters()):
        # y = F.cosine_similarity(model1[..., None, :, :], model2[..., :, None, :], dim=-1)
        y = F.cosine_similarity(layer1, layer2, dim=0)
        print(y)

    # y should be the shape of something, has to do with dim given in cosine_similarity
    # assert y.shape == torch.Size([32, 100, 100])

    return y

def pnorm_label_distance(model1, model2, dataloader, p=2.0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    embeds1 = []
    embeds2 = []
    labels = []
    for data in dataloader:
        samples, _labels = data[0].to(device), data[1]
        out1 = model1(samples)
        out2 = model2(samples)
        embeds1.append(out1)
        embeds2.append(out2)
        labels.append(_labels)

    # with p=2.0, this calculates the l2 distance
    dists = torch.cdist(embeds1, embeds2)
    # x = torch.randn(32, 100, 25)
    # compares distance between two models (??)
    # y = torch.cdist(model1, model2, p)
    # print(y)

    # torch.cdist(x1, x2, p)
    # If x1 has shape B×P×M and x2 has shape B×R×M then the output will have shape B×P×R

    return dists

def pnorm_tensor_distance(model1, model2, p=2.0):
    print(model1)
    dists = []
    for layer1, layer2 in zip(model1.parameters(), model2.parameters()):
        # for layer in m.children():
        #     weights = list(layer.parameters())

        # If x1 has shape B×P×M and x2 has shape B×R×M then the output will have shape B×P×R
        print(layer1)
        layer1_size = list(layer1.size())
        layer2_size = list(layer2.size())
        print(layer1_size)
        print(layer2_size)

        if len(layer1_size) == 3 and len(layer2_size) == 3:
            # with p=2.0, this calculates the l2 distance
            cdist = torch.cdist(layer1, layer2, p=p)
            print(f'\r[instability analysis] cdist value {cdist}')
            dists.append(cdist)

    return dists

def classification_differences(model1, model2, test_loader, loss, verbose=True):
    '''
    calculates the difference in predictions between two models
    :param model1: model, e.g. ResNet object
    :param model2: model, e.g. ResNet object
    :param test_loader: dataset
    :param loss: loss function
    :param verbose: bool for verbose
    :return: classification difference (percentage of max diff preds / diff preds)
                where 0% is the best and 100% the worst
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get top 1 accuracies and the predictions for every image
    # top 1 accuracy is the accuracy for all predictions where the true class is equal to the top class predicted
    model1_acc1, model1_preds = eval_preds(model1, loss, test_loader, device, verbose)
    model2_acc1, model2_preds = eval_preds(model2, loss, test_loader, device, verbose)
    # count the number of times the two models predict differently
    eval_diff = sum(1 for i, j in zip(model1_preds, model2_preds) if i != j)
    print('evall_diff', eval_diff)

    # get average error of the two models
    model1_error = 1 - model1_acc1/100
    print('model1_error', model1_error)
    model2_error = 1 - model2_acc1/100
    print('model2_error', model2_error)
    avg_error = (model1_error+model2_error) / 2
    print('avg_error', avg_error)

    # max possible value is average wrong preds per model times number of models (2)
    avg_wrong_preds = len(test_loader.dataset) * avg_error
    print('avg_wrong_preds', avg_wrong_preds)
    max_possible_value = avg_wrong_preds * 2
    print('max_possible_value', max_possible_value)
    # classification diff is max different preds divided by actual different preds
    if max_possible_value is not 0:
        classification_diff = eval_diff / max_possible_value
    else:
        classification_diff = 0
    print('classification_diff', classification_diff)

    print(f'\r[instability analysis] Classification differences for test eval was {eval_diff} on test dataset of length'
          f' {len(test_loader.dataset)} with max diff being {max_possible_value}')

    return classification_diff*100

