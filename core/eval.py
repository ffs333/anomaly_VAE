import random

import torch

from tools.gener import draw
from metrics.accuracy import AccuracyMetric


def evaluation(model, eval_loader, epoch, mfcc_step,
               mel_step, output_spec):
    """
    Model evaluation function
    :param model: (pytorch nn.Module) model to evaluate
    :param eval_loader: (pytorch DataLoader) instance of class to read eval data
    :param epoch: (int) epochs number
    :param mfcc_step: (int) save mfcc spectrogram every 'mfcc_step' epochs
    :param mel_step: (int) save mel spectrogram every 'mel_step' epochs
    :param output_spec: (str) path to save spectrogram outputs
    :return: (Tensor) loss function value
    """

    model.eval()

    epoch_loss = None
    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):

            data = batch

            pred, input, mu, log_var = model.forward(data)

            if epoch % mfcc_step == 0 and epoch != 0 and idx == 0:
                index = random.choice(range(int(pred.size(0))))
                draw(data[index], pred[index], epoch, mel_step, output_spec, ev=True)

            losses = model.loss_function(pred, input, mu, log_var, M_N=data.size(0))

            loss_val = losses['loss']
            loss_recons = losses["Reconstruction_Loss"]
            loss_KLD = losses["KLD"]

            if epoch_loss is None:
                epoch_loss = loss_val
                epoch_recons = losses["Reconstruction_Loss"]
                epoch_KLD = losses["KLD"]
            else:
                n = idx + 1
                epoch_loss = (epoch_loss * n / (n + 1)) + loss_val / (n + 1)
                epoch_recons = (epoch_recons * n / (n + 1)) + loss_recons / (n + 1)
                epoch_KLD = (epoch_KLD * n / (n + 1)) + loss_KLD / (n + 1)

    print(f'Eval epoch {epoch} loss: {epoch_loss}')
    print(f'KLD: {losses["KLD"]}  Recons: {losses["Reconstruction_Loss"]}\n')

    return epoch_loss


def evaluation_class(model, eval_loader, epoch, metric, num_classes=1):
    """
    Model evaluation function
    :param model: (pytorch nn.Module) model to evaluate
    :param eval_loader: (pytorch DataLoader) instance of class to read eval data
    :param epoch: (int) epochs number
    :param metric: (class) accuracy metric instance
    :param num_classes: (int)
    :return: (Tensor) loss function value
    """

    model.eval()
    metric.flush()
    eval_loss = None
    if num_classes > 1:
        class_metrics = {}
        for i in range(num_classes):
            class_metrics[f'acc_{i}'] = AccuracyMetric(silence=True)
    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):

            data, label = batch
            model_out = model.forward(data)

            if num_classes == 1:
                loss_val = model.loss_function(model_out, label.unsqueeze(1))
            else:
                loss_val = model.loss_function(model_out, label.squeeze().long())

            if eval_loss is None:
                eval_loss = loss_val

            else:
                n = idx + 1
                eval_loss = (eval_loss * n / (n + 1)) + loss_val / (n + 1)
            if num_classes == 1:
                model_out = torch.round(model_out)
            else:
                model_out = torch.argmax(model_out, dim=1)
            metric.update(torch.squeeze(model_out), label)
            if num_classes > 1:
                for i in range(num_classes):
                    low_index = (label == i).nonzero().squeeze()
                    class_metrics[f'acc_{i}'].update(torch.index_select(model_out, 0, low_index),
                                                     torch.index_select(label, 0, low_index))

    if num_classes > 1:
        for i in range(num_classes):
            cl_acc = round(class_metrics[f"acc_{i}"].compute().item(), 3)
            if i % 3 == 0 and i != 0:
                print(f'Accuracy for class {i}: {cl_acc}')
            else:
                print(f'Accuracy for class {i}: {cl_acc}', end='  |  ')

    print(f'\nEval epoch {epoch} loss: {eval_loss}')

    if eval_loss is None:
        return metric.compute()
    return eval_loss, metric.compute()