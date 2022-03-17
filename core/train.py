import random

import torch
from torch.utils.tensorboard import SummaryWriter

from core.eval import evaluation, evaluation_class
from tools.gener import draw
from metrics.accuracy import AccuracyMetric
from core.modules.utils import config_dump


def training(model, epochs, train_dataloader, eval_dataloader, optimizer,
             scheduler, output_check, output_spec, eval_step, mfcc_step,
             mel_step, save_step, conf_dumpers):
    """
    Model training process
    :param model: (pytorch nn.Module) model to train
    :param epochs: (int) number of epochs to train
    :param train_dataloader: (pytorch DataLoader) instance of class to read train data
    :param eval_dataloader: (pytorch DataLoader) instance of class to read eval data
    :param optimizer: (pytorch) entity optimizer
    :param scheduler: (pytorch) optimizer scheduler to control learning rate
    :param output_check: (str) path to save model checkpoints
    :param output_spec: (str) path to save spectrogram outputs
    :param eval_step: (int) make evaluation every 'eval_step' epochs
    :param mfcc_step: (int) save mfcc spectrogram every 'mfcc_step' epochs
    :param mel_step: (int) save mel spectrogram every 'mel_step' epochs
    :param save_step: (int)
    :param conf_dumpers: (list) path to dump; dump index; config data; dump folder; model type
    """
    model.train()
    best_metric_val = None
    writer = SummaryWriter()
    for epoch in range(epochs):
        writer.add_scalar('LR', *scheduler.get_last_lr(), epoch)

        epoch_loss = None

        for idx, batch in enumerate(train_dataloader):

            data = batch
            pred, input, mu, log_var = model.forward(data)

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

            if epoch % mfcc_step == 0 and epoch != 0 and idx == 0:
                index = random.choice(range(int(pred.size(0))))
                draw(data[index], pred[index], epoch, mel_step, output_spec, ev=False)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        writer.add_scalar('Loss Train', epoch_loss, epoch)
        writer.add_scalar('MSE Loss', losses["Reconstruction_Loss"], epoch)
        print(f'Epoch {epoch} loss: {epoch_loss}')
        print(f'KLD: {losses["KLD"]}  Recons: {losses["Reconstruction_Loss"]}')
        scheduler.step()

        if epoch % eval_step == 0:

            eval_loss = evaluation(model, eval_dataloader, epoch, mfcc_step,
                                   mel_step, output_spec)
            model.train()
            writer.add_scalar('Loss Eval', eval_loss, epoch)

        if epoch % save_step == 0 and epoch != 0:
            torch.save(model.state_dict(), f'{output_check}/autoenc_{epoch}.ckpt')

        if best_metric_val is None or eval_loss <= best_metric_val:
            torch.save(model.state_dict(), f'{conf_dumpers[3]}/{conf_dumpers[4]}_best_CHECK_{conf_dumpers[1]}.ckpt')
            conf_dumpers[2]['score'] = eval_loss.item()
            conf_dumpers[2]['recons'] = losses["Reconstruction_Loss"].item()
            config_dump(conf_dumpers[0], conf_dumpers[2])
            best_metric_val = eval_loss


def training_class(model, epochs, train_dataloader, eval_dataloader,
                   optimizer, scheduler, output_check, eval_step, save_step, conf_dumpers, num_classes=1):
    """
    Model training process
    :param model: (pytorch nn.Module) model to train
    :param epochs: (int) number of epochs to train
    :param train_dataloader: (pytorch DataLoader) instance of class to read train data
    :param eval_dataloader: (pytorch DataLoader) instance of class to read eval data
    :param optimizer: (pytorch) entity optimizer
    :param scheduler: (pytorch) optimizer scheduler to control learning rate
    :param output_check: (str) path to save model checkpoints
    :param eval_step: (int) make evaluation every 'eval_step' epochs
    :param save_step: (int)
    :param conf_dumpers: (list) path to dump; dump index; config data; dump folder; model type
    :param num_classes: (int)
    """
    model.train()
    acc_metric = AccuracyMetric()
    best_metric_val = None
    writer = SummaryWriter()
    for epoch in range(epochs):
        writer.add_scalar('LR', *scheduler.get_last_lr(), epoch)

        epoch_loss = None

        for idx, batch in enumerate(train_dataloader):

            data, label = batch
            model_out = model.forward(data)
            if num_classes == 1:
                loss_val = model.loss_function(model_out, label.unsqueeze(1))
            else:
                loss_val = model.loss_function(model_out, label.squeeze().long())

            if epoch_loss is None:
                epoch_loss = loss_val
            else:
                n = idx + 1
                epoch_loss = (epoch_loss * n / (n + 1)) + loss_val / (n + 1)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        writer.add_scalar('Loss Train', epoch_loss, epoch)
        print(f'Epoch {epoch} loss: {epoch_loss}')

        scheduler.step()

        if epoch % eval_step == 0:

            eval_loss, accuracy = evaluation_class(model, eval_dataloader, epoch, acc_metric, num_classes)
            model.train()
            writer.add_scalar('Loss Eval', eval_loss, epoch)
            writer.add_scalar('Accuracy Eval', accuracy, epoch)

        if epoch % save_step == 0 and epoch != 0:
            torch.save(model.state_dict(), f'{output_check}/classifier_{epoch}.ckpt')

        if best_metric_val is None or accuracy >= best_metric_val:
            torch.save(model.state_dict(), f'{conf_dumpers[3]}/{conf_dumpers[4]}_best_CHECK_{conf_dumpers[1]}.ckpt')
            conf_dumpers[2]['score'] = accuracy.item()
            config_dump(conf_dumpers[0], conf_dumpers[2])
            best_metric_val = accuracy