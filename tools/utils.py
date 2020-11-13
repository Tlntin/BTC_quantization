import torch
from config import Config

config = Config()


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'best_loss': best_loss,
             'state_dict': model.state_dict(),
             'optimizer': optimizer}  # BEST_checkpoint.tar

    torch.save(state, config.checkpoint_file)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, config.best_checkpoint_file)
        print('最佳模型已经保存模型在{}路径'.format(config.best_checkpoint_file))
