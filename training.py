import torch
from torch.utils.tensorboard import SummaryWriter
import time


def train(model, train_dataloader, epochs, lr, loss_fn, val_dataloader=None, clip_grad=False, method=None, input_sequence=None, dir=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    writer = SummaryWriter(f"{dir}/runs")
    train_losses = []
    for epoch in range(epochs):

        for step, (model_input, gt) in enumerate(train_dataloader):
            start_time = time.time()
        
            model_output = model(model_input)
            losses = loss_fn(model_output, gt)

            train_loss = 0.
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                train_loss += single_loss

            train_losses.append(train_loss.item())

            optim.zero_grad()
            train_loss.backward()

            if clip_grad:
                if isinstance(clip_grad, bool):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optim.step()

        #print("Running validation set...")
        model.eval()
        with torch.no_grad():
            val_loss = 0.
            for (model_input, gt) in val_dataloader:
                model_output = model(model_input)
                losses = loss_fn(model_output, gt)
            for loss_name, loss in losses.items():
                single_loss = loss.mean()
                val_loss += single_loss

        if epoch % 5==0 or epoch == epochs -1:
            iteration_time = time.time() - start_time 
            print("Epoch %d, Train loss %0.6f, Val loss %0.6f, iteration time %0.6f" % (epoch, train_loss, val_loss, iteration_time))
            writer.add_scalar("Loss/Train", train_loss, epoch)
            if val_dataloader is not None:
                writer.add_scalar("Loss/Validation", val_loss, epoch)    
            writer.add_scalar("Metrics/Iteration_Time", iteration_time, epoch)
            writer.flush()
        
        model.train()


