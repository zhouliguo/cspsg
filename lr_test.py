from torch.optim import SGD, lr_scheduler
from torchvision import models
from utils.general import one_cycle
net = models.resnet18()

lr0 = 0.01
lrf = 0.01
momentum = 0.937
cos_lr = False
epochs = 300

optimizer = SGD(net.parameters(), lr=lr0, momentum=momentum, nesterov=True)

# Scheduler
if cos_lr:
    lf = one_cycle(1, lrf, epochs)  # cosine 1->hyp['lrf']
else:
    lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf  # linear
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

for i in range(epochs):
    lr = [x['lr'] for x in optimizer.param_groups]
    print(i, lr)
    optimizer.step()
    scheduler.step()