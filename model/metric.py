import torch


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def relative_error(output, target):
    with torch.no_grad():
        assert output.shape == target.shape
        return torch.mean(torch.abs(torch.div(output.clone().detach()-target.clone().detach(), target.clone().detach())))

def absolute_error(output, target):
    with torch.no_grad():
        assert output.shape == target.shape
        return torch.mean(torch.abs(output.clone().detach()-target.clone().detach()))

def pearson_correlation(output, target):
    x = output.clone().detach()
    y = target.clone().detach()
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))