import torch.nn.functional as F
import torch


logits = torch.rand(4,6,100)
labels = torch.zeros(4,6).long()
loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=0)

print('loss: ', loss)


loss2 = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=0, reduction='sum')
print('loss2: ', loss2)

sample_size = torch.sum(labels.ne(0))
print('sample_size: ', sample_size)

print(loss2/(sample_size+1))