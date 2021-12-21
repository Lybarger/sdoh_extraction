import torch

n = 1000
m = 1000
size = (n, n, n)

x = torch.rand(size, device=0)
# x = x.to(0)

print('x', x.shape, x.device)
print('start')

for i in range(m):

    a = x*x
    z = a.sum()

    if i % 100 == 0:
        print(f'{i}/{m}', z)

print('end', i)
