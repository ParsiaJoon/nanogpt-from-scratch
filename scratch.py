import torch

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
chars = sorted(set(text))
print(len(chars))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(s):
    result = []
    
    for ch in s:
        result.append(stoi[ch])
    return result

def decode(l):
    characters = []
    
    for i in l:
        characters.append(itos[i])
    return ''.join(characters)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(len(train_data), len(val_data))


def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    listx = []
    listy = []
    for i in ix:
        listx.append(data[i:i+block_size])
        listy.append(data[i+1:i+block_size+1])
    x = torch.stack(listx)
    y = torch.stack(listy)
    return x, y

d = torch.arange(20)
for _ in range(1000):
    xb, yb = get_batch(d, block_size=8, batch_size=16)
    assert xb.shape == yb.shape == (16, 8)
    assert torch.equal(yb, xb + 1)
    assert yb.max() <= 19
seen = torch.cat([torch.cat(get_batch(d, 8, 16)) for _ in range(500)])
assert seen.min() == 0 and seen.max() == 19
print("get_batch OK")