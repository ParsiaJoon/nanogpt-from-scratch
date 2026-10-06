import torch
# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(len(text))
chars = sorted(set(text))
print(len(chars))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(s):
    result = []
    
    for c in s:
        result.append(stoi[c])
    return result
# encode = lambda s: [stoi[c] for c in s]

def decode(l):
    characters = []
    for i in l:
        characters.append(itos[i])
    return ''.join(characters)
# decode = lambda l: ''.join([itos[i] for i in l])

assert encode("hii") == [stoi['h'], stoi['i'], stoi['i']]
assert decode(encode("hii there")) == "hii there"
assert decode(encode(text[:5000])) == text[:5000]
print("encode/decode OK")

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])  

n = int(0.9*len(data))
train_data = data[:n]
val_data =  data[n:]
block_size = 8
print(len(train_data), len(val_data))

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    
    context_text = decode(context.tolist())
    target_text = decode([target.item()])
    print(f"When the input is {context_text!r}, the target is {target_text!r}")
    

batch_size = 4
block_size = 4
def get_batch(data, block_size, batch_size):
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