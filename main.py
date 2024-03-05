import json

data = None
with open('./vocab.json') as f:
    data = json.load(f)

print(len(data))

with open('vocab.bin', 'wb') as bin_file:
    bin_file.write(len(data).to_bytes(4, byteorder="big"))

    for word in data.keys():
        bin_file.write(word.encode('utf-8') + b'\n')
