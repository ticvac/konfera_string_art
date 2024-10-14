import numpy as np

h, w = 5, 60
h, w = w, h
mat = np.arange(0, h * w).reshape(h, w)

k = min(mat.shape) / max(mat.shape)
print(k)
x, y = 0, 0

to_right = mat.shape[0] < mat.shape[1]
print(to_right)

print(mat)
out = []

while max(x, y) != max(mat.shape):
    print()
    print()
    print(int(y), int(x), mat.shape)
    print(x, y)
    out.append(mat[int(y), int(x)])
    if to_right:
        x += 1
        y += k
    else:
        y += 1
        x += k


print(len(out))
