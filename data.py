import numpy as np

y = np.load('tags array.npy')
X = np.load('feats array.npy')

min_samples = 20

tag_dict = {}

for i in y:
	if i not in tag_dict.keys():
		tag_dict[i] = 1
	else:
		tag_dict[i] += 1

with open('our species list.txt', 'w') as file:
	for name in tag_dict.keys():
		file.write(f'{name}\n')



del_list = [y for y in tag_dict.keys() if tag_dict[y] < min_samples]

print(del_list)

count = 0

for item in del_list:
	count += tag_dict[item]

del_indexes = []

for index, tag in enumerate(y):
	if tag in del_list:
		del_indexes.append(index)

X_new = np.delete(X, del_indexes, 0)
y_new = np.delete(y, del_indexes, 0)


print(X.shape)
print(y.shape)
print(X_new.shape)
print(y_new.shape)

np.save('trimmed feats array.npy', X_new)
np.save('trimmed tags array.npy', y_new)

#new_X = np.delete(X, np.where(y in del_list))

