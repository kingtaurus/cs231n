
def sliding_window(image, stride=(16,16), window_size):
	for y in range(0, image.shape[0], stride[1]):
		for x in range(0, image.shape[1], stride[0]):
			yield(x, y, image[y,y+window_size[1], x:x+window_size[0]])
