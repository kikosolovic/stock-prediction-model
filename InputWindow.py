import matplotlib.pyplot as plt
impath = "/home/kikosolovic/Downloads/image2.jpg"
fig = plt.figure()

y = fig.add_subplot()
y.imshow(impath)
y.axes.get_xaxis().set_visible(False)
y.axes.get_yaxis().set_visible(False)
plt.show()