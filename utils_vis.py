import matplotlib.pyplot as plt
import seaborn as sns

def plot_train_curve(L, title='Training Curve'):
	x = range(1, len(L) + 1)
	plt.plot(x, L)
	# no ticks
	plt.xticks([])
	plt.title(title)
	plt.show()