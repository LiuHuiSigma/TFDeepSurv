import matplotlib.pyplot as plt
import seaborn as sns

def plot_train_curve(L, title='Training Curve', x_point=5):
	x = range(1, len(L) + 1)
	xticks = []
	for i in x:
		if i % (len(L)/x_point) == 0:
			xticks.append(i)
		else:
			xticks.append('')
	xticks[0] = '1'
	plt.plot(x, L)
	plt.xticks(x, xticks)
	plt.title(title)
	plt.show()
