import numpy as np
import matplotlib.pyplot as plt


def coeff(x, y):
    n = np.size(x)
    mx = np.mean(x)
    my = np.mean(y)
    SS_xy = np.sum(x*y - n*mx*my)
    SS_xx = np.sum(x*x - n*mx*mx)
    c1 = SS_xy / SS_xx
    c0 = my - c1*mx
    
    return(c0, c1)

def plot_regression_line(x, y, c):
    plt.scatter(x, y, color="g", s=30)
    yPred = c[0] + c[1]*x
    plt.plot(x, yPred, color="r")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression')
    plt.show()


def main():
    x = np.array([100, 105, 110, 115, 120, 125, 130, 135])
    y = np.array([110, 115, 120, 125, 130, 135, 140, 145])
    c = coeff(x, y)
    print("Coefficients:\nc0 = {} \
            \nc1 = {}".format(c[0], c[1]))
    plot_regression_line(x, y, c)

if __name__ == "__main__":
	main()


