import matplotlib.pyplot as plt

def plot_rates(df):
    df.plot(kind="bar", x="Call Against Team", y="rate")
    plt.tight_layout()
    plt.show()
