import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from mpl_toolkits.mplot3d import Axes3D


def plotMomenta(momenta:pd.DataFrame,periods:list[int]):
    for p in periods:
        momenta.loc[p,'momentum'].plot()
        plt.xlabel('Time')
        plt.ylabel('Momentum')
        plt.title('{}-Period Momentum'.format(p))
        plt.show()
def plotZScores(zscores:pd.DataFrame,periods:list[int]):
    for p in periods:
        zscores.loc[p, 'z'].plot()
        plt.xlabel('Time')
        plt.ylabel('z')
        plt.title('{}-Period Z-Score'.format(p))
        plt.show()

def plotReturns(returns:pd.DataFrame):

    plt.figure()
    returns.plot()
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.title('Strategy Returns')
    plt.show()


def plotReturnsHist(returns:pd.DataFrame):
    mean  = returns.mean()
    plt.figure(figsize=(10, 6))

    sea.histplot(returns, bins=60, color="skyblue", stat="density")
    plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')

    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.title('Distribution of Strategy Returns')
    plt.legend()
    plt.show()

def plotCumulative(cumulative:pd.DataFrame):
    plt.figure()
    cumulative.plot()
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.title('Strategy Cumulative Returns')
    plt.show()

def plotForecast(obs:pd.DataFrame,forecast:pd.DataFrame,n=1):
    selected = np.random.choice(obs.columns, n)
    T = forecast.index
    '''obs.loc[T,selected].plot(label='obs')
    forecast[selected].plot(label='forecast')'''
    plt.plot(obs.loc[T,selected].values, label='{} observed z-score'.format(selected))
    plt.plot(forecast[selected].values, label='{} forecast z-score'.format(selected))
    plt.title('Observed vs Forecast')
    plt.legend()
    plt.show()

def plotzThresh(z:pd.DataFrame,threshold:float):
    zthresh = z.loc[:,(z.dropna().abs()>threshold).any()]
    zthresh.plot()
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'+{threshold}')
    plt.axhline(y=-threshold, color='r', linestyle='--', label=f'-{threshold}')
    plt.xlabel('Time')
    plt.ylabel('z-score')
    plt.legend()
    plt.show()


def PlotSharpeSurface2Param(param_names, Sharpe_master):
    '''Plots Sharpe ratio surface when two parameters are being swept over. Takes param names,
    sharpe mater data frame, rebalancing period and doing.'''
    #TODO: Figure out how to make 3D plot interactive
    '''X = Sharpe_master.index
    Y = Sharpe_master.columns
    Z = Sharpe_master.values.T
    X, Y = np.meshgrid(Y,X)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('{} '.format(param_names[0]))
    ax.set_ylabel('{} '.format(param_names[1]))
    ax.set_zlabel('Sharpe Ratio')
    plt.tight_layout()
    ax.set_box_aspect(None, zoom=0.85)

    plt.show(block=True)'''
    # Assuming df is your DataFrame where the index is 'period' and columns are 'threshold'
    # Prepare data for plotting
    X, Y = np.meshgrid(Sharpe_master.columns.astype(float), Sharpe_master.index.astype(float))
    Z = Sharpe_master.values

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')


    # Add labels
    ax.set_xlabel('{} '.format(param_names[0]))
    ax.set_ylabel('{} '.format(param_names[1]))
    ax.set_zlabel('Sharpe Ratio')
    ax.set_title('Sharpe Ratio Surface')

    # Color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    plt.show()

def PlotCumulative2param(p1_space, p2_space, param_names, cumulative_master):
    '''Plots cumulative returns when two parameters are being swept over. Takes parameter spaces, names, cumulative returns mater data frame,
    rebalancing period and doing string'''

    p1_min, p1_mid, p1_max = p1_space[0], p1_space[(len(p1_space)) // 2], p1_space[- 1]
    p2_min, p2_mid, p2_max = p2_space[0], p2_space[(len(p2_space)) // 2], p2_space[- 1]

    p1_plot = [p1_min, p1_mid, p1_max]
    p2_plot = [p2_min, p2_mid, p2_max]

    plt.figure()
    for i in p1_plot:
        for j in p2_plot:
            cumulative_master.loc[i, j].plot(ylabel='Cumulative Returns', label='{} = {}, {} = {}'.
                                             format(param_names[0], i, param_names[1], j))

    plt.title("Cumulative Returns")
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.show()
def PlotReturnsHist2Param(p1_space, p2_space, param_names,R_master):
    '''Plots returns histograms when 2 parameters are being swept over. Takes parameter spaces, names, returns
     master data frame, rebalancing period, and doing string.. '''

    p1_min, p1_mid, p1_max = p1_space[0], p1_space[(len(p1_space)) // 2], p1_space[- 1]
    p2_min, p2_mid, p2_max = p2_space[0], p2_space[(len(p2_space)) // 2], p2_space[- 1]

    p1_plot = [p1_min, p1_mid, p1_max]
    p2_plot = [p2_min, p2_mid, p2_max]
    nrows = len(p1_plot)
    ncols = len(p2_plot)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10), sharex=True, sharey=True)
    for i, pw1 in enumerate(p1_plot):
        for j, pw2 in enumerate(p2_plot):
            returns = R_master.loc[pw1, pw2]
            axes[i, j].hist(returns, bins=30, alpha=0.7, color='blue')
            axes[i, j].set_title('{} = {}, {} = {}'.format(param_names[0], i, param_names[1], j))
            axes[i, j].set_xlabel('Returns')
            axes[i, j].set_ylabel('Density')
    plt.suptitle('Returns Distribution')
    plt.tight_layout()
    plt.show()
