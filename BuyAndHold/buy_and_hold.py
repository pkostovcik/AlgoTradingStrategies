# strategy 
import numpy as np
import pandas as pd
import fix_yahoo_finance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

class BuyAndHoldStrategy(object):
    def __init__(self, data, window = 100, quantile = 0.9, investment = 10000, days = 250, 
                 spx = True, maxdrawdown = True):
        self.data = data
        self.window = window
        self.quantile = quantile
        self.investment = investment
        self.days = days
        self.nstocks = len(self.data.columns)
        self.counts = self.calculate_counts()
        self.buy_dates = self.set_buying_dates()
        self.buy_weights = self.create_weights()
        self.portfolio = self.create_portfolio()
        self.strategy_income = self.backtest_portfolio_income()
        self.annual_returns = self.calculate_annual_returns()
        self.spx_adj_close = None
        self.sxp_returns = None
        self.spx_buyandhold_income = None
        self.spx_use = spx
        if spx:
            self.spx_returns_compare()
        if maxdrawdown:
            self.maxdrawdown = self.max_drawdown_calculate()
        self.summary = self.buying_summary()
        
    def calculate_counts(self):
        # returns dataframe where values represent the number when the price was above quantile
        returns = self.data.pct_change()
        percentile = returns.quantile(self.quantile, axis=1)
        ismax = {}
        for stock in self.data.columns:
            ismax[stock] = returns[stock] >= percentile
        ismax = pd.DataFrame(ismax).astype(int)
        return ismax.rolling(self.window).sum()        
    
    def calculate_buy_weights(self, index):
        # calculates weights for buying in certain date (index)
        series = self.counts.loc[index]
        weights = series.sort_values(ascending=False)[:5]
        weights = weights/weights.sum()
        buy = pd.Series(data = np.zeros(self.nstocks), index=self.data.columns)
        for stock in weights.index.values:
            buy[stock] = weights[stock]
        return buy    
       
    def portfolio_value(self, index, portfolio):
        # calculates portfolio value at some index
        return portfolio.dot(self.data.loc[index])
    
    def update_portfolio(self, index, portfolio):
        # updates portfolio by adding bought stocks
        return portfolio + self.buy_weights[index]/self.data.loc[index]*self.investment
    
    def set_buying_dates(self):
        # we will be buying on these dates
        date_index = np.arange(self.window-1,len(self.data),self.days)
        return self.data.index[date_index]
    
    def create_weights(self):
        # creates weights on every buying date
        weights = {}
        for index in self.buy_dates:
            weights[index] = self.calculate_buy_weights(index)
        return weights
    
    def create_portfolio(self):
        # creates porftolio for every buying date
        portfolio = [np.zeros(self.nstocks),]
        for i in self.buy_dates:
            portfolio.append(self.update_portfolio(i,portfolio[-1]))
        portfolio = portfolio[1:]
        dic = {}
        j=0
        for i in self.buy_dates:
            dic[i] = portfolio[j]
            j += 1
        portfolio = dic
        portfolio = pd.DataFrame(portfolio).T
        portfolio.index = pd.to_datetime(portfolio.index)
        return portfolio
    
    def backtest_portfolio_income(self):
        # calculates results for strategy for every trading day
        portfolio_income = []
        index = None
        start = self.data.index[self.window-1]
        for i in self.data.index:
            if i < start:
                pass
            else:
                if i in self.buy_dates:
                    index = i
            if index is not None:
                portfolio_income.append(self.portfolio_value(i, self.portfolio.loc[index]))
        income = pd.DataFrame(portfolio_income, index = self.data.index[self.window-1:])
        income.columns = ["Buy&Hold"]
        income.index = pd.to_datetime(income.index)
        return income
    
    def spx_returns_compare(self):
        # downloads SPX prices for comparison 
        # and creates SPX BuyAndHold strategy (buying only SPX every year)
        try:
            start = self.strategy_income.index[0]
            end = self.strategy_income.index[-1] + dt.timedelta(1)
            start = str(str(start.year)+"-"+str(start.month)+"-"+str(start.day))
            end = str(str(end.year)+"-"+str(end.month)+"-"+str(end.day))
            if self.spx_adj_close is None:
                spx = yf.download("^GSPC", start = start, end = end)
                self.spx_adj_close = spx["Adj Close"]
            dates = pd.to_datetime(self.buy_dates[1:])
            inv = self.investment
            lst = []
            for i in range(len(self.strategy_income)):
                if self.strategy_income.index[i] in dates:
                    inv += self.investment
                lst.append(self.spx_adj_close.iloc[i]/self.spx_adj_close[0]*inv)
            spx = pd.DataFrame(lst, index=self.strategy_income.index)
            spx.index = pd.to_datetime(spx.index)
            self.spx_buyandhold_income = spx
            self.spx_returns = self.calculate_annual_returns(spx = True)
        except KeyError:
            self.spx_use = False
            print("  ==================================   ")
            print(" fix_yahoo_finance downloading error!   ")
            print("Please restart the kernel and try again.")
  

    
    def plot(self, figure = None, save = None):
        # plot results, max drawdown and yearly returns
        plt.rcParams["font.family"] = "Cambria"
        self.data.index = pd.to_datetime(self.data.index)
        if type(figure) == type(None):
            figure = plt.figure(figsize=(8,6))
        plt.subplot2grid((5,1),(0,0),rowspan=3)
        plt.plot(self.strategy_income.index, self.strategy_income.values, label = "Buy&Hold")
        plt.vlines(self.buy_dates, ymin = np.repeat(0,len(self.buy_dates)), label = "deposit",
                   ymax = 0.9*np.repeat(self.strategy_income.loc[self.buy_dates[-1]],len(self.buy_dates)), 
                   linestyle = "--", lw=1)
        if self.spx_use:
            plt.plot(self.spx_buyandhold_income, lw=1.3, color = "g", label = "SPX")
        plt.legend()        
        plt.xticks(pd.date_range(start = dt.datetime(int(self.data.index[0].year),1,1), 
                                 end=self.strategy_income.index[-1], freq="AS"))
        plt.tick_params(axis="x", labelbottom = "off")
        plt.xlabel("")
        plt.xlim(self.strategy_income.index[0], self.strategy_income.index[-1])
        plt.suptitle("Annual Buy & Hold Strategy", fontsize=15, fontweight='bold')
        plt.title("annual deposit = %s USD, window = %s, quantile = %s" 
                     %(self.investment, self.window, self.quantile), fontsize=12)
                
        plt.subplot(5,1,4)
        plt.fill_between(self.maxdrawdown.index, y1= np.zeros(len(self.maxdrawdown)), 
                         y2 = self.maxdrawdown.values.T[0]*100, color="r", alpha=0.5)
        plt.xticks(pd.date_range(start = dt.datetime(int(self.data.index[0].year),1,1), 
                                 end=self.strategy_income.index[-1], freq="AS"))
        plt.tick_params(axis="x", labelbottom = "off")
        plt.xlabel("")
        plt.xlim(self.strategy_income.index[0], self.strategy_income.index[-1])
        plt.ylabel("%")
        plt.yticks(np.linspace(int((self.maxdrawdown*10).min())*10, 0.0, 4))
        plt.title("Maximum Drawdown")
        
        plt.subplot(5,1,5)
        a = self.annual_returns.copy()
        a.index = pd.to_datetime((self.annual_returns.index - dt.timedelta(6*365/12)))
        plt.bar(a.index,100*a.values, width=250, alpha=0.5)
        if self.spx_use:
            b = self.spx_returns.copy()
            b.index = pd.to_datetime((self.annual_returns.index - dt.timedelta(6*365/12)))
            plt.bar(b.index,100*b.values, width=250, alpha=0.5)
        plt.xticks(pd.date_range(start = dt.datetime(int(self.data.index[0].year),1,1), 
                                 end = self.strategy_income.index[-1], freq="AS"), rotation = 45)
        plt.xlim(self.strategy_income.index[0], self.strategy_income.index[-1])
        plt.ylabel("%")
        plt.yticks(np.arange(((a.values*10).round()*10).min(), ((a.values*10).round()*10).max(),40))
        plt.hlines(0, xmin = self.strategy_income.index[0], xmax = self.strategy_income.index[-1], 
                   color = "k", lw = 1, linestyle="--")
        plt.title("Annual returns")
        
        if save is not None:
            plt.savefig(save)
        plt.show()
    
    def calculate_annual_returns(self, spx = False):
        # calculate yearly returns for strategy itself or for strategy for SPX only
        returns = []
        if not spx:
            for i in range(len(self.buy_dates)):
                if i == 0: 
                    pass
                else:
                    returns.append((self.strategy_income.loc[self.buy_dates[i]] - self.investment 
                                   - self.strategy_income.loc[self.buy_dates[i-1]])/
                                   self.strategy_income.loc[self.buy_dates[i-1]])
        else:
               returns = self.spx_adj_close[self.buy_dates].pct_change()[1:]
        returns = pd.DataFrame(returns, index=pd.to_datetime(self.buy_dates[1:]))
        returns.columns = ["returns"]
        return returns
    
    def buying_summary(self):
        # summary stocks bought every year
        weights = pd.DataFrame(self.buy_weights).T
        buy = []
        for i in range(len(weights.index)):
            index = weights.index[i]
            buy.append(pd.DataFrame((weights.iloc[i]/self.data.loc[index]*
                                     self.investment)[weights.iloc[i] != 0]).round())
        buys = {}
        for i in range(len(weights.index)):
            l = []
            for s in range(5):
                l.append(str(str(int(buy[i].values[:,0][s])) + " " + buy[i].index[s]))
            buys[weights.index[i]] = l
        buys = pd.DataFrame(buys).T
        buys.columns = ["Stock 1", "Stock 2", "Stock 3", "Stock 4", "Stock 5"]
        return buys
    
    def max_drawdown_calculate(self):
        # calculates max drawdown for plots
        maxdrawdown = []
        maximum = 0
        for i in np.arange(len(self.strategy_income)):
            s = self.strategy_income.values[i]
            if s >= maximum:
                maximum = s[0]
                maxdrawdown.append(0)
            else:
                maxdrawdown.append((s[0]-maximum)/maximum)
        maxdrawdown = pd.DataFrame(maxdrawdown, index=self.strategy_income.index)    
        return maxdrawdown

