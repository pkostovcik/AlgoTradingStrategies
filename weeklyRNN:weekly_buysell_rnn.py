import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import fix_yahoo_finance as yf
from time import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM



class WeeklyRebalanceRNN(object):
    
    def __init__(self, data, rnn_input_window = 10, investment = 100000, quantile = 0.8, 
                 training_window = 250, spx = True, log_info = True):
        """
        Trading strategy: RNN choose which stocks to buy for every 10 trading days
        More stocks == more expensive to train RNN
        RNN is trained in the beginning with 1000+ epochs, every 2 weeks it predicts next 5 or 3 buys,
        before prediction it's retrained with newer data (20+ epochs), every 30 days it's retrained
        with newer data (100+ epochs), training data for RNN are always the last 250 days (a year)
        (after 1000 epochs with 20 stocks we can get 0.5 accuracy, using only 5 stocks gives
        us around 0.7 acc, with my laptop I will use only 10 stocks (3 buy) - 0.6 acc)
        I believe that with better computational power, more neurons in layers and more epochs
        we can get 0.7+ accuracy for 20 stocks.
        
        Strategy can be modified : 
                    - adding some stock picking: pick 20 out of 100 every 2 months, 
                      new RNN is needed == 1000+ epochs every 2 months
                      stock picking = create new special method
                    - picking only 10 stocks with criterium from BuyAndHoldStrategy
                    - changing a number of the neurons and/or add new layers for
                      better accuracy
        ===========================================================================
        inputs:
        data            pandas Panel (items = stock symbols (sorted), major_axis = time_index, 
                        minor_axis = [open-close-low-high-volume])
                        OHLCV can be shuffled, without NaN values 
        """
        self.data = data
        self.shape = self.data.shape
        self.dates = self.data.major_axis
        self.symbols = data.items.values
        self.window = rnn_input_window
        self.investment = investment
        self.quantile = quantile
        self.training_window = training_window
        self.spx_use = spx
        self.log_info = log_info
        self.spx_adj_close = None
        self.train, self.scalers = None, None
        self.X, self.y = self.process_data()
        self.model = None
        self.weights = {}
        self.portfolio_value, self.portfolio = None, None
        self.maxdrawdown = None
        
    def process_data(self):
        self.data = self.data.loc[:,:,["open","close","low","high","volume"]]
        scaler = {}
        train = []
        # scaling everything at once doesn't produce look ahead bias
        for s in self.data.items:
            scaler[s] = MinMaxScaler()
            train.append(scaler[s].fit_transform(self.data.loc[s,:,:]))
        train = np.array(train)
        self.train = train
        self.scalers = scaler
        X = []
        y = []
        # for backtest purposes: last y = stock in 80th percentile of last 10 days % gain, 
        # last X = window of 10 OCLHV, the last OCLHV is -10th in whole dataset
        for i in range(self.window, self.shape[1]-self.window + 1):
            X.append(train[:, i-self.window : i, :])
            y.append(self.data.iloc[:, i+self.window-1, 1]) # only close prices after next 10 days
        X, y = np.array(X), np.array(y)
        
        y_price_compare = self.data.iloc[:, self.window - 1 : self.shape[1]-self.window, 1].values
        y_pct_change = (y - y_price_compare) / y_price_compare # we use pct change for 10 days
        
        y = np.greater_equal(y_pct_change, 
                             np.array([np.percentile(y_pct_change, 100*self.quantile,
                                                     axis=1)]*self.shape[0]).T,
                            ).astype("int")
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])
        return X, y
        
    
    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(None, self.X.shape[2]), return_sequences=True))
        self.model.add(LSTM(units = 32))
        self.model.add(Dense(units = 16, activation="relu"))
        self.model.add(Dense(units = self.shape[0], activation="sigmoid"))
        self.model.compile(optimizer = 'rmsprop', loss = "categorical_crossentropy", metrics=["accuracy"])
    
    def calculate_weights(self, predict):
        w = predict.copy().reshape(-1)
        lowest = np.equal(w <= np.percentile(w,100*self.quantile), 
                          w >= np.percentile(w,100*self.quantile - 10))
        w[w < np.percentile(w,100*self.quantile - 10)] = 0
        w = w/w.sum()
        if w[lowest] < 0.05:
            w[lowest] = 0.1
            w = w/w.sum()
        return w
    
    def backtest(self):
        if self.log_info:
            print("Backtest has begun...")
        start = time()
        self.get_weights_backtest()
        self.portfolio_backtest()
        duration = time() - start
        if self.log_info:
            print(20*"=")
            print("Backtest finished after", np.round(duration) , "s")
        
    def get_weights_backtest(self):
        # we create weights according to RNN model for stock prices
        count = 1
        for i in range(len(self.X)-self.window): 
            if i < self.training_window:
                pass
            elif i == self.training_window:
                # create and fit model for the first time
                if self.log_info:
                    print(20*"=")
                    print("%s/%s train, date %s" % (int((i+10 - self.training_window)/10), 
                                                       int((len(self.X)-self.training_window)/10), 
                                                       self.dates.date[i + 2*self.window -1]))
                self.create_model()
                X, y = self.X[:i], self.y[:i]
                self.model.fit(X, y, epochs = 800, batch_size = 32, verbose = 0)
                self.model.fit(X, y, epochs = 300, batch_size = 32, verbose = 0)
                self.model.fit(X, y, epochs = 100, batch_size = 32, verbose = 0)
                #self.model.fit(X, y, epochs = 200, batch_size = 32, verbose = 0)
                if self.log_info:
                    scores = self.model.evaluate(X, y, verbose=0)
                    print("Initial training finished, accuracy: %.2f%%" % (scores[1]*100))
                prediction = self.model.predict(self.X[i : i+self.window])[-1]
                # the last fitted prices were X[i-1] and y[i-1] is pct_change at time i+window-1
                # prices are known until i + window -1: we use known prices X[i:i+window]
                # to predict pct_change for next window, (at date i + 2*self.window -1)
                self.weights[self.dates[i + 2*self.window -1]] = self.calculate_weights(prediction)
                # we created data X from time self.window (see process data)
                # so real date is i + 2*self.window -1
                count += 1
            else:
                if i%self.window == 0:
                    if self.log_info and count%3 == 0:
                        print(20*"=")
                        print("%s/%s retrain, date %s" % (int((i+10 - self.training_window)/10),
                                                           int((len(self.X)-self.training_window)/10), 
                                                           self.dates.date[i + 2*self.window - 1]))
                    X, y = self.X[i-self.training_window:i], self.y[i-self.training_window:i]
                    self.model.fit(X, y, epochs = 250 if count%3 == 0 else 100, batch_size = 32, verbose = 0)
                    if self.log_info and count%3 == 0:
                        scores = self.model.evaluate(X, y, verbose=0)
                        print("Retrain finished, accuracy: %.2f%%" % (scores[1]*100))
                    prediction = self.model.predict(self.X[i : i+self.window])[-1] # the same as before
                    self.weights[self.dates[i + 2*self.window -1]] = self.calculate_weights(prediction)
                    count += 1
        self.weights = pd.DataFrame(self.weights).T
        self.weights.columns = self.symbols
    
    def portfolio_backtest(self):
        # separately we calculate portfolio values for given weights
        portfolio = []
        portfolio_value = [self.investment,]
        ind = self.data.loc[:, self.weights.index[0] : ,:].major_axis
        for date in ind:
            if date == self.weights.index[0]:
                portfolio.append(self.weights.loc[date]/self.data.loc[:,date,"close"].values*portfolio_value[-1])
            else:
                portfolio_value.append(portfolio[-1].dot(self.data.loc[:,date,"close"].values))
                if date in self.weights.index:
                    portfolio.append(self.weights.loc[date]/self.data.loc[:,date,"close"].values*portfolio_value[-1])
        self.portfolio = pd.DataFrame(portfolio, index = self.weights.index, columns = self.symbols)
        self.portfolio_value = pd.Series(portfolio_value, index = ind)
    
    
    def plot(self, figure = None, save = None):
        plt.rcParams["font.family"] = "Cambria"
        if figure is None:
            figure = plt.figure(figsize=(8,6))
        plt.subplot2grid((5,1),(0,0),rowspan=3)
        plt.plot(self.portfolio_value.index, self.portfolio_value.values, label = "RNNstrategy")
        plt.scatter(self.weights.index, self.portfolio_value[self.weights.index], c = "r", s = 15, 
                    edgecolors="k", alpha = 0.8, label = "rebalance")
        if self.spx_use:
            self.download_spx_prices(self.portfolio_value.index[0],
                                     self.portfolio_value.index[-1] + dt.timedelta(1))
        if self.spx_use:
            plt.plot(self.spx_adj_close/self.spx_adj_close.values[0]*self.investment,
                     lw=1.3, color = "g", label = "SPX")
        plt.legend()        
        plt.xticks(pd.date_range(start = dt.datetime(int(self.portfolio_value.index[0].year),1,1), 
                                 end = self.portfolio_value.index[-1], freq="AS"))
        plt.tick_params(axis="x", labelbottom = "off")
        plt.xlabel("")
        plt.xlim(self.portfolio_value.index[0], self.portfolio_value.index[-1])
        plt.suptitle("Weekly Rebalance with RNN", fontsize=15, fontweight='bold')
        plt.title("investment = %s " %(self.investment), fontsize=12)
                
        plt.subplot(5,1,4)
        self.maxdrawdown = self.max_drawdown_calculate()
        plt.fill_between(self.maxdrawdown.index, y1= np.zeros(len(self.maxdrawdown)), 
                         y2 = self.maxdrawdown.values.T[0]*100, color="r", alpha=0.5)
        plt.xticks(pd.date_range(start = dt.datetime(int(self.portfolio_value.index[0].year),1,1), 
                                 end=self.portfolio_value.index[-1], freq="AS"))
        plt.tick_params(axis="x", labelbottom = "off")
        plt.xlabel("")
        plt.xlim(self.portfolio_value.index[0], self.portfolio_value.index[-1])
        plt.ylabel("%")
        plt.yticks(np.linspace((self.maxdrawdown*10).min().round()*10, 0.0, 4))
        plt.title("Maximum Drawdown")

        
        plt.subplot(5,1,5)
        a = self.calculate_annual_returns()
        ix = a.index
        a.index = pd.to_datetime((ix - dt.timedelta(365/2)))
        k = (a.index[-1] - a.index[-2]).days if len(a)>=2 else 321
        plt.bar(a.index,100*a.values, width = (len(a)-1)*[200]+[180] if k<320 else 200,
                alpha=0.5)
        if self.spx_use:
            b = self.calculate_annual_returns(spx = True)
            b.index = pd.to_datetime((ix - dt.timedelta(365/2)))
            plt.bar(b.index,100*b.values, width = (len(a)-1)*[200]+[180] if k<320 else 200,
                    alpha=0.5)
        plt.xticks(pd.date_range(start = dt.datetime(int(self.portfolio_value.index[0].year),1,1), 
                                 end = self.portfolio_value.index[-1], freq="AS"), rotation = 45)
        plt.xlim(self.portfolio_value.index[0], self.portfolio_value.index[-1])
        plt.ylabel("%")
        plt.yticks(np.linspace(((a.values*10).round()*10).min(), ((a.values*10).round()*10).max(),4))
        plt.hlines(0, xmin = self.portfolio_value.index[0], xmax = self.portfolio_value.index[-1], 
                   color = "k", lw = 1, linestyle="--")
        plt.title("Annual returns")

        
        if save is not None:
            plt.savefig(save)
        plt.show()
    
    
    def max_drawdown_calculate(self):
        maxdrawdown = []
        maximum = 0
        for i in range(len(self.portfolio_value)):
            s = self.portfolio_value.values[i]
            if s >= maximum:
                maximum = s
                maxdrawdown.append(0)
            else:
                maxdrawdown.append((s-maximum)/maximum)
        maxdrawdown = pd.DataFrame(maxdrawdown, index=self.portfolio_value.index)    
        return maxdrawdown
    
    def download_spx_prices(self, start, end):
        try:
            s = str(str(start.year)+"-"+str(start.month)+"-"+str(start.day))
            e = str(str(end.year)+"-"+str(end.month)+"-"+str(end.day))
            if self.spx_adj_close is None:
                spx = yf.download("^GSPC", start = s, end = e)
                self.spx_adj_close = spx["Adj Close"]
        except KeyError:
            self.spx_use = False
            print("  ==================================   ")
            print(" fix_yahoo_finance downloading error!   ")
            print("Please restart the kernel and try again.")
            
    
    
    def calculate_annual_returns(self, spx = False):
        returns = []
        dates = np.arange(0,len(self.portfolio_value),250)
        if (self.portfolio_value.index[-1] - self.portfolio_value.index[dates[-1]]).days > 200:
            dates = np.append(dates,len(self.portfolio_value)-1)
        if not spx:
               returns = self.portfolio_value.iloc[dates].pct_change()[1:]
        else:
               returns = self.spx_adj_close.iloc[dates].pct_change()[1:]
        returns = pd.DataFrame(returns)
        returns.columns = ["returns"]
        return returns
