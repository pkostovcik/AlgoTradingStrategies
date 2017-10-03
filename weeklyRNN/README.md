### Weekly buy and sell with RNN
Application of recurrent neural networks in stock trading. We are rebalancing portfolio every 2 weeks and weights are chosen by RNN. Strategy can be modified or improved in many ways, this is only the basic idea. Change of layers or nodes in RNN can boost performance. Weights for shorting can be added, also some risk management and so on. 

Using more than 10 stock is harder to train (it needs few tousands of epochs). We can use database with all NASDAQ 100 stocks but every 2 months choose only 10 by some criterium (like one in BuyAndHold). Strategy isn't robust because wasn't backtested during crisis (2008). 

I used split adjusted daily data from nyse which are freely downloadable - I chose stocks from NASDAQ100 list which IPOs were before 2012. Than I chose 10 stocks what can generate look ahead bias because we know that these stocks had really good performance last years.
