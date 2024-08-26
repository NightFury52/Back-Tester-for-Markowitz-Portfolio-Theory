The Back-tester runs Simulation starting with a capital of 100$ (taking short position on stocks not allowed). At the start of each period, based on past 30 (it can be changed) periods of Gross Returns data for 4 different stocks, (SPY, MSFT, BX, V in ticker symbols) 
(These stocks can be changed also) the algorithm finds optimal combination of stocks so as to increase expected returns on this period's portfolio and minimize the volatility of the portfolio.
At the end of each period the capital is withdrawn and the next period, we again find optimal diversification of the capital accumulated to invest in the given stocks.
The algorithm has a Risk Aversion Parameter, higher value of which determines that the invester wants low risk and lower value of which signifies that the investor is a higher risk-taker.
The Back-tester creates a log file, which contains the data of each an every period trade commenced and their retuens. Running the Algorithm in any IDE will create an interactive plotly graph of Portfolio
Capital accumulated over time, along with capital accumulated had we invested in the stocks comprising the portfolio individually. Below is a static image of the Capital gains plot.


![Cap Gain](https://github.com/user-attachments/assets/01578061-efc5-4d47-a003-6b72d50551c7)
