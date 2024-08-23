# Necessory packages
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm
import plotly.express as px
import yfinance as yf
import pandas as pd
import numpy as np

class Simulation():
    def __init__(self, tickers_list: list, simulation_period: int, start_date: str, end_date: str):
        self.tickers_list = tickers_list
        self.simulation_period = simulation_period
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {'Price': pd.DataFrame(), 'Return': pd.DataFrame()}
        self.sim_capital = {}
        self.sim_dates = []
        self.log = None
        
    def gather_data(self):
        for ticker in self.tickers_list:
            data = yf.Ticker(ticker).history(interval = '1wk', start = self.start_date, end = self.end_date)
            self.stock_data['Price'][ticker] = data['Open']
            self.stock_data['Return'][ticker] = (data['Close']) / (data['Open'])
        
        self.stock_data['Price'].sort_index(inplace = True, ascending = False)
        self.stock_data['Return'].sort_index(inplace = True, ascending = False)
    
    def run_simulation(self, capital = 100):
        self.gather_data()
        capital_dict = {'Portfolio': capital}
        self.sim_capital = {'Portfolio': []}
        
        log_dict = {}
        for ticker in self.tickers_list:
            capital_dict[ticker] = 100
            self.sim_capital[ticker] = []
            log_dict[f'{ticker} Capital assigned'] = []
            log_dict[f'{ticker} Return'] = []
            log_dict[f'{ticker} Capital Return'] = []
            log_dict[f'{ticker} Gain'] = []
            
        log_dict[f'Portfolio Price'] = []
        log_dict[f'Portfolio Return'] = []

        self.log = pd.DataFrame(log_dict)

        for i in range(self.simulation_period):
            row_no = (self.simulation_period - 1 - i)
            date = str(self.stock_data['Price'].iloc[row_no].name)
            stock_prices = self.stock_data['Price'].iloc[row_no].values
            returns = self.stock_data['Return'].iloc[row_no]
            
            asset_dist = self.Markowitz_Model(self.stock_data['Return'].iloc[row_no:(row_no + 100)], 3)
            
            weights = (capital_dict['Portfolio'])* (asset_dist / stock_prices)
            # print(weights)
            
            log_list = []
            for ticker, asset, return_value in zip(self.tickers_list, list(asset_dist), list(returns)):
                cap_assigned = asset*capital_dict['Portfolio']
                log_list.append(cap_assigned)
                log_list.append(return_value)
                log_list.append(return_value*cap_assigned)
                log_list.append((return_value-1)*cap_assigned)
            
            for ticker in self.tickers_list:
                capital_dict[ticker] = capital_dict[ticker] * returns.loc[ticker]
            returns = returns.values
            
            log_list.append(capital_dict['Portfolio'])
            capital_dict['Portfolio'] = capital_dict['Portfolio']*np.dot(asset_dist, returns)
            log_list.append(capital_dict['Portfolio'])
            
            self.log.loc[date] = log_list
            
            for stock, capital in capital_dict.items():
                self.sim_capital[stock].append(capital)
            self.sim_dates.append(date)
            
            print(capital_dict['Portfolio'])
            if capital_dict['Portfolio'] <= 0: break
        
        # print(self.sim_dates)
        # print(self.sim_capital)
        self.log.to_excel('log.xlsx')
        self.plot_sim_capital(self.sim_capital, self.sim_dates)
    
    def plot_sim_capital(self, capital_data_dict, dates_list):
        fig = go.Figure()
        
        for stock, capital_data in capital_data_dict.items():
            fig.add_trace(go.Scatter(x = dates_list, y = capital_data,
                                     mode = 'lines', name = stock))
        
        fig.update_xaxes(tickfont_size = 18, title = 'Dates', titlefont_size = 20)
        fig.update_yaxes(tickfont_size = 16, title = 'Capital (in $)', titlefont_size = 20)            
        fig.update_layout(width = 1100, height = 600, title = 'Returns Plot', titlefont_size = 30)

        fig.write_html('Returns Plot.html', full_html = False)
        fig.show()
    
    def Markowitz_Model(self, train_data, rap: float):
        d = len(train_data.columns)
        cov_mat = train_data.cov().values
        cov_mat_inv = np.linalg.inv(cov_mat)
        mu = train_data.mean().values
        lambda_const = (1 - (1/rap)*np.dot(np.ones(d), np.dot(cov_mat_inv, mu)))
        lambda_const = lambda_const / ((1/rap)*np.dot(np.ones(d), np.dot(cov_mat_inv, np.ones(d))))
        asset_dist = (1/rap)*np.dot(cov_mat_inv, (mu + lambda_const*np.ones(d)))
        
        return asset_dist


tickers_list = ['BX', 'SPY', 'MSFT', 'V']
simulation = Simulation(tickers_list, 50, '2018-01-01', '2024-01-01')
simulation.run_simulation()
