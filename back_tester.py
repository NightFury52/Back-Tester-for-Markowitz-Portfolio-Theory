# Necessory packages
import plotly.graph_objects as go
from scipy.optimize import minimize
import yfinance as yf
import pandas as pd
import numpy as np

class Simulation():
    def __init__(self, n_assets: int, simulation_period: int, start_date: str, end_date: str):
        self.simulation_period = simulation_period
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {'Price': pd.DataFrame(), 'Return': pd.DataFrame()}
        self.n_assets = n_assets
        self.sim_capital = {}
        self.sim_dates = []
        self.log = None
        
    def gather_data(self, start_date, end_date):
        self.stock_data['Price'] = pd.read_csv('Data/stock_price_data.csv', index_col = 0)
        self.stock_data['Price'] = self.stock_data['Price'].loc[start_date : end_date]
        self.stock_data['Return'] = pd.read_csv('Data/stock_returns_data.csv', index_col = 0)
        self.stock_data['Return'] = self.stock_data['Return'].loc[start_date : end_date]
        
        self.stock_data['Price'].sort_index(inplace = True, ascending = False)
        self.stock_data['Return'].sort_index(inplace = True, ascending = False)
    
    def run_simulation(self, capital = 100):
        # Gathering historical data
        self.gather_data(self.start_date, self.end_date)
        
        # we can't update self.sim_capital rn, so we take a temporary variable
        capital_dict = {'Portfolio': capital}
        self.sim_capital = {'Portfolio': []}
        
        log_dict = {}
        for idx in range(1, self.n_assets + 1):
            log_dict[f'{idx} ticker'] = []
            log_dict[f'{idx} Cap ass.'] = []
            log_dict[f'{idx} Return'] = []
            log_dict[f'{idx} Cap Return'] = []
            log_dict[f'{idx} Cap Gain'] = []
        log_dict[f'Portfolio Cap'] = []
        log_dict[f'Portfolio Return'] = []
        log_dict[f'Portfolio Gain'] = []

        # Initiating the Log File
        self.log = pd.DataFrame(log_dict)

        for i in range(self.simulation_period):
            # current period
            row_no = (self.simulation_period - 1 - i)
            date = str(self.stock_data['Price'].iloc[row_no].name)
            
            returns_data = self.stock_data['Return'].iloc[(row_no + 1):(row_no + 1 + 100)]
            assets = self.select_assets(self.n_assets, returns_data)
            # print(assets)
            returns_data = returns_data[assets]
            
            stock_prices = self.stock_data['Price'].iloc[row_no][assets].values
            returns = self.stock_data['Return'].iloc[row_no][assets].values
            
            # cap_dist is given by the model (proportion of total capital assigned in each stock)
            cap_dist = self.Markowitz_Model(returns_data, 30)
            # print(cap_dist)
            
            # Which stock, how many we're buying
            weights = (capital_dict['Portfolio'])* (cap_dist / stock_prices)
            
            # stores the log for one period
            log_list = []
            for ticker, cap_prop, return_value in zip(assets, list(cap_dist), list(returns)):
                cap_assigned = cap_prop*capital_dict['Portfolio']
                log_list.append(ticker)
                log_list.append(cap_assigned)
                log_list.append(return_value)
                log_list.append(return_value*cap_assigned)
                log_list.append((return_value-1)*cap_assigned)
            
            # updating capital from individual stocks
            # for ticker in assets:
            #     capital_dict[ticker] = capital_dict[ticker] * returns.loc[ticker]
            # returns = returns.values
            
            # updating portfolio capital
            log_list.append(capital_dict['Portfolio'])
            new_capital = capital_dict['Portfolio']*np.dot(cap_dist, returns)
            log_list.append(new_capital)
            log_list.append(new_capital - capital_dict['Portfolio'])
            capital_dict['Portfolio'] = new_capital
            
            # updating the log DataFram
            self.log.loc[date] = log_list
            
            # updating simulation capital and dates
            for stock, capital in capital_dict.items():
                self.sim_capital[stock].append(capital)
            self.sim_dates.append(date)
            
            print(capital_dict['Portfolio'])
            if capital_dict['Portfolio'] <= 0: break
        
        # Storing the Log File
        self.log.to_excel('log.xlsx')
        
        # Plotting the Capital Time-Series
        self.plot_sim_capital(self.sim_capital, self.sim_dates)
    
    def select_assets(self, n_assets: int, returns_data: pd.DataFrame):
        sd = np.sqrt(np.diag(returns_data.cov()))
        mu = returns_data.mean(axis = 0)
        efficiency = (mu / sd)
        efficiency.sort_values(ascending = False, inplace = True)
        best_stocks = list(efficiency.index[:n_assets])
        return best_stocks
    
    def plot_sim_capital(self, capital_data_dict: dict, dates_list: list):
        fig = go.Figure()
        
        for stock, capital_data in capital_data_dict.items():
            fig.add_trace(go.Scatter(x = dates_list, y = capital_data,
                                     mode = 'lines', name = stock))
        
        fig.update_xaxes(tickfont_size = 18, title = 'Dates', titlefont_size = 20)
        fig.update_yaxes(tickfont_size = 16, title = 'Capital (in $)', titlefont_size = 20)            
        fig.update_layout(width = 1100, height = 600, title = 'Returns Plot', titlefont_size = 30)

        # Saving the Plot in a HTML File
        fig.write_html('Returns Plot.html', full_html = True)
        fig.show()
    
    def Markowitz_Model(self, train_data: pd.DataFrame, rap: float):
        d = len(train_data.columns)
        mu = train_data.mean(axis = 0).values
        cov_mat = train_data.cov().values
        
        constraints = [{'type': 'ineq', 'fun': lambda x: x},
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = minimize(self.Markowitz_func, np.zeros(d), args=(mu, cov_mat, rap),
                            method='SLSQP', constraints=constraints)

        cap_dist = result.x
        cap_dist = np.round(np.abs(cap_dist), 3)
        cap_dist = cap_dist / np.sum(cap_dist)
        
        return cap_dist
    
    def Markowitz_func(self, weight, mu, cov_mat, rap):
        # Returns the function which has to be minimized
        L = np.dot(weight, mu) - (rap/2)*np.dot(weight, np.dot(cov_mat, weight))
        return (-L)

simulation = Simulation(30, 250, '2014-01-01', '2018-01-01')
simulation.run_simulation()
