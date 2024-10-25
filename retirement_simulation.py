import yfinance as yf
import numpy as np
import pandas as pd

starting_corpus = 1000000
annual_expense = 50000
inflation = 0.02
portfolio_alloc_pre = {'bonds': 0.3, 'stocks': 0.7}
portfolio_alloc_post = {'bonds': 0.6, 'stocks': 0.4}
country = "US"
years_to_retirement = 15
horizon = 30
annual_addition = 20000

SIMULATIONS = 1000

# Fetch historical returns for stocks and bonds using yfinance
def get_historical_returns(symbol, start_year, end_year):
    stock_data = yf.download(symbol, start=f'{start_year}-01-01', end=f'{end_year}-12-31', progress=False)
    # Compute annual returns
    stock_data['Return'] = stock_data['Adj Close'].pct_change().resample('Y').last()
    return stock_data['Return'].dropna()


# Fetch stock and bond returns based on country and simulate based on historical samples
def get_country_returns(country, years):
    if country == "US":
        stock_returns = get_historical_returns('^GSPC', '1980', '2023')  # S&P 500 Index
        bond_returns = get_historical_returns('^IRX', '1980', '2023')  # US 3 Month Treasury Bill
    else:
        raise ValueError("Country data not available")

    stock_sample = np.random.choice(stock_returns, size=(years, SIMULATIONS), replace=True)
    bond_sample = np.random.choice(bond_returns, size=(years, SIMULATIONS), replace=True)

    return stock_sample, bond_sample


def simulate_retirement(starting_corpus, annual_expense, inflation, portfolio_alloc_pre, portfolio_alloc_post,
                        country, years_to_retirement, horizon, annual_addition):
    results = []

    stock_sample, bond_sample = get_country_returns(country, years_to_retirement + horizon)

    for sim in range(SIMULATIONS):
        balance = starting_corpus
        annual_expense_with_inflation = annual_expense
        balances = []
        likelihood_out_of_money = []

        for year in range(years_to_retirement + horizon):
            if year < years_to_retirement:
                portfolio_returns = (portfolio_alloc_pre['bonds'] * bond_sample[year][sim] +
                                     portfolio_alloc_pre['stocks'] * stock_sample[year][sim])
            else:
                portfolio_returns = (portfolio_alloc_post['bonds'] * bond_sample[year][sim] +
                                     portfolio_alloc_post['stocks'] * stock_sample[year][sim])

            balance -= annual_expense_with_inflation

            if year < years_to_retirement:
                balance += annual_addition

            balance *= (1 + portfolio_returns)

            annual_expense_with_inflation *= (1 + inflation)

            balances.append(balance)

            if balance < 0:
                likelihood_out_of_money.append(1)
            else:
                likelihood_out_of_money.append(0)

        results.append({
            'balances': balances,
            'likelihood_out_of_money': likelihood_out_of_money
        })

    final_balances = np.mean([result['balances'] for result in results], axis=0)
    out_of_money_prob = np.mean([result['likelihood_out_of_money'] for result in results], axis=0)

    output_df = pd.DataFrame({
        'Year': np.arange(1, years_to_retirement + horizon + 1),
        'Expected_Balance': final_balances,
        'Likelihood_Out_Of_Money': out_of_money_prob
    })

    return output_df

output_df = simulate_retirement(starting_corpus, annual_expense, inflation, portfolio_alloc_pre, portfolio_alloc_post,
                                country, years_to_retirement, horizon, annual_addition)

print(output_df)
