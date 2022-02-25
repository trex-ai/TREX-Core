async def process_ledger(last_deliver, ledger, market_info):
    # strip out time from transactions and simplify for reward calculation
    grid_prices = market_info[str(last_deliver)]['grid']

    # market transactions uses a simplified transaction format
    # it is a dictionary containing lists of transactions ('bid/ask', quantity, price) for each source
    market_transactions = await ledger.get_simplified_transactions(last_deliver)
    extra_transactions = ledger.extra[last_deliver] if last_deliver in ledger.extra else {}

    # grid params
    # grid information is special. It is a tuple of the following format:
    # buy quantity, buy price, sell quantity, sell price
    grid_transactions = ()
    if 'grid' in extra_transactions:
        if extra_transactions['grid']['buy']:
            grid_buy_qty = sum([transaction['quantity'] for transaction in extra_transactions['grid']['buy']])
        else:
            grid_buy_qty = 0

        if extra_transactions['grid']['sell']:
            grid_sell_qty = sum([transaction['quantity'] for transaction in extra_transactions['grid']['sell']])
        else:
            grid_sell_qty = 0

        grid_transactions = (grid_buy_qty, grid_prices['buy_price'], grid_sell_qty, grid_prices['sell_price'])

    # financial transactions only gives financial cost and profit
    # this is OK because the transaction details don't matter too much in reward calculations
    financial_transactions = ()
    if 'financial' in extra_transactions:
        financial_buy_qty = 0
        financial_sell_qty = 0
        financial_costs = 0
        financial_profit = 0
        # print(extra_transactions['financial'])
        if extra_transactions['financial']['buy']:
            financial_buy_qty = sum([transaction['quantity'] for transaction in extra_transactions['financial']['buy']])
            financial_costs_s = [transaction['quantity'] * transaction['settlement_price_sell'] for transaction in
                                 extra_transactions['financial']['buy']]
            financial_costs_b = [transaction['quantity'] * transaction['settlement_price_buy'] for transaction in
                               extra_transactions['financial']['buy']]
            financial_costs = sum(financial_costs_s) + sum(financial_costs_b)
        if extra_transactions['financial']['sell']:
            financial_sell_qty = sum([transaction['quantity'] for transaction in extra_transactions['financial']['sell']])
            financial_profit_s = [transaction['quantity'] * transaction['settlement_price_sell'] for transaction in
                                extra_transactions['financial']['sell']]
            financial_profit_b = [transaction['quantity'] * transaction['settlement_price_buy'] for transaction in
                                  extra_transactions['financial']['sell']]
            financial_profit = sum(financial_profit_s) + sum(financial_profit_b)
        financial_transactions = (financial_costs, financial_profit, financial_buy_qty, financial_sell_qty)
    return market_transactions, grid_transactions, financial_transactions