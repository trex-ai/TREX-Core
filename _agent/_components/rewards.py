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
        financial_costs = 0
        financial_profit = 0
        if extra_transactions['financial']['buy']:
            financial_costs = [transaction['quantity'] * transaction['settlement_price'] for transaction in
                               extra_transactions['financial']['buy']]
            financial_costs = sum(financial_costs)
        if extra_transactions['financial']['sell']:
            financial_profit = [transaction['quantity'] * transaction['settlement_price'] for transaction in
                                extra_transactions['financial']['sell']]
            financial_profit = sum(financial_profit)
        financial_transactions = (financial_costs, financial_profit)
    return market_transactions, grid_transactions, financial_transactions

class EconomicAdvantage:
    def __init__(self, timing=None, ledger=None, market_info=None, **kwargs):
        self.__timing = timing
        self.__ledger = ledger
        self.__market_info = market_info
        # self.last = {}

    async def calculate(self, last_deliver = None, market_transactions=None, grid_transactions=None, financial_transactions=None):
        """
        Parameters:
            dict : settlements
            dict : grid_transactions
        """
        if not last_deliver:
            if 'last_deliver' not in self.__timing:
                return None
            else:
                last_deliver = self.__timing['last_deliver']

        if (market_transactions==None and grid_transactions==None and financial_transactions==None):
            market_transactions, grid_transactions, financial_transactions = \
                await process_ledger(last_deliver, self.__ledger, self.__market_info)

        asks_qty = sum([t[1] for t in market_transactions if t[0] == 'ask'])
        bids_qty = sum([t[1] for t in market_transactions if t[0] == 'bid'])
        market_profit = sum([t[1] * t[2] for t in market_transactions if t[0] == 'ask'])
        market_cost = sum([t[1] * t[2] for t in market_transactions if t[0] == 'bid'])

        grid_sell_price = grid_transactions[3]
        grid_buy_price = grid_transactions[1]

        nme_profit = grid_sell_price * asks_qty
        market_advantage_profit = market_profit - nme_profit

        nme_cost = grid_buy_price * bids_qty
        market_advantage_cost = market_cost - nme_cost

        financial_costs = financial_transactions[0] if financial_transactions else 0
        financial_profit = financial_transactions[1] if financial_transactions else 0

        profit_diff =  market_advantage_profit + financial_profit
        cost_diff =  market_advantage_cost + financial_costs
        reward = float(profit_diff - cost_diff)/1000 #divide by 1000 because units so far are $/kWh * wh
        return reward

class NetProfit:
    def __init__(self, timing=None, ledger=None, market_info=None, **kwargs):
        self.__timing = timing
        self.__ledger = ledger
        self.__market_info = market_info

    async def calculate(self, last_deliver=None, market_transactions=None, grid_transactions=None, financial_transactions=None):
        """
        Parameters:
            dict : settlements
            dict : grid_transactions
        """
        if not last_deliver:
            if 'last_deliver' not in self.__timing:
                return None
            else:
                last_deliver = self.__timing['last_deliver']

        if (market_transactions==None and grid_transactions==None and financial_transactions==None):
            market_transactions, grid_transactions, financial_transactions = \
                await process_ledger(last_deliver, self.__ledger, self.__market_info)

        market_cost = sum([t[1] * t[2] for t in market_transactions if t[0] == 'bid'])
        market_profit = sum([t[1] * t[2] for t in market_transactions if t[0] == 'ask'])

        grid_cost = grid_transactions[0] * grid_transactions[1]
        grid_profit = grid_transactions[2] * grid_transactions[3]

        financial_cost = financial_transactions[0] if financial_transactions else 0
        financial_profit = financial_transactions[1] if financial_transactions else 0

        total_profit = market_profit + grid_profit + financial_profit
        total_cost = market_cost + grid_cost + financial_cost
        reward = float(total_profit - total_cost)/1000
        return reward

class UnitProfitAndCost:
    # Calculates unit profit and cost
    # needed for sma_crossover_trader
    def __init__(self, timing=None, ledger=None, market_info=None, **kwargs):
        self.__timing = timing
        self.__ledger = ledger
        self.__market_info = market_info

    async def calculate(self, last_deliver = None, market_transactions=None, grid_transactions=None, financial_transactions=None):
        """
        Parameters:
            dict : settlements
            dict : grid_transactions
        """
        if not last_deliver:
            if 'last_deliver' not in self.__timing:
                return None
            else:
                last_deliver = self.__timing['last_deliver']

        if (market_transactions==None and grid_transactions==None and financial_transactions==None):
            market_transactions, grid_transactions, financial_transactions = \
                await process_ledger(last_deliver, self.__ledger, self.__market_info)

        asks_qty = sum([t[1] for t in market_transactions if t[0] == 'ask'])
        bids_qty = sum([t[1] for t in market_transactions if t[0] == 'bid'])
        market_profit = sum([t[1] * t[2] for t in market_transactions if t[0] == 'ask'])
        market_cost = sum([t[1] * t[2] for t in market_transactions if t[0] == 'bid'])

        grid_sell_price = grid_transactions[3]
        grid_buy_price = grid_transactions[1]

        unit_profit =  market_profit/asks_qty if asks_qty > 0 else grid_sell_price
        unit_cost = market_cost/bids_qty if bids_qty > 0 else grid_buy_price

        unit_profit_diff =  unit_profit - grid_sell_price
        unit_cost_diff =  unit_cost - grid_buy_price

        return [unit_profit, unit_profit_diff, unit_cost, unit_cost_diff]