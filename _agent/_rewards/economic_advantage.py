from _agent._rewards.utils import process_ledger

class Reward:
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