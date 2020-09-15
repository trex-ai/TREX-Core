from _agent._rewards.utils import process_ledger

class Reward:
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
