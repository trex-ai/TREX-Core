from _agent.rewards.utils import process_ledger

class Reward:
    # Calculates unit profit and cost
    # needed for sma_crossover_trader
    def __init__(self, timing=None, ledger=None, market_info=None, **kwargs):
        self.__timing = timing
        self.__ledger = ledger
        self.__market_info = market_info

    async def calculate(self, last_deliver=None, market_transactions=None, grid_transactions=None,
                        financial_transactions=None):
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

        if (market_transactions == None and grid_transactions == None and financial_transactions == None):
            market_transactions, grid_transactions, financial_transactions = \
                await process_ledger(last_deliver, self.__ledger, self.__market_info)

        asks_qty = sum([t[1] for t in market_transactions if t[0] == 'ask'])
        bids_qty = sum([t[1] for t in market_transactions if t[0] == 'bid'])
        market_profit = sum([t[1] * t[2] for t in market_transactions if t[0] == 'ask'])
        market_cost = sum([t[1] * t[2] for t in market_transactions if t[0] == 'bid'])

        grid_sell_price = grid_transactions[3]
        grid_buy_price = grid_transactions[1]

        unit_profit = market_profit / asks_qty if asks_qty > 0 else grid_sell_price
        unit_cost = market_cost / bids_qty if bids_qty > 0 else grid_buy_price

        unit_profit_diff = unit_profit - grid_sell_price
        unit_cost_diff = unit_cost - grid_buy_price

        return [unit_profit, unit_profit_diff, unit_cost, unit_cost_diff]