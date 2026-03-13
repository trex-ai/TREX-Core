async def process_ledger(last_deliver, ledger, market_info):
    # strip out time from transactions and simplify for reward calculation
    # print(last_deliver, market_info)
    grid_prices = market_info[last_deliver]["grid"]

    # market transactions use a simplified format.
    # Each source maps to a list of ('bid/ask', quantity, price) entries.
    market_transactions = await ledger.get_simplified_transactions(last_deliver)
    extra_transactions = ledger.extra.get(last_deliver, {})

    # grid params
    # grid information is special. It is a tuple of the following format:
    # buy quantity, buy price, sell quantity, sell price
    grid_transactions: tuple[()] | tuple[float, float, float, float] = ()
    if "grid" in extra_transactions:
        if extra_transactions["grid"]["buy"]:
            grid_buy_qty = sum(
                transaction["quantity"]
                for transaction in extra_transactions["grid"]["buy"]
            )
        else:
            grid_buy_qty = 0

        if extra_transactions["grid"]["sell"]:
            grid_sell_qty = sum(
                transaction["quantity"]
                for transaction in extra_transactions["grid"]["sell"]
            )
        else:
            grid_sell_qty = 0

        grid_transactions = (
            grid_buy_qty,
            grid_prices["buy_price"],
            grid_sell_qty,
            grid_prices["sell_price"],
        )

    # Financial transactions only track aggregate cost and profit.
    # The individual transaction details are not needed for this reward.
    financial_transactions: tuple[()] | tuple[float, float, float, float] = ()
    if "financial" in extra_transactions:
        financial_buy_qty = 0
        financial_sell_qty = 0
        financial_costs = 0.0
        financial_profit = 0.0
        if extra_transactions["financial"]["buy"]:
            financial_buy_qty = sum(
                transaction["quantity"]
                for transaction in extra_transactions["financial"]["buy"]
            )
            financial_costs = sum(
                transaction["quantity"] * transaction["settlement_price_buy"]
                for transaction in extra_transactions["financial"]["buy"]
            )
        if extra_transactions["financial"]["sell"]:
            financial_sell_qty = sum(
                transaction["quantity"]
                for transaction in extra_transactions["financial"]["sell"]
            )
            financial_profit = sum(
                transaction["quantity"] * transaction["settlement_price_sell"]
                for transaction in extra_transactions["financial"]["sell"]
            )
        financial_transactions = (
            financial_costs,
            financial_profit,
            financial_buy_qty,
            financial_sell_qty,
        )
    return market_transactions, grid_transactions, financial_transactions
