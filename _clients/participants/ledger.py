class Ledger:
    """Ledger helps participants keep track of accepted bids/asks, and successsful settlements.

    Contains functions that will process raw ledger data into formats more useful downstream in the data pipeline
    """
    
    def __init__(self, participant_id, **kwargs):
        self.__participant_id = participant_id
        self.bids = {}
        self.asks = {}
        self.settled = {}
        self.extra = {}

    async def bid_success(self, confirmation):
        """Track bid that was accepted by the Market

        Args:
            confirmation ([type]): [description]
        """

        time_delivery = tuple(confirmation['time_delivery'])
        if time_delivery not in self.bids:
            self.bids[time_delivery] = {}

        self.bids[time_delivery][confirmation['uuid']] = {
            'time_submission': confirmation['time_submission'],
            'price': confirmation['price'],
            'quantity': confirmation['quantity']
        }

    async def ask_success(self, confirmation):
        """Track ask that was accepted by the Market

        Args:
            confirmation ([type]): [description]
        """

        time_delivery = tuple(confirmation['time_delivery'])
        if time_delivery not in self.asks:
            self.asks[time_delivery] = {}

        self.asks[time_delivery][confirmation['uuid']] = {
            'time_submission': confirmation['time_submission'],
            'source': confirmation['source'],
            'price': confirmation['price'],
            'quantity': confirmation['quantity']
        }

    async def settle_success(self, confirmation):
        """Track successful settlement

        Args:
            confirmation ([type]): [description]
        """
        # todo: add validity checks, and feedback messages for invalid settlements
        time_delivery = tuple(confirmation['time_delivery'])
        if confirmation['buyer_id'] == self.__participant_id:
            # make sure settled bid exists in local record as well
            if confirmation['bid_id'] in self.bids[time_delivery]:
                if time_delivery not in self.settled:
                    self.settled[time_delivery] = {'bids': {},
                                                     'asks': {}}
                self.settled[time_delivery]['bids'][confirmation['commit_id']] = {
                    'source': confirmation['source'],
                    'bid_price': self.bids[time_delivery][confirmation['bid_id']]['price'],
                    'price': confirmation['price'],
                    'quantity': confirmation['quantity']
                }
                # update local bid entry
                self.bids[time_delivery][confirmation['bid_id']]['quantity'] -= confirmation['quantity']
                if self.bids[time_delivery][confirmation['bid_id']]['quantity'] <= 0:
                    self.bids[time_delivery].pop(confirmation['bid_id'])

        elif confirmation['seller_id'] == self.__participant_id:
            if confirmation['ask_id'] in self.asks[time_delivery]:
                if time_delivery not in self.settled:
                    self.settled[time_delivery] = {'bids': {},
                                                     'asks': {}}

                self.settled[time_delivery]['asks'][confirmation['commit_id']] = {
                    'source': confirmation['source'],
                    'ask_price': self.asks[time_delivery][confirmation['ask_id']]['price'],
                    'price': confirmation['price'],
                    'quantity': confirmation['quantity']
                }
                # update local ask entry
                self.asks[time_delivery][confirmation['ask_id']]['quantity'] -= confirmation['quantity']
                if self.asks[time_delivery][confirmation['ask_id']]['quantity'] <= 0:
                    self.asks[time_delivery].pop(confirmation['ask_id'])

    def get_settled_info(self, time_interval, **kwargs):
        """Summarizes ledger data for a certain time interval

        Args:
            time_interval (tuple): This time interval is not arbitrary and must be one of the market rounds that occurred in the past

        Returns:
            [type]: [description]
        """
        info = {
            'asks': {
                'quantity': 0,
                'total_profit': 0
            },
            'bids': {
                'quantity': 0,
                'total_cost': 0
            }
        }

        if time_interval not in self.settled:
            return info

        settlements = self.settled[time_interval]
        if 'asks' in settlements:
            for commit_id in settlements['asks']:
                info['asks']['quantity'] += settlements['asks'][commit_id]['quantity']
                info['asks']['total_profit'] += settlements['asks'][commit_id]['quantity'] * settlements['asks'][commit_id]['price']
        if 'bids' in settlements:
            for commit_id in settlements['bids']:
                info['bids']['quantity'] += settlements['bids'][commit_id]['quantity']
                info['bids']['total_cost'] += settlements['bids'][commit_id]['quantity'] * settlements['bids'][commit_id]['price']
        return info

    async def get_simplified_transactions(self, time_interval):
        """returns a list of transactions for a time interval
        

        Args:
            time_interval (tuple): This time interval is not arbitrary and must be one of the market rounds that occurred in the past

        Returns:
            [type]: [description]
        """
        transactions = []
        if time_interval not in self.settled:
            return transactions
        settlements = self.settled[time_interval]
        for action in ('bids', 'asks'):
            if action in settlements:
                for commit_id in settlements[action]:
                    source = settlements[action][commit_id]['source']
                    quantity = settlements[action][commit_id]['quantity']
                    price = settlements[action][commit_id]['price']

                    # note how action is converted from plural to singular (asks -> ask, bids -> bid).
                    # This only works because bid/ask have the same character length
                    transactions.append((action[:3], quantity, price, source))
        return transactions
    
    async def clear_history(self, time_interval):
        self.bids.pop(time_interval, None)
        self.asks.pop(time_interval, None)
        self.settled.pop(time_interval, None)
        self.extra.pop(time_interval, None)

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self.settled.clear()
        self.extra.clear()
