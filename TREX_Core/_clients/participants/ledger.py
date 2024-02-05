class Ledger:
    """Ledger helps participants keep track of accepted bids/asks, and successsful settlements.

    Contains functions that will process raw ledger data into formats more useful downstream in the data pipeline
    """
    
    def __init__(self, participant_id, **kwargs):
        self.__participant_id = participant_id
        self.bids_hold = {}
        self.asks_hold = {}
        self.bids = {}
        self.asks = {}
        self.settled = {}
        self.extra = {}

    async def bid_success(self, entry_id):
        """Track bid that was accepted by the Market

        Args:
            confirmation ([type]): [description]
        """

        entry = self.bids_hold.pop(entry_id, None)
        if not entry:
            return
        time_delivery = entry.pop('time_delivery')
        if time_delivery not in self.bids:
            self.bids[time_delivery] = {}
        self.bids[time_delivery][entry_id] = entry

    async def ask_success(self, entry_id):
        """Track ask that was accepted by the Market

        Args:
            confirmation ([type]): [description]
        """

        entry = self.asks_hold.pop(entry_id, None)
        if not entry:
            return
        time_delivery = entry.pop('time_delivery')
        if time_delivery not in self.asks:
            self.asks[time_delivery] = {}
        self.asks[time_delivery][entry_id] = entry

    async def settle_success(self, confirmation):
        """Track successful settlement

        Args:
            confirmation ([type]): [description]
        """
        # print(confirmation, self.bids, self.asks)
        # todo: add validity checks, and feedback messages for invalid settlements
        # print(confirmation)

        commit_id = confirmation[0]
        entry_id = confirmation[1]
        source = confirmation[2]
        quantity = confirmation[3]
        time_delivery = tuple(confirmation[4])

        if time_delivery not in self.settled:
            self.settled[time_delivery] = {'bids': {}, 'asks': {}}
        # if 'buyer_id' in confirmation and confirmation['buyer_id'] == self.__participant_id:
            # make sure settled bid exists in local record as well
        entry_list = []
        if time_delivery in self.bids and entry_id in self.bids[time_delivery]:
            entry_list = ['bids', self.bids]
        elif time_delivery in self.asks and entry_id in self.asks[time_delivery]:
            entry_list = ['asks', self.asks]
        else:
            print(confirmation)

        self.settled[time_delivery][entry_list[0]][commit_id] = {
            'source': source,
            'price': entry_list[1][time_delivery][entry_id]['price'],
            'quantity': quantity
        }
        # update local bid entry
        entry_list[1][time_delivery][entry_id]['quantity'] -= quantity
        if entry_list[1][time_delivery][entry_id]['quantity'] <= 0:
            entry_list[1][time_delivery].pop(entry_id)

        # elif 'seller_id' in confirmation and confirmation['seller_id'] == self.__participant_id:
        #     print(confirmation)
        # elif time_delivery in self.asks and confirmation['id'] in self.asks[time_delivery]:
        #     self.settled[time_delivery]['asks'][confirmation['id']] = {
        #         'source': confirmation['source'],
        #         'price': self.asks[time_delivery][confirmation['id']]['price'],
        #         # 'price': confirmation['price'],
        #         'quantity': confirmation['quantity']
        #     }
        #     # update local ask entry
        #     self.asks[time_delivery][confirmation['id']]['quantity'] -= confirmation['quantity']
        #     if self.asks[time_delivery][confirmation['id']]['quantity'] <= 0:
        #         self.asks[time_delivery].pop(confirmation['id'])
        return commit_id

    async def get_settled_info(self, time_interval, **kwargs):
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
        self.bids_hold.clear()
        self.asks_hold.clear()
        self.bids.pop(time_interval, None)
        self.asks.pop(time_interval, None)
        self.settled.pop(time_interval, None)
        self.extra.pop(time_interval, None)

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self.settled.clear()
        self.extra.clear()
