from TREX_Core._utils import utils, db_utils
import sqlalchemy
from sqlalchemy import MetaData, Column
import asyncio
import databases
import tenacity

class Metrics:
    def __init__(self, agent_id, track):
        self.__agent_id = agent_id
        self.__track = track
        self.__db = {}
        self.__metrics = {}
        self.__metrics_meta = {}
        # self.__transactions_count = 0

    def add(self, metric_name:str, column_type):
        if metric_name not in self.__metrics:
            self.__metrics[metric_name] = []
            self.__metrics_meta[metric_name] = column_type

    async def track(self, metric_name:str, value):
        if not self.__track:
            return

        if metric_name not in self.__metrics:
            raise Exception('metric not defined')
        self.__metrics[metric_name].append(value)

    def __create_metrics_table(self, table_name):
        columns = [Column(metric, self.__metrics_meta[metric]) for metric in self.__metrics_meta]
        table = sqlalchemy.Table(
            table_name,
            MetaData(),
            *columns
        )
        return table

    def update_db_info(self, db_string, table_name):
        self.__db['path'] = db_string
        self.__db['table_name'] = table_name + '_' + self.__agent_id

    def reset(self):
        self.__db.clear()
        self.__transactions_count = 0
        for metric_name in self.__metrics:
            self.__metrics[metric_name].clear()

    async def save(self, buf_len=0):
        if not self.__track:
            return
        # code converts dictionaries of lists to list of dictionaries
        # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
        # v = [dict(zip(DL, t)) for t in zip(*DL.values())]

        metrics_len = len(self.__metrics['timestamp'])
        if metrics_len > buf_len:
            # a table will be created the first time metrics are being saved
            # this increases the likelihood of complete columnss
            if 'table' not in self.__db or self.__db['table'] is None:
                table_name = self.__db.pop('table_name')
                await db_utils.create_table(db_string=self.__db['path'],
                                            table_type='custom',
                                            custom_table=self.__create_metrics_table(table_name))
                self.__db['table'] = db_utils.get_table(self.__db['path'], table_name)

            if self.__db['table'] is None:
                return

            metrics = {key: value for key, value in self.__metrics.items() if value}
            metrics = [dict(zip(metrics, t)) for t in zip(*metrics.values())]
            self.__metrics = {key: value[metrics_len:] for key, value in self.__metrics.items()}
            # await db_utils.dump_data(metrics, self.__db['path'], self.__db['table'])
            await asyncio.create_task(db_utils.dump_data(metrics, self.__db['path'], self.__db['table']))
            await self.__ensure_transactions_complete(metrics_len)

    @tenacity.retry(wait=tenacity.wait_random(1, 5))
    async def __ensure_transactions_complete(self, metrics_len):
        table_len = db_utils.get_table_len(self.__db['path'], self.__db['table'])
        # numbers don't match all the time for some reason
        print('comparing recorded metrics vs number of metrics', table_len, metrics_len)
        if table_len < metrics_len:
            raise Exception
        return True
            
    async def fetch_one(self, timestamp):
        if 'db' not in self.__db:
            self.__db['db'] = databases.Database(self.__db['path'])

        if 'table' not in self.__db or self.__db['table'] is None:
            table_name = self.__db.pop('table_name')
            self.__db['table'] = db_utils.get_table(self.__db['path'], table_name)
            await self.__db['db'].connect()

        table = self.__db['table']
        query = table.select().where(table.c.timestamp == timestamp)
        async with self.__db['db'].transaction():
            row = await self.__db['db'].fetch_one(query)
        return row['actions_dict']
