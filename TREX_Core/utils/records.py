from TREX_Core.utils import utils, db_utils
import sqlalchemy
from sqlalchemy import MetaData, Column
import asyncio
import databases
import tenacity
import datetime
import ast
from pprint import pprint

class Records:
    def __init__(self, db_string, columns):
        self.__db = {
            'path': db_string,
            'sa_engine': sqlalchemy.create_engine(db_string),
            'connection': None  # Added to store a reusable database connection
        }
        # self.__columns = columns
        self.__columns = columns

        # pprint(self.__columns)

        self.__records = list()
        self.__last_record_time = 0
        self.__transactions_count = 0
        self.__meta = MetaData()

    async def create_table(self, table_name):
        table_name += '_records'
        columns = [Column(record,
                          getattr(sqlalchemy, self.__columns[record]['type']),
                          primary_key=self.__columns[record]['primary'] if 'primary' in self.__columns[record] else False)
                   for record in self.__columns]
        table = sqlalchemy.Table(
            str(table_name),
            self.__meta,
            *columns
        )
        return await db_utils.create_table(self.__db['path'], table)
        # return table

    async def open_db(self, table_name, db_string=None):
        if not db_string:
            db_string = self.__db['path']
        table_name += '_records'
        self.__db['table'] = db_utils.get_table(db_string, table_name)
        # Initialize the database connection
        if not self.__db.get('connection'):
            self.__db['connection'] = databases.Database(db_string)
            await self.__db['connection'].connect()

    # def add(self, record: str, column_type):
    #     if record not in self.__columns:
    #         # self.__records[record] = []
    #         self.__columns[record] = column

    async def track(self, records):
        # if not self.__track:
        #     return
        filtered_records = {key: records[key] for key in self.__columns.keys() if key in records}
        self.__records.append(filtered_records)

    # def update_db_info(self, db_string, table_name):
    #     self.__db['path'] = db_string
    #     self.__db['table_name'] = table_name
    #     # self.__db['table_name'] = table_name + '_' + self.__agent_id

    # def reset(self):
    #     self.__db.clear()
    #     self.__transactions_count = 0
    #     for metric_name in self.__records:
    #         self.__records[metric_name].clear()

    async def save(self, buf_len=0, final=False, check_table_len=False):
        """This function records the transaction records into the ledger

        """

        # if check_table_len:
        #     table_len = db_utils.get_table_len(self.__db['path'], self.__db['table'])
        #     if table_len < self.transactions_count:
        #         return False

        if buf_len:
            delay = buf_len / 100
            ts = datetime.datetime.now().timestamp()
            if ts - self.__last_record_time < delay:
                return False

        records_len = len(self.__records)
        if records_len < buf_len:
            return False

        records = self.__records[:records_len]
        if not final:
            await asyncio.create_task(db_utils.dump_data(records, self.__db['path'], self.__db['table'], existing_connection=self.__db.get('connection')))
        else:
            await db_utils.dump_data(records, self.__db['path'], self.__db['table'], existing_connection=self.__db.get('connection'))
        self.__last_record_time = datetime.datetime.now().timestamp()
        del self.__records[:records_len]
        # self.transactions_count += transactions_len
        return True


    # async def save(self, buf_len=0):
    #     if not self.__track:
    #         return
    #     # code converts dictionaries of lists to list of dictionaries
    #     # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    #     # v = [dict(zip(DL, t)) for t in zip(*DL.values())]
    #
    #     # metrics_len = len(self.__records['timestamp'])
    #     if metrics_len > buf_len:
    #         # a table will be created the first time metrics are being saved
    #         # this increases the likelihood of complete columnss
    #         if 'table' not in self.__db or self.__db['table'] is None:
    #             table_name = self.__db.pop('table_name')
    #             # await db_utils.create_table(db_string=self.__db['path'],
    #             #                             table=self.__create_metrics_table(table_name))
    #             # try:
    #             self.__db['table'] = db_utils.get_table(self.__db['path'], table_name)
    #             if self.__db['table'] is None:
    #                 await db_utils.create_table(db_string=self.__db['path'],
    #                                             table=self.__create_metrics_table(table_name))
    #                 self.__db['table'] = db_utils.get_table(self.__db['path'], table_name)
    #
    #         if self.__db['table'] is None:
    #             return
    #
    #         metrics = {key: value for key, value in self.__records.items() if value}
    #         metrics = [dict(zip(metrics, t)) for t in zip(*metrics.values())]
    #         self.__records = {key: value[metrics_len:] for key, value in self.__records.items()}
    #         # await db_utils.dump_data(metrics, self.__db['path'], self.__db['table'])
    #         await asyncio.create_task(db_utils.dump_data(metrics, self.__db['path'], self.__db['table']))
    #         # await self.__ensure_transactions_complete(metrics_len)

    # @tenacity.retry(wait=tenacity.wait_random(1, 5))
    # async def __ensure_transactions_complete(self, metrics_len):
    #     table_len = db_utils.get_table_len(self.__db['path'], self.__db['table'])
    #     # numbers don't match all the time for some reason
    #     print('comparing recorded metrics vs number of metrics', table_len, metrics_len)
    #     if table_len < metrics_len:
    #         raise Exception
    #     return True
    #
    # async def fetch_one(self, timestamp):
    #     if 'db' not in self.__db:
    #         self.__db['db'] = databases.Database(self.__db['path'])
    #
    #     if 'table' not in self.__db or self.__db['table'] is None:
    #         table_name = self.__db.pop('table_name')
    #         self.__db['table'] = db_utils.get_table(self.__db['path'], table_name)
    #         await self.__db['db'].connect()
    #
    #     table = self.__db['table']
    #     query = table.select().where(table.c.timestamp == timestamp)
    #     async with self.__db['db'].transaction():
    #         row = await self.__db['db'].fetch_one(query)
    #     return row['actions_dict']

    # Add method to properly close the connection when done
    async def close_connection(self):
        """Close the database connection when done"""
        if self.__db.get('connection'):
            await self.__db['connection'].disconnect()
            self.__db['connection'] = None
