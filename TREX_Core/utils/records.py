from TREX_Core.utils import utils, db_utils
import sqlalchemy
from sqlalchemy import MetaData, Column
import asyncio
import databases
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
        
        # Track pending database write tasks
        self.__pending_write_tasks = []

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
        """Record the buffered records into the database
        
        Args:
            buf_len: Minimum buffer length to trigger a write (default=0)
            final: If True, wait for the write to complete before returning
            check_table_len: If True, verify record count after write (not implemented)
        
        Returns:
            False if no write was performed (due to buffer conditions)
            True if a write was initiated
        """
        # Apply rate limiting if not a final write
        # if buf_len and not final:
        #     delay = buf_len / 100
        #     ts = datetime.datetime.now().timestamp()
        #     if ts - self.__last_record_time < delay:
        #         return False

        records_len = len(self.__records)
        if records_len < buf_len:
            return False

        # Swap the entire buffer instead of slicing (more efficient)
        records_to_write = self.__records
        self.__records = []  # Create a fresh list for new records
        
        # Create and track the database write task
        db_task = asyncio.create_task(
            db_utils.dump_data(records_to_write, self.__db['path'], self.__db['table'], 
                              existing_connection=self.__db.get('connection'))
        )
        
        # Add to our tracking list
        self.__pending_write_tasks.append(db_task)
        
        # Set up callback to remove from our list when done
        def task_done_callback(completed_task):
            if completed_task in self.__pending_write_tasks:
                self.__pending_write_tasks.remove(completed_task)
        
        db_task.add_done_callback(task_done_callback)
        
        # For critical writes (final=True), wait for completion
        if final:
            await db_task

        self.__last_record_time = datetime.datetime.now().timestamp()
        self.__transactions_count += records_len
        return True

    async def ensure_records_complete(self):
        """Ensure all database write tasks are complete before continuing.
        
        This method will:
        1. Trigger a final write of any pending records
        2. Wait for all pending database write tasks to complete
        3. Verify the record count if needed
        
        Returns:
            True if all records completed successfully
            
        Raises:
            TimeoutError: If writes don't complete within timeout period
        """
        # First do one final write and wait for it to complete
        await self.save(final=True)
        
        # Now wait for ALL remaining in-flight tasks
        if self.__pending_write_tasks:
            # Wait for all pending tasks to complete
            await asyncio.wait(self.__pending_write_tasks)
            
            # Check if we timed out and still have pending tasks
            remaining = [task for task in self.__pending_write_tasks if not task.done()]
            if remaining:
                raise TimeoutError(f"Timed out waiting for {len(remaining)} database writes to complete")
        
        return True

    # Add method to properly close the connection when done
    async def close_connection(self):
        """Close the database connection when done"""
        # First ensure all write tasks are complete
        try:
            await self.ensure_records_complete()
        except Exception as e:
            # Log the error but continue to close the connection
            print(f"Warning: Error ensuring records complete: {e}")
            
        # Now safe to close the connection
        if self.__db.get('connection'):
            await self.__db['connection'].disconnect()
            self.__db['connection'] = None
