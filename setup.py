
from setuptools import setup, find_packages

#ToDO, make nicer, for example by importing requirements or sth
setup(
    name='TREX_Core',
    version='0.0.1',
    install_requires=[  "aiohttp",
                        "aiosqlite",
                        "asyncpg",
                        "commentjson",
                        "cuid",
                        "databases",
                        "dataset",
                        "numpy",
                        "packaging",
                        "psycopg2-binary",
                        "python-dateutil",
                         "python-socketio<=4.6.1",
                        "pytz",
                        "python-rapidjson",
                        "sqlalchemy-utils",
                        "tenacity",
                        "tensorflow>=2.0",
                        "websockets",
              ],
    packages=find_packages()
    )

