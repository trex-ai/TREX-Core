
from setuptools import setup, find_packages

#ToDO, make nicer, for example by importing requirements or sth
setup(
    name='TREX_Core',
    version='4.0.0',
    install_requires=[  "asyncpg",
                        "commentjson",
                        "asyncpg",
                        "cuid2",
                        "databases",
                        "databases[postgresql]",
                        "dataset",
                        "gmqtt",
                        "packaging",
                        "psycopg2-binary",
                        "python-dateutil",
                        "pytz",
                        "python-rapidjson",
                        "sqlalchemy",
                        "sqlalchemy-utils",
                        "tenacity",
                        "websockets"
              ],
    packages=find_packages()
    )
