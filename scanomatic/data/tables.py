from sqlalchemy import Table, MetaData, Column, Text

metadata = MetaData()


scanners = Table(
    'scanners', metadata,
    Column('id', Text(), primary_key=True),
    Column('name', Text(), unique=True),
)
