"""Create scanners table

Revision ID: 6736f58587af
Revises:
Create Date: 2018-02-21 10:12:27.068506

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6736f58587af'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'scanners',
        sa.Column('id', sa.Text, primary_key=True),
        sa.Column('name', sa.Text, unique=True),
    )


def downgrade():
    op.drop_table('scanners')
