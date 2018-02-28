"""add last_seen column to scanners

Revision ID: 7c8828cac5fd
Revises: da56e259f2e6
Create Date: 2018-02-26 20:36:47.974583

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7c8828cac5fd'
down_revision = 'da56e259f2e6'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'scanners',
        sa.Column('last_seen', sa.DateTime(timezone=True)),
    )


def downgrade():
    pass
