"""Catch-up for pre-squash databases: the Auth.js (ProConnect) tables.

qe-front introduced ProConnect SSO with `@auth/drizzle-adapter`, whose four
tables were created by `drizzle/0019_organic_monster_badoon.sql`. The commit
`refactor: Remove drizzle migrations and let backend handle it` deleted the
whole `drizzle/` directory, and for a window nothing owned that DDL: the
adapter's first query failed with `42P01 relation "accounts" does not exist`
and login died on Auth.js's "There is a problem with the server
configuration".

`62ff467c436e` closes that window — it creates the four tables, so any
database built from it is fine and this migration is a no-op there. What it
does not cover is a database provisioned *before* it: its `create_table`
calls are unguarded, which makes it a fresh-install migration, never
replayed on an existing database. That gap is what this fills.

Hence `CREATE TABLE IF NOT EXISTS` throughout — three states must all end up
on a working schema: built from the squash (tables present, no-op),
provisioned before the drizzle removal (tables present from `0019`, no-op),
and provisioned in the window between the two (tables missing, created here).

The shape is dictated by the adapter's Postgres contract, so it is
transcribed verbatim from `qe-front/src/db/schema-auth.ts` rather than
normalised: camelCase identifiers stay quoted, and `verificationTokens`
keeps its plural-camel name. `users.role` ('user' | 'admin') is the one
app-specific addition — it gates /admin/* in qe-front's middleware. Signup
is restricted to *.gouv.fr addresses in the Auth.js `signIn` callback, not
at the schema level.
"""

from collections.abc import Sequence
from typing import Union

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "e7f8a9b0c1d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS "users" (
            "id" text PRIMARY KEY NOT NULL,
            "name" text,
            "email" text NOT NULL,
            "emailVerified" timestamp with time zone,
            "image" text,
            "role" text DEFAULT 'user' NOT NULL,
            CONSTRAINT "users_email_unique" UNIQUE ("email"),
            CONSTRAINT "users_role_check" CHECK ("role" IN ('user', 'admin'))
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS "accounts" (
            "userId" text NOT NULL,
            "type" text NOT NULL,
            "provider" text NOT NULL,
            "providerAccountId" text NOT NULL,
            "refresh_token" text,
            "access_token" text,
            "expires_at" integer,
            "token_type" text,
            "scope" text,
            "id_token" text,
            "session_state" text,
            CONSTRAINT "accounts_provider_providerAccountId_pk"
                PRIMARY KEY ("provider", "providerAccountId"),
            CONSTRAINT "accounts_userId_users_id_fk"
                FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE CASCADE
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS "sessions" (
            "sessionToken" text PRIMARY KEY NOT NULL,
            "userId" text NOT NULL,
            "expires" timestamp with time zone NOT NULL,
            CONSTRAINT "sessions_userId_users_id_fk"
                FOREIGN KEY ("userId") REFERENCES "users" ("id") ON DELETE CASCADE
        );
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS "verificationTokens" (
            "identifier" text NOT NULL,
            "token" text NOT NULL,
            "expires" timestamp with time zone NOT NULL,
            CONSTRAINT "verificationTokens_identifier_token_pk"
                PRIMARY KEY ("identifier", "token")
        );
        """
    )
    # The adapter looks accounts up by user on sign-out and account
    # unlinking; the composite PK is keyed on (provider, providerAccountId)
    # so it cannot serve those. Same for sessions.
    op.execute('CREATE INDEX IF NOT EXISTS "accounts_userId_idx" ON "accounts" ("userId");')
    op.execute('CREATE INDEX IF NOT EXISTS "sessions_userId_idx" ON "sessions" ("userId");')


def downgrade() -> None:
    # Dropping these logs every agent out and discards the role grants,
    # which are set by hand and recorded nowhere else. Ordered so the FK
    # children go first.
    op.execute('DROP TABLE IF EXISTS "verificationTokens";')
    op.execute('DROP TABLE IF EXISTS "sessions";')
    op.execute('DROP TABLE IF EXISTS "accounts";')
    op.execute('DROP TABLE IF EXISTS "users";')
