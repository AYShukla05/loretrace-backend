"""Create the first super admin from SUPER_ADMIN_EMAIL/SUPER_ADMIN_PASSWORD in .env.

Idempotent: if an admin with that email already exists, does nothing and
leaves the existing row untouched. Run with:

    python scripts/seed_super_admin.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.core.config import settings
from app.core.security import hash_password
from app.db.session import async_session
from app.models.admin import Admin


async def main() -> None:
    if not settings.super_admin_email or not settings.super_admin_password:
        print("SUPER_ADMIN_EMAIL and SUPER_ADMIN_PASSWORD must be set in .env")
        return

    async with async_session() as db:
        existing = await db.scalar(select(Admin).where(Admin.email == settings.super_admin_email))
        if existing is not None:
            print(f"Admin with email {settings.super_admin_email} already exists, skipping")
            return

        admin = Admin(
            email=settings.super_admin_email,
            password_hash=hash_password(settings.super_admin_password),
            is_active=True,
            is_super_admin=True,
        )
        db.add(admin)
        await db.commit()
        print(f"Created super admin {settings.super_admin_email}")


if __name__ == "__main__":
    asyncio.run(main())
