import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.admin import Admin
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
# auto_error=False: /chat stays usable with no login at all (see
# get_current_user_optional below); a hard-401ing scheme would break that.
optional_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login", auto_error=False)


async def get_current_admin(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Admin:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
    except jwt.InvalidTokenError:
        raise credentials_error from None

    email = payload.get("sub")
    if email is None:
        raise credentials_error

    admin = await db.scalar(select(Admin).where(Admin.email == email))
    if admin is None or not admin.is_active:
        raise credentials_error

    return admin


async def get_current_super_admin(admin: Admin = Depends(get_current_admin)) -> Admin:
    if not admin.is_super_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin privileges required",
        )
    return admin


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)
    except jwt.InvalidTokenError:
        raise credentials_error from None

    email = payload.get("sub")
    if email is None:
        raise credentials_error

    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active:
        raise credentials_error

    return user


# Chat stays anonymous-first per the project's own account-scoping decision:
# a missing or invalid token means "not logged in", never a 401. Only a
# valid, active user's token unlocks chat-history persistence.
async def get_current_user_optional(
    token: str | None = Depends(optional_oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    if token is None:
        return None

    try:
        payload = decode_access_token(token)
    except jwt.InvalidTokenError:
        return None

    email = payload.get("sub")
    if email is None:
        return None

    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active:
        return None

    return user
