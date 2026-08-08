from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_super_admin
from app.core.security import hash_password
from app.db.session import get_db
from app.models.admin import Admin
from app.schemas.admin import AdminCreate, AdminRead, AdminUpdate

router = APIRouter(prefix="/admin/admins", tags=["admin"])


def _to_read(admin: Admin) -> AdminRead:
    return AdminRead(
        id=admin.id,
        email=admin.email,
        is_active=admin.is_active,
        is_super_admin=admin.is_super_admin,
        created_at=admin.created_at,
    )


@router.post("", response_model=AdminRead, status_code=status.HTTP_201_CREATED)
async def create_admin(
    payload: AdminCreate,
    super_admin: Admin = Depends(get_current_super_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminRead:
    existing = await db.scalar(select(Admin).where(Admin.email == payload.email))
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Admin email already registered")

    admin = Admin(
        email=payload.email,
        password_hash=hash_password(payload.password),
        is_active=True,
        is_super_admin=payload.is_super_admin,
    )
    db.add(admin)
    await db.commit()
    await db.refresh(admin)
    return _to_read(admin)


@router.get("", response_model=list[AdminRead])
async def list_admins(
    super_admin: Admin = Depends(get_current_super_admin),
    db: AsyncSession = Depends(get_db),
) -> list[AdminRead]:
    admins = await db.scalars(select(Admin).order_by(Admin.created_at.desc()))
    return [_to_read(admin) for admin in admins]


@router.patch("/{admin_id}", response_model=AdminRead)
async def update_admin(
    admin_id: int,
    payload: AdminUpdate,
    super_admin: Admin = Depends(get_current_super_admin),
    db: AsyncSession = Depends(get_db),
) -> AdminRead:
    if admin_id == super_admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Cannot change your own account")

    admin = await db.get(Admin, admin_id)
    if admin is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Admin not found")

    if payload.is_active is not None:
        admin.is_active = payload.is_active
    if payload.is_super_admin is not None:
        admin.is_super_admin = payload.is_super_admin

    await db.commit()
    await db.refresh(admin)
    return _to_read(admin)
