from datetime import datetime

from pydantic import BaseModel, EmailStr


class AdminCreate(BaseModel):
    email: EmailStr
    password: str
    is_super_admin: bool = False


class AdminUpdate(BaseModel):
    is_active: bool | None = None
    is_super_admin: bool | None = None


class AdminRead(BaseModel):
    id: int
    email: EmailStr
    is_active: bool
    is_super_admin: bool
    created_at: datetime
