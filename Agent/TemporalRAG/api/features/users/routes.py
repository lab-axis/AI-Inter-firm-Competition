from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from models.database import get_db, User
from api.auth.jwt_handler import get_current_active_user
from api.auth.schemas import UserResponse

router = APIRouter(
    prefix="/users",
    tags=["Users"],
    dependencies=[Depends(get_current_active_user)],
)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_active_user)):
    return current_user
