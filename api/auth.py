from __future__ import annotations

import os
from typing import Optional

import bcrypt
import psycopg2
from fastapi import HTTPException, Request, status

from api.monitoring import PREDICTIONS_DATABASE_URL

SESSION_SECRET_KEY = os.getenv(
    "SESSION_SECRET_KEY",
    "demo-secret-change-me-in-production-aaaaaaaaaaaaaaaaaaaa",
)

SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 365 * 100  # ~100 years


def _fetch_user(username: str) -> Optional[dict]:
    with psycopg2.connect(PREDICTIONS_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, username, password_hash, role FROM users WHERE username = %s",
            (username,),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return {"id": row[0], "username": row[1], "password_hash": row[2], "role": row[3]}


def authenticate(username: str, password: str) -> Optional[dict]:
    user = _fetch_user(username)
    if user is None:
        return None
    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return None
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


def current_user(request: Request) -> Optional[dict]:
    user = request.session.get("user")
    if not user or "username" not in user or "role" not in user:
        return None
    return user


def require_role(*roles: str):
    allowed = tuple(roles)

    def dependency(request: Request) -> dict:
        user = current_user(request)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
            )
        if allowed and user["role"] not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user['role']}' not permitted",
            )
        return user

    return dependency


def home_path_for_role(role: str) -> str:
    if role == "reviewer":
        return "/review.html"
    return "/index.html"
