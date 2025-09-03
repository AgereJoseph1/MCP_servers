from typing import List

from core.authentication.auth_token import verify_access_token
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from schemas.token import TokenData

security = HTTPBearer()


def get_current_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> TokenData:
    """
    This function extracts the token from the Authorization header.
    You can forward this token to the external authentication service for validation.
    """
    token = credentials.credentials

    token_data = verify_access_token(token)

    if token_data.type != "bearer":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token type",
        )

    return token_data



class RoleBasedAccessControl:
    """Defines role based access control"""

    def __init__(self, roles: List[str]) -> None:
        self.allowed_roles = roles

    def __call__(
        self, current_token: TokenData = Depends(get_current_token)
    ) -> None:
        if current_token.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User role not permitted to perform this action",
            )