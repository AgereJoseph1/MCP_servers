from core.authentication.auth_middleware import RoleBasedAccessControl

allow_resource_admin = RoleBasedAccessControl(["admin", "super_admin"])
