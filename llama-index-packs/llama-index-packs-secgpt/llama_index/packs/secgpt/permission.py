"""
SecGPT implements a permission system for app invocation and collaboration as well as data sharing. To enable the permission system, we define several helper functions here. SecGPT maintains a JSON file to store the information of user-granted permission information, which is stored at permissions.json by default.
"""

import json


class PermissionType:
    """
    A class to define different types of permissions.
    """

    ONE_TIME = "one_time"
    SESSION = "session"
    PERMANENT = "permanent"


def read_permissions_from_file(perm_path="./permissions.json"):
    """
    Read the permissions from a JSON file.

    Args:
        perm_path (str): The path to the permissions JSON file.

    Returns:
        dict: A dictionary containing the permissions.
    """
    try:
        with open(perm_path) as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def write_permissions_to_file(permissions, perm_path="./permissions.json"):
    """
    Write the permissions to a JSON file.

    Args:
        permissions (dict): A dictionary containing the permissions.
        perm_path (str): The path to the permissions JSON file.
    """
    with open(perm_path, "w") as file:
        json.dump(permissions, file, indent=4)


def clear_temp_permissions():
    """
    Clear temporary permissions (one-time and session) from the permissions file.
    """
    permissions = read_permissions_from_file()
    for user_id in permissions:
        for app in list(permissions[user_id]):
            for perm_category in list(permissions[user_id][app]):
                if not permissions[user_id][app][perm_category]:
                    del permissions[user_id][app][perm_category]
                elif permissions[user_id][app][perm_category] in PermissionType.SESSION:
                    del permissions[user_id][app][perm_category]
    write_permissions_to_file(permissions)


def set_permission(user_id, app, permission_type, perm_category):
    """
    Set a specific permission for a user and app.

    Args:
        user_id (str): The ID of the user.
        app (str): The name of the app.
        permission_type (str): The type of permission (one-time, session, permanent).
        perm_category (str): The category of the permission (exec, data, collab).
    """
    permissions = read_permissions_from_file()
    permissions[user_id] = permissions.get(user_id, {})
    permissions[user_id][app] = permissions[user_id].get(app, {})
    permissions[user_id][app][perm_category] = permission_type
    write_permissions_to_file(permissions)


def get_permission(user_id, app, perm_category):
    """
    Get the permission type for a specific user, app, and permission category.

    Args:
        user_id (str): The ID of the user.
        app (str): The name of the app.
        perm_category (str): The category of the permission (exec, data, collab).

    Returns:
        str: The type of permission if exists, otherwise None.
    """
    permissions = read_permissions_from_file()
    app_permissions = permissions.get(user_id, {}).get(app)
    if app_permissions:
        return app_permissions.get(perm_category)
    return None


def request_permission(user_id, app, action, perm_category, flag):
    """
    Request permission from the user for a specific action.

    Args:
        user_id (str): The ID of the user.
        app (str): The name of the app.
        action (str): The action description for which permission is requested.
        perm_category (str): The category of the permission (exec, data, collab).
        flag (bool): A flag indicating additional information about the request.

    Returns:
        bool: True if permission is granted, otherwise False.
    """
    if perm_category == "exec":
        action_type = "execute"
    elif perm_category == "data":
        action_type = "access data"
    elif perm_category == "collab":
        action_type = "share data"
    print("\n=====================================")
    print(f"Allow {app} to {action_type}")

    if not flag:
        if perm_category == "exec":
            print(
                f"\nWarning: {app} is not expected to be used and may pose security or privacy risks if being used."
            )
        elif perm_category == "data":
            print(
                f"\nWarning: {app} is not expected to access your data and may pose security or privacy risks if gaining access."
            )
        elif perm_category == "collab":
            print(
                f"\nWarning: {app} are not expected to share its data and may pose security or privacy risks if allowed."
            )

    print(f"\nDetails: {action}\n")
    print("Choose permission type for this operation:")
    print("1. Allow Once")
    print("2. Allow for this Session")
    print("3. Always Allow")
    print("4. Don't Allow")
    print("=====================================\n")
    choice = input("Enter your choice: ")

    if choice == "1":
        set_permission(user_id, app, PermissionType.ONE_TIME, perm_category)
    elif choice == "2":
        set_permission(user_id, app, PermissionType.SESSION, perm_category)
    elif choice == "3":
        set_permission(user_id, app, PermissionType.PERMANENT, perm_category)
    else:
        return False

    return True


def get_user_consent(user_id, app, action, flag, perm_category="exec"):
    """
    Get user consent for a specific action, and set or check permissions.

    Args:
        user_id (str): The ID of the user.
        app (str): The name of the app.
        action (str): The action description for which permission is requested.
        flag (bool): A flag indicating additional information about the request.
        perm_category (str): The category of the permission (exec, data, collab).

    Returns:
        bool: True if user consent is granted, otherwise False.
    """
    permission_type = get_permission(user_id, app, perm_category)

    if perm_category == "exec":
        permission_obj = "Execution"
    elif perm_category == "data":
        permission_obj = "Data Access"
    elif perm_category == "collab":
        permission_obj = "Data Sharing"

    if not permission_type:
        if not request_permission(user_id, app, action, perm_category, flag):
            print(f"\n{permission_obj} Permission denied for {app}.\n")
            return False
        permission_type = get_permission(user_id, app, perm_category)

    if permission_type == PermissionType.ONE_TIME:
        print(f"\nOne-time {permission_obj} Permission granted for {app}.\n")
        set_permission(user_id, app, None, perm_category)  # Remove permission after use

    elif permission_type == PermissionType.SESSION:
        print(f"\nSession {permission_obj} Permission granted for {app}.\n")

    elif permission_type == PermissionType.PERMANENT:
        print(f"\nPermanent {permission_obj} Permission granted for {app}.\n")

    else:
        print(f"\n{permission_obj} Permission denied for {app}.\n")
        return False

    return True
