import os

def format_path(path):
    '''
    Formats the path to the correct format based on the operating system.
    Also verifies if the path exists.
    '''

    if path == None:
        return None
    
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found at {path}")
    return path