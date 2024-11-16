import os


def format_path(path):
    if path == None:
        return None
    
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')
    return path