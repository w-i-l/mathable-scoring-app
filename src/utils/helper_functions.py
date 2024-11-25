import os

def format_path(path):
    '''
    Formats the path to the correct format based on the operating system.
    '''

    if path == None:
        return None
    
    if os.name == 'nt':
        path = path.replace('/', '\\')
    else:
        path = path.replace('\\', '/')

    return path