def _pyscript(commands):
    script = ";".join(commands)
    return f'python -c "{script}"'

def moveTo(x, y):
    return {
        'command': _pyscript([
            'import pyautogui',
            f'pyautogui.moveTo({x}, {y})'
        ])
    }

def click():
    return {
        'command': _pyscript([
            'import pyautogui',
            f'pyautogui.click()'
        ])
    }

def rightClick():
    return {
        'command': _pyscript([
            'import pyautogui',
            f'pyautogui.rightClick()'
        ])
    }

def doubleClick():
    return {
        'command': _pyscript([
            'import pyautogui',
            f'pyautogui.doubleClick()'
        ])
    }

def position(ignore_by_mock = True):
    return {
        'command': _pyscript([
            'import pyautogui',
            'import json',
            'p = pyautogui.position()',
            "print(json.dumps({'x': p.x, 'y': p.y}))",       
        ]),
        'ignore_by_mock': ignore_by_mock
    }

def screensize(ignore_by_mock = True):
    return {
        'command': _pyscript([
            'import pyautogui',
            'import json',
            'sz = pyautogui.size()',
            "print(json.dumps({'width': sz.width, 'height': sz.height}))",       
        ]),
        'ignore_by_mock': ignore_by_mock
    }