#!C:\ZWJ\code\test_matminer\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'pymatgen==2019.2.28','console_scripts','pmg'
__requires__ = 'pymatgen==2019.2.28'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('pymatgen==2019.2.28', 'console_scripts', 'pmg')()
    )
