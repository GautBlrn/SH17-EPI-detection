"""Rend `app/conformite.py` importable par les tests.

L'application est un script Streamlit, pas un paquet installable : `app/` n'est
pas sur le chemin d'import quand pytest démarre depuis `Applications/`. Ce
fichier l'y met, et c'est tout ce qu'il fait.
"""

import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / "app").resolve()))
