#purpose of logger: Any execution that probably happen, we should be able to log all those 
# information,execution, everything in some files , so we'll be able to track if there is some errors,
#  even the custom exception error. Any exception that basically comes, we'll try to log that into the text
# file 
#Thhat's why we're creating the logger

import logging
import os
from datetime import datetime
# logging pour gérer les logs.

# os pour manipuler les chemins et dossiers.

# datetime pour obtenir la date et l’heure actuelle

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# Crée un nom de fichier basé sur la date et l’heure actuelles (ex: 12_19_2025_17_00_00.log).

# Cela permet d’avoir un fichier unique pour chaque exécution.

logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
# Construit le chemin complet vers le dossier où seront stockés les logs,
# en utilisant le répertoire courant (os.getcwd()), le sous-dossier "logs", et le nom du fichier de log.

os.makedirs(logs_path,exist_ok=True) #Crée le dossier pour les logs si celui-ci n’existe pas
# déjà (exist_ok=True évite une erreur si le dossier existe déjà).

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)
# Construit le chemin complet vers le fichier de log,
# combinant le chemin du dossier et le nom du fichier.

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO, # définit le niveau minimal de gravité des messages qui seront enregistrés ou affichés.

)

# Configure le module logging pour :

# Écrire les logs dans le fichier spécifié (LOG_FILE_PATH).

# Utiliser un format spécifique pour chaque entrée (date, numéro de ligne, nom, niveau, message).

# N’enregistrer que les messages de niveau INFO ou supérieur.

# Ce niveau est souvent utilisé en production car il donne des informations utiles sur le déroulement normal du programme, sans noyer les logs dans les détails de débogage.
# ​

# Il permet de garder un équilibre entre la lisibilité et la quantité d’informations, ce qui facilite la surveillance et le diagnostic.
# ​

# En résumé, level=logging.INFO filtre les messages pour ne garder que ceux qui sont utiles pour suivre le fonctionnement du script, en excluant les détails de débogage.

# if __name__ =="__main__":
#     logging.info("Logging started")
#     logging.warning("Attention : ceci est un avertissement")

# La structure if __name__ == "__main__": permet d’exécuter un bloc de code uniquement lorsque le script est lancé directement, et non lorsqu’il est importé comme un module dans un autre script.
# ​

# Fonctionnement
# Quand tu exécutes le script avec python nom_du_script.py, Python attribue à la variable spéciale __name__ la valeur "__main__".
# ​

# Si le script est importé dans un autre fichier (par exemple avec import nom_du_script), alors __name__ prend le nom du fichier, et le bloc de code à l’intérieur de if __name__ == "__main__": n’est pas exécuté.
