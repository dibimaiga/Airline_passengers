import sys #any exception that will be controlled, the info will be find automaticaly in the sys 
#permet d'accéder aux informations sur les exceptions
#Sera utilisé dans la fonction error_message_detail() pour capturer les détails de l'erreur

from src.logger import logging #Checking  the logs 

def error_message_detail(error,error_detail:sys):
    """
    Rôle : Crée une fonction qui va formater un message d'erreur détaillé
Paramètres :

error : Le message d'erreur original
error_detail:sys : L'objet sys qui contient les infos sur l'exception


Relation : Cette fonction sera appelée par la classe CustomException
    """
    _,_,exc_tb=error_detail.exc_info()
#     Rôle : Récupère le traceback (la trace) de l'erreur
# Explication : exc_info() retourne 3 éléments : (type, valeur, traceback)

# Les deux premiers _,_ sont ignorés (on ne les utilise pas)
# exc_tb stocke le traceback qui contient où l'erreur s'est produite


# Relation : exc_tb sera utilisé dans les lignes suivantes

    file_name=exc_tb.tb_frame.f_code.co_filename

#     Rôle : Extrait le nom du fichier où l'erreur s'est produite
# Chemin d'accès :

# exc_tb.tb_frame : accède au frame (contexte d'exécution)
# .f_code : accède au code
# .co_filename : récupère le nom du fichier


# Relation : file_name sera utilisé dans le message formaté

    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))
    
#     Rôle : Crée un message d'erreur structuré avec :

# {0} : le nom du fichier (file_name)
# {1} : le numéro de ligne (exc_tb.tb_lineno)
# {2} : le message d'erreur original (str(error))


# Exemple de sortie : "Error occured in python script name [main.py] line number [25] error message[division by zero]"
# Relation : Ce message sera retourné et utilisé par la classe

    return error_message #Rôle : Renvoie le message d'erreur formaté
# Relation : La classe CustomException récupérera ce message

    

class CustomException(Exception):
    """
    Rôle : Crée une classe d'exception personnalisée qui hérite de Exception
Relation : Utilisera la fonction error_message_detail() pour formater les erreurs

Utilise le mécanisme natif des exceptions Python pour propager automatiquement les erreurs,
 les attraper spécifiquement, et centraliser la gestion des erreurs.
 C'est beaucoup plus puissant et propre qu'une simple fonction !

    """
    def __init__(self,error_message,error_detail:sys):
        """
        Rôle : Initialise l'exception personnalisée
Paramètres :

error_message : le message d'erreur
error_detail:sys : l'objet sys pour capturer les détails

        """
        super().__init__(error_message) #Rôle : Appelle le constructeur de la classe Exception parente
# Relation : Initialise l'exception de base avec le message

        self.error_message=error_message_detail(error_message,error_detail=error_detail)
#     Rôle : Appelle la fonction error_message_detail() pour créer un message détaillé
# Stockage : Le message formaté est stocké dans self.error_message
# Relation : Ce message sera utilisé quand on affiche l'exception

    def __str__(self):
        """
        Rôle : Définit comment l'exception sera affichée quand on la convertit en string

        """
        return self.error_message #Retourne le message d'erreur détaillé quand on fait print(exception) ou str(exception)
    
if __name__ =="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("divided by zero")
        raise CustomException(e,sys)
