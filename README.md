📥 Acquisition des Données

Pour faire fonctionner ce projet, vous devez récupérer les données officielles de l'offre de transport (format GTFS) :

    Source : Portail Open Data Île-de-France Mobilités

    Téléchargement : Cliquez sur l'onglet "Export" puis téléchargez l'archive IDFM-gtfs.zip (environ 500 Mo compressé, > 2 Go décompressé).

    Installation :

        À la racine de ce projet, créez un dossier nommé data/.

        Extrayez le contenu de l'archive .zip directement dans ce dossier data/.

        Vérifiez que les fichiers comme stops.txt et stop_times.txt sont bien présents à l'adresse ./data/stops.txt.

    Note : Le dossier data/ est suprimé afin d'éviter l'envoi de fichiers volumineux sur le dépôt GitHub, conformément aux limites de taille de fichier de la plateforme.
