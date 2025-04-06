Exekutatuko den fitxategia: weka-project.jar

Exekutagarri honek programaren ataza guztiak betetzen dituen .jar artxiboa da. Bi parametro sartu behar zaizkio:
    - Datuen zerrenda: Kritika guztiak .txt fitxategietan dituen karpetaren helbidea, hiru azpikarpetekin; kritika positibo eta negatibotan bereizita dauden "train" eta "dev" eta klase ezezaguneko instantziak dituen "test_blind".
    - Irteeraren izena: Programak sortuko dituen fitxategiak gordetzeko erabili nahi den karpetaren helbidea. Bertan iragarpenak, nahasmen matrizeak eta 5-fold cross-validation emaitzak gordeko dira.

Terminaletik exekutatu behar da; guk hurrengo komandoa erabili dugu:
    $ java -jar weka-project.jar movies_reviews emaitzak
    
Prozesuan minutu batzuk iraun ditzake, exekutatzeko erabili den ordenagailuaren arabera. Prozesuan zehar hainbat mezu inprimatuko dira terminalean, prozesua aurrera doala egiaztatzeko.