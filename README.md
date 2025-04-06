Exekutatuko den fitxategia: weka-project.jar

Exekutagarri honek programaren ataza guztiak betetzen dituen .jar artxiboa da. Bi parametro sartu behar zaizkio:
    - Datuen zerrenda: Kritika guztiak .txt fitxategietan dituen karpetaren helbidea, hiru azpikarpetekin; kritika positibo eta negatibotan bereizita dauden "train" eta "dev" eta klase ezezaguneko instantziak dituen "test_blind".
    - Irteeraren izena: Programak sortuko dituen fitxategiak gordetzeko erabili nahi den karpetaren helbidean. Bertan iragarpenak, nahasmen matrizeak eta 5-fold cross-validation emaitzak gordeko dira. Emaitza hauek bi modelo mota ezberdinekin aterako dira: LinearRegression eta SMO (kernel bakoitzarentzat bat) algortimoak erabilita hain zuzen ere.

Honetaz aparte, beste irteera batzuk gehituko ditu:
    - Karpeta bat .arff guztiekin. Prozesu ezberdinetan sortutako .arff izango dira, non formatuan ezberdinduko diren.

Terminaletik exekutatu behar da; guk hurrengo komandoa erabili dugu:
    $ java -jar weka-project.jar movies_reviews emaitzak

Gomendagarria da aurreko komandoan honako hau gehitzea java bateraezintasunak ekiditzeko: -add-opens java.base/java.lang=ALL-UNNAMED
    
Prozesuan minutu batzuk iraun ditzake, exekutatzeko erabili den ordenagailuaren arabera. Prozesuan zehar hainbat mezu inprimatuko dira terminalean, prozesua aurrera doala egiaztatzeko.
