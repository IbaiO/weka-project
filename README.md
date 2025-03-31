# weka-project

Bat bi hiru lau, bat bi hiru lau
Bost sei zazpi, bost sei zazpi
Sanjuandarrak estropadak irabazi
AURRERA!

2025/03/21
- SW : Prototipoak libreak (arg murrizp. ez)
- BaseLine Algoritmoa: EZ!
- Proiektuko lanen banaketa, atazak zehaztasun handiz
- Rolen esleipena: Arduraduna zehaztu ataza bakoitzerako
- Algoritmoa maila teoriko labur + parm(teorik) sentikorrak -> weka ekorketa
- kfold EZ --> hold-out BAI
- 
https://www.csie.ntu.edu.tw/~cjlin/libsvm/

4 Entregatu beharreko emaitzak

Hurrengo ataletan entregatu behar diren emaitzak, alegia, fitxategien izenak eta formatuak,
zehazten dira.

4.1 Softwarea

• Softwarearen fluxu diagrama (design.pdf): Inplementatzen hasi aurretik, proiek-
tuaren diseinua egin behar da (adb. fluxu-diagrama bat). Sintetizatu softwarearen
espezifikazioak (gidoia irakurri diseinatzen hasi aurretik).

• Proiektua: workspace.tgz proiektuko iturri-kodea konprimatuta2. Esan beharrik ez
dago, programa modularra espero dela, datuekiko menpekotasunik gabea, parame-
troak komando-lerrotik jaso eta auto-informatzailea dena (alegia, espero diren argu-
mentuak jaso ezean, erabiltzaileari helburua eta beharreko argumentuez informatuko
dion salbuespena abiaraziko du). Softwarearen JavaDoc ez ahaztu klaseak dokumen-
tatu eta metodoen xehetasunak modu argian ematearren. Pakete honek izan behar
dituen funtzionalitateak 3 atalean zehaztu dira.

• Exekutagarriak: hiru exekutagarri sortu beharrean bi sor daitezke aurreprozesua eta
inferentzia batean elkartuta (weka.classifiers.meta.FilteredClassifier).
(a) Aurreprozesamendua: Preprocess.jar
(b) Inferentzia: GetModel.jar
(c) Sailkapena: Classify.jar

• Readme.txt: programa nola exekutatu, aurre-baldintzak eta ondorengo-baldintzak
labur eta zehazki azaltzen duen fitxategia testu launean. Programaren murrizpen
guztiak esplizituki azaldu, besteak beste, erabilitako liburutegien JDK-ren bertsioeki-
ko menpekotasunak, msistema eragilearekiko. . . Oso erabilgarria izaten da erabileraren
adibide bat sartzea, bertan datuak eta guzti erantziz.

4.2 Emaitza esperimentalak

ExperimentalResults.tgz paketean inplementatutako softwarearekin lortutako emaitza
esperimentalak sartuko dira:

• Ebaluazioa: ereduaren kalitatearen estimazioa jaso baseline naiz esleitutako algorit-
morako hurrengo fitxategietan hurrenez hurren: EvaluationBaseline.txt eta
EvaluationAlgorithm.txt. Fitxategi hauen edukia

– Hold-out (train vs. dev): 3.3.1. atalean aipatu den bezala, probesten da errepi-
katutako hold-out estratifikatua (avg±stdev)
– Borne errealista 10-fold Cross Validation eskemak emanda 3.3.2. atalean aipatu
den bezala. atalean aipatu den bezala).
– Kalitatearen goi bornea ebaluazio teknika ez-zintzoak emanda (3.3.2. atalean ai-
patu den bezala).

• Iragarpenak: test multzoko instantzien klasea lortu erabilitako algoritmo guztien
bidez:
– Baseline: TestPredictionsBaseline.txt
– Algoritmo: TestPredictionsAlgorithm.txt
