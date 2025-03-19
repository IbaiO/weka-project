import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import java.io.File;

public class lab6 {
    public static void main(String[] args) {
        //For test
        String dataSource = "Datuak-20250314/1_ToyStringExample/toyStringExample_train_RAW.arff"; // Input
        String gordeleku = "Datuak-20250314/1_ToyStringExample/toyStringExample_train_BoW2.arff"; // Output

        //String dataSource = args[0];
        //String gordeleku = args[1];

        Instances data = loadData(dataSource);
        if (data == null) {
            System.out.println("Mesedez, egiaztatu helbidea ondo sartu duzula.");
            return;
        }        

        Instances bowData = transformToBagOfWords(data);
        if (bowData == null) {
            System.out.println("Errorea: Ezin izan da Bag of Words transformazioa burutu.");
            return;
        }

        saveData(bowData, gordeleku);
    }

    private static Instances loadData(String filePath) { // Oso totxo, igual ez dira behar horrenbeste if
        DataSource source = null;
        try {
            source = new DataSource(filePath);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategia aurkitu.");
            return null;
        }
        Instances data = null;
        try {
            data = source.getDataSet();
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izandu da " + filePath + " fitxategia irakurri.");
            return null;
        }

        if (data == null) {
            System.out.println("ERROREA: " + filePath + " fitxategiaren edukia hutsik dago.");
            return null;
        }

        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        System.out.println("Class index set to: " + data.classIndex());
        return data;
    }

    private static Instances transformToBagOfWords(Instances data) {
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true); // Convertir a minúsculas
        filter.setOutputWordCounts(true); // Contar palabras
        filter.setTFTransform(true); // Transformación TF
        filter.setIDFTransform(true); // Transformación IDF
        filter.setAttributeIndices("first-last"); // Aplicar a todos los atributos
        filter.setWordsToKeep(1000); // Número de palabras a mantener
        filter.setDoNotOperateOnPerClassBasis(true); // No operar por clase
        filter.setStemmer(null); // No usar stemmer
        filter.setTokenizer(new weka.core.tokenizers.WordTokenizer()); // Tokenizador por defecto

        try {
            filter.setInputFormat(data);
            Instances newData = Filter.useFilter(data, filter);
            return newData;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da Bag of Words transformazioa burutu.");
            e.printStackTrace();
            return null;
        }
    }

    private static void saveData(Instances data, String filePath) {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        try {
            saver.setFile(new File(filePath));
            saver.writeBatch();
            System.out.println("Datuak gorde dira: " + filePath);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da " + filePath + " fitxategia gorde.");
            e.printStackTrace();
        }
    }
}