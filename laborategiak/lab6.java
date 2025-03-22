import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ArffSaver;
import java.io.File;

public class lab6 {
    public static void main(String[] args) {
        //For test
        String dataSource = "laborategiak/Datuak-20250314/1_ToyStringExample/toyStringExample_train_RAW.arff"; // Input
        String gordeleku = "laborategiak/Datuak-20250314/1_ToyStringExample/toyStringExample_train_BoW_NonSparse2.arff"; // Output

        //String dataSource = args[0];
        //String gordeleku = args[1];

        Instances data = loadData(dataSource);
        if (data == null) {
            System.out.println("Mesedez, egiaztatu helbidea ondo sartu duzula.");
            return;
        }        

        Instances bowData = transformToBoW(data);
        if (bowData == null) {
            System.out.println("Errorea: Ezin izan da Bag of Words transformazioa burutu.");
            return;
        }

        Instances nonSparseData = transformToBoWNonSparse(bowData);
        if (nonSparseData == null) {
            System.out.println("Errorea: Ezin izan da Non Sparse transformazioa burutu.");
            return;
        }

        saveData(nonSparseData, gordeleku);
    }

    private static Instances loadData(String filePath) {
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

    private static Instances transformToBoW(Instances data) {
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true); // Convertir a minúsculas
        filter.setOutputWordCounts(false); // No contar palabras, solo presencia (binario)
        filter.setTFTransform(false); // No aplicar transformación TF
        filter.setIDFTransform(false); // No aplicar transformación IDF
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

    private static Instances transformToBoWNonSparse(Instances data) {
        SparseToNonSparse filter = new SparseToNonSparse();
        try {
            filter.setInputFormat(data);
            Instances newData = Filter.useFilter(data, filter);
            return newData;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da Non Sparse transformazioa burutu.");
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