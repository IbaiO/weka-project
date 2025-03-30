package src;

import java.util.regex.Pattern;
import java.io.File;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;


@SuppressWarnings("all")
public class NonSparseBoW {
    private static NonSparseBoW nireNonSparseBoW = null;

    public static NonSparseBoW getNonSparseBoW() {
        if (nireNonSparseBoW == null) {
            nireNonSparseBoW = new NonSparseBoW();
        } return nireNonSparseBoW;
    }

    public Instances transform(Instances data, String outFile) {
        Instances dataGarbi = datu_garbiketa(data);
        Instances BoWData = transformToBoW(data);
        Instances NonSparseBoWData = transformToBoWNonSparse(BoWData);
        Instances filteredData = datu_garbiketa2(NonSparseBoWData);
        save(filteredData, outFile);
        return filteredData;
    }

    private Instances datu_garbiketa(Instances datuak) {
        Pattern hashtagPattern = Pattern.compile("#\\w+");
        Pattern punctuationPattern = Pattern.compile("\\p{Punct}");

        for (int i = 0; i < datuak.numInstances(); i++) {
            Instance instance = datuak.instance(i);
            for (int j = 0; j < instance.numAttributes(); j++) {
                if (instance.attribute(j).isString()) {
                    String text = instance.stringValue(j);
                    text = text.replaceAll("\"", ""); // Komatxoak kendu
                    text = text.toLowerCase(); // Letra xehetan bihurtu
                    text = hashtagPattern.matcher(text).replaceAll(""); // Hashtagak kendu
                    text = punctuationPattern.matcher(text).replaceAll(""); // Puntuazioak kendu
                    instance.setValue(j, text);
                }
            }
        }

        return datuak;
    }

    private Instances datu_garbiketa2(Instances datuak) {
        // Use the AttributeSelection class to perform feature selection
        AttributeSelection attributeSelection = new AttributeSelection();
        CfsSubsetEval evaluator = new CfsSubsetEval();
        BestFirst search = new BestFirst();

        try {
            attributeSelection.setEvaluator(evaluator);
            attributeSelection.setSearch(search);
            attributeSelection.SelectAttributes(datuak);

            // Get the selected attributes
            int[] selectedAttributes = attributeSelection.selectedAttributes();
            System.out.println("Selected attributes: ");
            for (int i = 0; i < selectedAttributes.length; i++) {
                System.out.print(selectedAttributes[i] + " ");
            }

            // Apply the filter to the training set
            System.out.println("Number of attributes before feature selection: " + datuak.numAttributes());
            Instances filteredTrain = attributeSelection.reduceDimensionality(datuak);
            System.out.println("Number of attributes after feature selection: " + filteredTrain.numAttributes());

            return filteredTrain;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da datu garbiketa burutu.");
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    public Instances transformToBoW(Instances data) {
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true); // Letra xehez jarri testua
        filter.setOutputWordCounts(false); // Ez zenbatu hitzak, bakarrik presentzia (binarioa)
        filter.setAttributeIndices("first-last"); // Atributu guztiei aplikatu
        filter.setDoNotOperateOnPerClassBasis(true); // Ez erabili klase bakoitzeko
        filter.setTokenizer(new weka.core.tokenizers.WordTokenizer()); // Tokenizatzailea

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

    public Instances transformToBoWNonSparse(Instances data) {
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

    private void save(Instances data, String outFile) {
        try {
            // .model luzapena kendu eta BoW.arff bihurtu
            String arffOutFile = outFile.replaceAll("\\.model$", "BoW.arff");
            // Gorde datuak .arff formatuan
            weka.core.converters.ArffSaver saver = new weka.core.converters.ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(arffOutFile));
            saver.writeBatch();
            System.out.println("Transformed data saved to: " + arffOutFile);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da BoW datuak gorde.");
            e.printStackTrace();
        }
    }
}
