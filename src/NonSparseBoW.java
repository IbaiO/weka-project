package src;

import java.util.regex.Pattern;
import java.io.File;
import java.io.FileWriter;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
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

    public Instances transformTrain(Instances data) {
        Instances dataGarbi = datu_garbiketa(data);
        Instances BoWData = transformToBoW(data);
        Instances NonSparseBoWData = transformToBoWNonSparse(BoWData);
        Instances filteredData = filteredAttributes(NonSparseBoWData);
        return filteredData;
    }

    public Instances transformDevTest (Instances data) {
        Instances BoWData = transformToBoW(data);
        Instances NonSparseBoW = transformToBoWNonSparse(BoWData);
        return NonSparseBoW;
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

    private Instances filteredAttributes(Instances datuak) {
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

    private Instances filterAttributesByInfoGain(Instances datuak) {
        InfoGainAttributeEval evaluator = new InfoGainAttributeEval(); // Use InfoGain evaluator
        try {
            // Get the InfoGain values for each attribute
            double[] infoGainValues = new double[datuak.numAttributes()];
            for (int i = 0; i < datuak.numAttributes(); i++) {
                infoGainValues[i] = evaluator.evaluateAttribute(i);
            }

            // Print InfoGain values
            System.out.println("InfoGain values for attributes:");
            for (int i = 0; i < infoGainValues.length; i++) {
                System.out.println("Attribute " + i + ": " + infoGainValues[i]);
            }

            // Filter attributes with InfoGain > 0
            Instances filteredTrain = new Instances(datuak);
            for (int i = infoGainValues.length - 1; i >= 0; i--) {
                if (infoGainValues[i] == 0) {
                    filteredTrain.deleteAttributeAt(i);
                }
            }

            System.out.println("Number of attributes before filtering: " + datuak.numAttributes());
            System.out.println("Number of attributes after filtering: " + filteredTrain.numAttributes());

            return filteredTrain;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da InfoGain bidezko atributuen iragazketa burutu.");
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

    private Instances transformToBoWNonSparse(Instances data) {
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

    public Instances transformDevTest(Instances data, String[] attributes){
        Instances datuak = datu_garbiketa(data);
        // Iragazi atributuak
        Instances filteredData = new Instances(datuak);
        for (int i = filteredData.numAttributes() - 1; i >= 0; i--) {
            boolean keep = false;
            for (String attribute : attributes) {
                if (filteredData.attribute(i).name().equals(attribute)) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                filteredData.deleteAttributeAt(i);
            }
        }
        Instances BoWData = transformToBoW(filteredData);
        Instances NonSparseBoWData = transformToBoWNonSparse(BoWData);
        return NonSparseBoWData;
    }

    public static void main(String[] args) {
        try {
            DataSource source = new DataSource("datuak/datuakTrain.arff");
            Instances data = source.getDataSet();
            NonSparseBoW.getNonSparseBoW().transformTrain(data);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da datu multzoa kargatu.");
            e.printStackTrace();
        }
    }
}
