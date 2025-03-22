package src;

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

    public Instances transform(Instances data) {
        Instances BoWData = transformToBoW(data);
        Instances NonSparseBoWData = transformToBoWNonSparse(BoWData);
        return NonSparseBoWData;
    }

    private Instances transformToBoW(Instances data) {
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
}
