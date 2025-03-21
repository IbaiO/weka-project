import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.core.converters.ArffSaver;

public class stratified70percentSplit {
    public static void main(String[] args) throws Exception {
        if(args.length < 3) {
            System.out.println("Ez duzu sartu beharreko parametro guztiak");
            System.exit(1);
        }
        String dataArff = args[0];
        String trainOut = args[1];
        String testBlindOut = args[2];
        
        DataSource source = new DataSource(dataArff); /// "home/mikel/Desktop/data/breast-cancer.arff"
        Instances data = source.getDataSet();

        //Klase indizea azken atributurura jarri
        if(data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Randomize the data
        Random rand = new Random(1); // Using a fixed seed for reproducibility
        
        // Stratified split: 70% train, 30% test using Resample filter
        Resample filter = new Resample();
        filter.setInputFormat(data);
        filter.setInvertSelection(false);
        filter.setNoReplacement(true);
        filter.setSampleSizePercent(70);
        filter.setBiasToUniformClass(1.0);
        filter.setRandomSeed(rand.nextInt()); // Random seed for reproducibility

        // Apply the filter
        Instances train = Filter.useFilter(data, filter);
        
        // Invert the filter to get the test set
        filter.setInvertSelection(true);
        Instances test = Resample.useFilter(data, filter);

        // Replace class attribute values in the test set with '?'
        for (int i = 0; i < test.numInstances(); i++) {
            test.instance(i).setClassMissing();
        }

        // Save the train and test sets to ARFF files
        ArffSaver saver = new ArffSaver();
        saver.setInstances(train);
        saver.setFile(new java.io.File(trainOut));
        saver.writeBatch();

        saver.setInstances(test);
        saver.setFile(new java.io.File(testBlindOut));
        saver.writeBatch();

    }
}
